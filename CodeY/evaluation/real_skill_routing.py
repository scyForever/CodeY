"""Real-model evaluation for Skill selection at increasing catalog sizes."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..config import load_project_env, provider_env
from ..providers.clients import (
    AnthropicCompatibleModelClient,
    ModelCompletion,
    OpenAICompatibleModelClient,
)


SCHEMA_VERSION = 1
DEFAULT_SCALES = (5, 15, 25, 50, 100)
MODES = ("flat_full", "structured_index")
QUOTED_TRIGGER_RE = re.compile(r'"([^"\r\n]+)"')
DESCRIPTION_NEAR_MISS_RE = re.compile(
    r"It should not activate for (?P<first>[^,]+), (?P<second>[^,]+), or (?P<third>[^.]+)\.\s*$"
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
WORKFLOW_ENUMERATION_RE = re.compile(
    r"\b(?:first|second|third|then|next|finally|step\s+\d+)\b",
    flags=re.IGNORECASE,
)


class RealSkillEvaluationError(ValueError):
    """Raised when the real-Skill benchmark or a model response is invalid."""


def merge_artifact_axes(payload, scales, modes):
    """Keep artifact axes truthful when an evaluation resumes in scale slices."""
    payload["skill_scales"] = sorted(
        set(payload.get("skill_scales", ())) | set(scales)
    )
    requested_modes = set(payload.get("modes", ())) | set(modes)
    payload["modes"] = [mode for mode in MODES if mode in requested_modes]


@dataclass(frozen=True)
class CategorySpec:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class SkillSpec:
    id: str
    category: str
    description: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class RequestCase:
    id: str
    text: str
    language: str
    kind: str
    expected_skills: tuple[str, ...]
    anchor: bool = False


@dataclass(frozen=True)
class BenchmarkDataset:
    categories: tuple[CategorySpec, ...]
    skills: tuple[SkillSpec, ...]
    requests: tuple[RequestCase, ...]
    scales: tuple[int, ...] = DEFAULT_SCALES


def _load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealSkillEvaluationError(f"cannot read benchmark JSON {path}: {exc}") from exc


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise RealSkillEvaluationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_string_list(value, field, allow_empty=False):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RealSkillEvaluationError(f"{field} must be a string array")
    if not allow_empty and not value:
        raise RealSkillEvaluationError(f"{field} must not be empty")
    normalized = tuple(dict.fromkeys(item.strip() for item in value))
    if len(normalized) != len(value):
        raise RealSkillEvaluationError(f"{field} contains duplicates")
    return normalized


def _normalized_trigger_phrase(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _description_rule_violations(skill):
    description = skill.description
    violations = []
    english_words = ENGLISH_WORD_RE.findall(description)
    cjk_characters = CJK_RE.findall(description)
    if len(english_words) < 20:
        violations.append("must contain at least 20 English words")
    if cjk_characters:
        violations.append("must be English-only and contain no CJK characters")
    if not description.startswith("This skill should be used when"):
        violations.append(
            "must start with the third-person activation frame This skill should be used when"
        )
    if "when" not in description[:50].casefold():
        violations.append("must state the activation context and core intent")

    phrases = tuple(match.group(1).strip() for match in QUOTED_TRIGGER_RE.finditer(description))
    if len(phrases) != 2:
        violations.append("must contain exactly two quoted trigger phrases")
    else:
        if any(CJK_RE.search(phrase) for phrase in phrases):
            violations.append("quoted trigger phrases must be English-only")
        if any(not re.search(r"[A-Za-z]", phrase) for phrase in phrases):
            violations.append("both quoted trigger phrases must be English")
        for phrase in phrases:
            english_length = len(ENGLISH_WORD_RE.findall(phrase))
            if english_length < 3:
                violations.append(f"trigger phrase is too generic: {phrase}")
            if any(separator in phrase for separator in (",", "，", "、", "/")):
                violations.append(f"trigger phrase looks like a keyword list: {phrase}")
        if len({_normalized_trigger_phrase(phrase) for phrase in phrases}) != 2:
            violations.append("quoted trigger phrases must be unique")

    near_miss = DESCRIPTION_NEAR_MISS_RE.search(description)
    if not near_miss:
        violations.append("must end with three near-miss non-activation examples")
    else:
        values = tuple(near_miss.group(name).strip() for name in ("first", "second", "third"))
        if any(not value for value in values) or len({_normalized_trigger_phrase(value) for value in values}) != 3:
            violations.append("near-miss boundary must name three distinct examples")
        if {
            _normalized_trigger_phrase(value)
            for value in values
        } & {_normalized_trigger_phrase(phrase) for phrase in phrases}:
            violations.append("near-miss examples must not repeat trigger phrases")
    if WORKFLOW_ENUMERATION_RE.search(description):
        violations.append("must not enumerate workflow steps")
    return phrases, violations


def validate_skill_descriptions(skills):
    phrase_owners = {}
    errors = []
    for skill in skills:
        phrases, violations = _description_rule_violations(skill)
        if violations:
            errors.append(f"{skill.id}: " + "; ".join(violations))
        for phrase in phrases:
            normalized = _normalized_trigger_phrase(phrase)
            previous = phrase_owners.get(normalized)
            if previous is not None:
                errors.append(
                    f'{skill.id}: trigger phrase "{phrase}" duplicates {previous}'
                )
            else:
                phrase_owners[normalized] = skill.id
    if errors:
        raise RealSkillEvaluationError(
            "Skill description rules failed:\n- " + "\n- ".join(errors)
        )
    return tuple(phrase_owners)


def load_benchmark_dataset(skills_path, requests_path):
    skills_payload = _load_json(skills_path)
    requests_payload = _load_json(requests_path)
    if skills_payload.get("schema_version") != SCHEMA_VERSION:
        raise RealSkillEvaluationError("unsupported real-Skill catalog schema")
    if requests_payload.get("schema_version") != SCHEMA_VERSION:
        raise RealSkillEvaluationError("unsupported real-Skill request schema")

    raw_scales = skills_payload.get("scales")
    if raw_scales != list(DEFAULT_SCALES):
        raise RealSkillEvaluationError(
            f"catalog scales must be exactly {list(DEFAULT_SCALES)}"
        )

    categories = []
    category_ids = set()
    for index, raw in enumerate(skills_payload.get("categories", [])):
        if not isinstance(raw, dict):
            raise RealSkillEvaluationError(f"category {index} must be an object")
        category = CategorySpec(
            id=_require_text(raw.get("id"), f"category {index}.id"),
            label=_require_text(raw.get("label"), f"category {index}.label"),
            description=_require_text(
                raw.get("description"), f"category {index}.description"
            ),
        )
        if category.id in category_ids:
            raise RealSkillEvaluationError(f"duplicate category id: {category.id}")
        category_ids.add(category.id)
        categories.append(category)
    if len(categories) != 10:
        raise RealSkillEvaluationError("real-Skill catalog must define 10 categories")

    skills = []
    skill_ids = set()
    for index, raw in enumerate(skills_payload.get("skills", [])):
        if not isinstance(raw, dict):
            raise RealSkillEvaluationError(f"skill {index} must be an object")
        skill = SkillSpec(
            id=_require_text(raw.get("id"), f"skill {index}.id"),
            category=_require_text(raw.get("category"), f"skill {index}.category"),
            description=_require_text(
                raw.get("description"), f"skill {index}.description"
            ),
            capabilities=_require_string_list(
                raw.get("capabilities"), f"skill {index}.capabilities"
            ),
        )
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill.id):
            raise RealSkillEvaluationError(f"invalid skill id: {skill.id}")
        if skill.id in skill_ids:
            raise RealSkillEvaluationError(f"duplicate skill id: {skill.id}")
        if skill.category not in category_ids:
            raise RealSkillEvaluationError(
                f"skill {skill.id} references unknown category {skill.category}"
            )
        skill_ids.add(skill.id)
        skills.append(skill)
    if len(skills) != 100:
        raise RealSkillEvaluationError("real-Skill catalog must define exactly 100 skills")
    validate_skill_descriptions(skills)

    requests = []
    request_ids = set()
    valid_kinds = {"single", "multi", "negative"}
    for index, raw in enumerate(requests_payload.get("requests", [])):
        if not isinstance(raw, dict):
            raise RealSkillEvaluationError(f"request {index} must be an object")
        expected = _require_string_list(
            raw.get("expected_skills"),
            f"request {index}.expected_skills",
            allow_empty=True,
        )
        request = RequestCase(
            id=_require_text(raw.get("id"), f"request {index}.id"),
            text=_require_text(raw.get("text"), f"request {index}.text"),
            language=_require_text(raw.get("language"), f"request {index}.language"),
            kind=_require_text(raw.get("kind"), f"request {index}.kind"),
            expected_skills=expected,
            anchor=bool(raw.get("anchor", False)),
        )
        if request.id in request_ids:
            raise RealSkillEvaluationError(f"duplicate request id: {request.id}")
        if request.kind not in valid_kinds:
            raise RealSkillEvaluationError(f"request {request.id} has invalid kind")
        if any(skill_id not in skill_ids for skill_id in request.expected_skills):
            raise RealSkillEvaluationError(
                f"request {request.id} references an unknown skill"
            )
        if request.kind == "single" and len(request.expected_skills) != 1:
            raise RealSkillEvaluationError(
                f"single request {request.id} must reference one skill"
            )
        if request.kind == "multi" and len(request.expected_skills) < 2:
            raise RealSkillEvaluationError(
                f"multi request {request.id} must reference at least two skills"
            )
        if request.kind == "negative" and request.expected_skills:
            raise RealSkillEvaluationError(
                f"negative request {request.id} must not reference a skill"
            )
        request_ids.add(request.id)
        requests.append(request)

    single_targets = [
        request.expected_skills[0] for request in requests if request.kind == "single"
    ]
    if len(single_targets) != 100 or set(single_targets) != skill_ids:
        raise RealSkillEvaluationError(
            "request set must contain exactly one single-Skill request per Skill"
        )
    anchor_single = [request for request in requests if request.kind == "single" and request.anchor]
    if len(anchor_single) != 5:
        raise RealSkillEvaluationError("request set must define five anchor single-Skill cases")
    if not any(request.kind == "multi" for request in requests):
        raise RealSkillEvaluationError("request set must include multi-Skill cases")
    if not any(request.kind == "negative" for request in requests):
        raise RealSkillEvaluationError("request set must include negative cases")
    return BenchmarkDataset(
        categories=tuple(categories),
        skills=tuple(skills),
        requests=tuple(requests),
    )


def dataset_fingerprint(skills_path, requests_path):
    digest = hashlib.sha256()
    for path in (Path(skills_path), Path(requests_path)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cases_for_scale(dataset, scale):
    if scale not in dataset.scales:
        raise RealSkillEvaluationError(f"unsupported Skill scale: {scale}")
    active = {skill.id for skill in dataset.skills[:scale]}
    return tuple(
        case
        for case in dataset.requests
        if not case.expected_skills or set(case.expected_skills).issubset(active)
    )


def _category_map(dataset):
    return {category.id: category for category in dataset.categories}


def render_flat_catalog(dataset, scale):
    categories = _category_map(dataset)
    sections = []
    for skill in dataset.skills[:scale]:
        category = categories[skill.category]
        capabilities = "\n".join(f"- {item}" for item in skill.capabilities)
        sections.append(
            f"<skill id=\"{skill.id}\">\n"
            f"Name: {skill.id}\n"
            f"Category: {category.label}\n"
            f"Purpose: {skill.description}\n"
            f"Responsibilities:\n{capabilities}\n"
            "Operating guidance:\n"
            "1. Inspect the request and repository evidence before acting.\n"
            "2. Apply the smallest change that satisfies this specialty.\n"
            "3. Verify the specialty-specific outcome and report limitations.\n"
            "Boundary: select this Skill only when the request materially needs its stated responsibilities.\n"
            "</skill>"
        )
    return "\n\n".join(sections)


def render_skill_document(dataset, skill):
    """Render a benchmark Skill in the same shape as a project SKILL.md."""
    category = _category_map(dataset)[skill.category]
    responsibilities = "\n".join(f"- {item}" for item in skill.capabilities)
    return (
        "---\n"
        f"name: {skill.id}\n"
        f"description: {skill.description}\n"
        "---\n\n"
        "<always-applicable>\n"
        f"Domain: {category.label}\n"
        f"Responsibilities:\n{responsibilities}\n"
        "</always-applicable>\n\n"
        "<task-routing>\n"
        "```json\n"
        '{"tasks":[{"id":"primary","label":"Primary workflow","triggers":[],"reads":[]},'
        '{"id":"other","label":"Other","triggers":[],"reads":[]}]}\n'
        "```\n"
        "</task-routing>\n"
    )


def materialize_skill_documents(dataset, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for skill in dataset.skills:
        skill_dir = output_root / skill.id
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            render_skill_document(dataset, skill), encoding="utf-8"
        )
    return output_root


def render_structured_catalog(dataset, scale):
    grouped = {category.id: [] for category in dataset.categories}
    for skill in dataset.skills[:scale]:
        grouped[skill.category].append(skill)
    sections = []
    for category in dataset.categories:
        skills = grouped[category.id]
        if not skills:
            continue
        lines = [
            f"## {category.label} [{category.id}]",
            category.description,
        ]
        for skill in skills:
            capabilities = "; ".join(skill.capabilities)
            lines.append(
                f"- {skill.id}: {skill.description} Use for: {capabilities}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_selection_prompt(dataset, scale, mode, cases, seed=20260712):
    if mode not in MODES:
        raise RealSkillEvaluationError(f"unsupported evaluation mode: {mode}")
    if not cases:
        raise RealSkillEvaluationError("evaluation batch must not be empty")
    catalog = (
        render_flat_catalog(dataset, scale)
        if mode == "flat_full"
        else render_structured_catalog(dataset, scale)
    )
    shuffled_cases = list(cases)
    random.Random(seed + scale).shuffle(shuffled_cases)
    request_rows = "\n".join(
        f'- {json.dumps(case.id, ensure_ascii=False)}: '
        f'{json.dumps(case.text, ensure_ascii=False)}'
        for case in shuffled_cases
    )
    valid_ids = ", ".join(skill.id for skill in dataset.skills[:scale])
    return (
        "You are a Skill router in a controlled evaluation. Select every Skill that is materially required "
        "to handle each request. A request may need zero, one, or multiple Skills. Do not select a Skill "
        "merely because the request contains a related word.\n\n"
        "Return one JSON object and no prose or Markdown. The exact schema is:\n"
        '{"predictions":[{"id":"request-id","skills":["skill-id"]}]}\n'
        "Include every request exactly once. Use [] when no listed Skill applies. Use only exact Skill IDs.\n\n"
        f"Valid Skill IDs ({scale}): {valid_ids}\n\n"
        f"Candidate Skill catalog ({mode}):\n{catalog}\n\n"
        f"Requests:\n{request_rows}"
    )


def _extract_json_value(text):
    text = str(text or "").strip()
    if not text:
        raise RealSkillEvaluationError("model returned an empty response")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise RealSkillEvaluationError("model response does not contain valid JSON")


def parse_predictions(text, cases):
    value = _extract_json_value(text)
    if isinstance(value, list):
        predictions = value
    elif isinstance(value, dict):
        predictions = value.get("predictions")
    else:
        predictions = None
    if not isinstance(predictions, list):
        raise RealSkillEvaluationError("model JSON requires a predictions array")

    expected_ids = {case.id for case in cases}
    parsed = {}
    for index, raw in enumerate(predictions):
        if not isinstance(raw, dict):
            raise RealSkillEvaluationError(f"prediction {index} must be an object")
        request_id = raw.get("id")
        skills = raw.get("skills")
        if request_id not in expected_ids:
            raise RealSkillEvaluationError(
                f"prediction {index} has unknown request id: {request_id}"
            )
        if request_id in parsed:
            raise RealSkillEvaluationError(f"duplicate prediction id: {request_id}")
        if not isinstance(skills, list) or any(not isinstance(item, str) for item in skills):
            raise RealSkillEvaluationError(
                f"prediction {request_id} skills must be a string array"
            )
        parsed[request_id] = tuple(dict.fromkeys(item.strip() for item in skills if item.strip()))
    missing = sorted(expected_ids - set(parsed))
    if missing:
        raise RealSkillEvaluationError(
            "model omitted request predictions: " + ", ".join(missing)
        )
    return parsed


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metric(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": _ratio(numerator, denominator),
    }


def score_predictions(cases, predictions, active_skill_ids):
    rows = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    active_skill_ids = set(active_skill_ids)
    for case in cases:
        expected = set(case.expected_skills)
        predicted = set(predictions.get(case.id, ()))
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        rows.append(
            {
                "id": case.id,
                "text": case.text,
                "language": case.language,
                "kind": case.kind,
                "anchor": case.anchor,
                "expected_skills": sorted(expected),
                "predicted_skills": sorted(predicted),
                "unknown_skills": sorted(predicted - active_skill_ids),
                "exact_match": expected == predicted,
                "recall": _ratio(tp, len(expected)) if expected else None,
            }
        )

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    anchors = [row for row in rows if row["anchor"]]
    singles = [row for row in rows if row["kind"] == "single"]
    multis = [row for row in rows if row["kind"] == "multi"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    return {
        "case_count": len(rows),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_match": _metric(sum(row["exact_match"] for row in rows), len(rows)),
        "anchor_exact_match": _metric(
            sum(row["exact_match"] for row in anchors), len(anchors)
        ),
        "single_skill_accuracy": _metric(
            sum(row["exact_match"] for row in singles), len(singles)
        ),
        "multi_skill_exact_match": _metric(
            sum(row["exact_match"] for row in multis), len(multis)
        ),
        "negative_accuracy": _metric(
            sum(row["exact_match"] for row in negatives), len(negatives)
        ),
        "false_activations": false_positive,
        "missed_skills": false_negative,
        "rows": rows,
    }


def _external_provider_profile(workspace_root, provider=None, model_override=None):
    load_project_env(workspace_root)
    provider = str(provider or provider_env("CODEY_PROVIDER", default="")).strip().lower()
    if provider not in {"openai", "anthropic", "deepseek"}:
        raise RealSkillEvaluationError(
            "CODEY_PROVIDER must select openai, anthropic, or deepseek for this external-model evaluation"
        )
    if provider == "openai":
        api_key = provider_env(
            "CODEY_OPENAI_API_KEY",
            (
                "OPENAI_API_KEY",
                "CODEY_RIGHT_CODES_API_KEY",
                "RIGHT_CODES_API_KEY",
                "CODEY_ANTHROPIC_API_KEY",
                "ANTHROPIC_API_KEY",
            ),
        )
        model = provider_env("CODEY_OPENAI_MODEL", ("OPENAI_MODEL",), "gpt-5.4")
        base_url = provider_env(
            "CODEY_OPENAI_API_BASE", ("OPENAI_API_BASE",), "https://api.openai.com/v1"
        )
        client_type = "openai"
    elif provider == "anthropic":
        api_key = provider_env(
            "CODEY_ANTHROPIC_API_KEY",
            (
                "ANTHROPIC_API_KEY",
                "CODEY_RIGHT_CODES_API_KEY",
                "RIGHT_CODES_API_KEY",
                "CODEY_OPENAI_API_KEY",
                "OPENAI_API_KEY",
            ),
        )
        model = provider_env(
            "CODEY_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",), "claude-sonnet-4-6"
        )
        base_url = provider_env(
            "CODEY_ANTHROPIC_API_BASE",
            ("ANTHROPIC_API_BASE",),
            "https://api.anthropic.com/v1",
        )
        client_type = "anthropic"
    else:
        api_key = provider_env("CODEY_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        model = provider_env(
            "CODEY_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",), "deepseek-v4-pro"
        )
        base_url = provider_env(
            "CODEY_DEEPSEEK_API_BASE",
            ("DEEPSEEK_API_BASE",),
            "https://api.deepseek.com/anthropic",
        )
        client_type = "anthropic"
    if model_override:
        model = str(model_override).strip()
    if not api_key:
        raise RealSkillEvaluationError(f"{provider} API key is not configured in .env")
    if not model:
        raise RealSkillEvaluationError(f"{provider} model is not configured")
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "client_type": client_type,
    }


def build_external_model_client(
    workspace_root,
    provider=None,
    model_override=None,
    timeout=300,
    transport_attempts=1,
):
    profile = _external_provider_profile(
        workspace_root,
        provider=provider,
        model_override=model_override,
    )
    if profile["client_type"] == "openai":
        client = OpenAICompatibleModelClient(
            model=profile["model"],
            base_url=profile["base_url"],
            api_key=profile["api_key"],
            temperature=0.0,
            timeout=timeout,
            attempts=transport_attempts,
        )
    else:
        client = AnthropicCompatibleModelClient(
            model=profile["model"],
            base_url=profile["base_url"],
            api_key=profile["api_key"],
            temperature=0.0,
            timeout=timeout,
            attempts=transport_attempts,
        )
    public_profile = {key: value for key, value in profile.items() if key != "api_key"}
    return client, public_profile


def run_model_batch(client, prompt, cases, max_output_tokens=8192, parse_retries=1):
    attempts = []
    current_prompt = prompt
    for attempt in range(parse_retries + 1):
        started = time.perf_counter()
        completion = client.complete(current_prompt, max_new_tokens=max_output_tokens)
        if not isinstance(completion, ModelCompletion):
            raise TypeError("routing evaluation clients must return ModelCompletion")
        response = completion.text
        elapsed = time.perf_counter() - started
        metadata = dict(completion.metadata)
        record = {
            "attempt": attempt + 1,
            "elapsed_seconds": elapsed,
            "response_chars": len(response),
            "provider_metadata": metadata,
        }
        try:
            predictions = parse_predictions(response, cases)
        except RealSkillEvaluationError as exc:
            record["parse_error"] = str(exc)
            record["raw_response"] = response
            attempts.append(record)
            if attempt >= parse_retries:
                raise
            current_prompt = (
                prompt
                + "\n\nYour previous response was invalid: "
                + str(exc)
                + " Return the complete JSON object again, with no prose."
            )
            continue
        record["raw_response"] = response
        attempts.append(record)
        return predictions, attempts
    raise RealSkillEvaluationError("model prediction parsing failed")


def _format_percentage(value):
    return "N/A" if value is None else f"{value:.2%}"


def _attempt_total(run, key):
    return sum(
        int(attempt.get("provider_metadata", {}).get(key) or 0)
        for attempt in run.get("attempts", [])
    )


def render_markdown_report(payload):
    lines = [
        "# Real-Model Skill Routing Evaluation",
        "",
        f"- Provider: `{payload['provider']['provider']}`",
        f"- Model: `{payload['provider']['model']}`",
        f"- Dataset SHA-256: `{payload['dataset_sha256']}`",
        f"- Repetitions: {payload['repetitions']}",
        "",
        "| Skills | Mode | Cases | Exact | Anchor exact | Single | Multi | Negative | Micro F1 | Input tokens | Parse retries | Prompt chars | Seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        score = run["score"]
        lines.append(
            "| {scale} | `{mode}` | {cases} | {exact} | {anchor} | {single} | {multi} | {negative} | {f1} | {tokens} | {retries} | {chars} | {seconds:.2f} |".format(
                scale=run["scale"],
                mode=run["mode"],
                cases=score["case_count"],
                exact=_format_percentage(score["exact_match"]["value"]),
                anchor=_format_percentage(score["anchor_exact_match"]["value"]),
                single=_format_percentage(score["single_skill_accuracy"]["value"]),
                multi=_format_percentage(score["multi_skill_exact_match"]["value"]),
                negative=_format_percentage(score["negative_accuracy"]["value"]),
                f1=_format_percentage(score["micro_f1"]),
                tokens=_attempt_total(run, "input_tokens"),
                retries=sum(
                    1 for attempt in run.get("attempts", []) if attempt.get("parse_error")
                ),
                chars=run["prompt_chars"],
                seconds=run["elapsed_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## Structured Minus Flat",
            "",
            "Positive accuracy/F1 values favor the structured index. Positive prompt reduction means the structured index used less prompt text.",
            "",
            "| Skills | Exact delta | Anchor delta | Micro F1 delta | Prompt reduction | Time delta seconds |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in sorted(
        payload.get("comparisons", []),
        key=lambda item: (item["scale"], item["repetition"]),
    ):
        delta = comparison["structured_minus_flat"]
        flat_run = next(
            run
            for run in payload["runs"]
            if run["scale"] == comparison["scale"]
            and run["repetition"] == comparison["repetition"]
            and run["mode"] == "flat_full"
        )
        prompt_reduction = (
            -delta["prompt_chars"] / flat_run["prompt_chars"]
            if flat_run["prompt_chars"]
            else None
        )
        lines.append(
            "| {scale} | {exact} | {anchor} | {f1} | {prompt} | {seconds:+.2f} |".format(
                scale=comparison["scale"],
                exact=_format_percentage(delta["exact_match"]),
                anchor=_format_percentage(delta["anchor_exact_match"]),
                f1=_format_percentage(delta["micro_f1"]),
                prompt=_format_percentage(prompt_reduction),
                seconds=delta["elapsed_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- This experiment measures model Skill selection from a controlled catalog, not task execution quality.",
            "- `flat_full` receives every full Skill definition; `structured_index` receives the same essential capabilities in a grouped compact index.",
            "- Anchor metrics use the same five single-Skill requests at every scale to isolate catalog-size degradation.",
            "- Raw responses and per-request predictions remain in the JSON artifact for audit.",
            "",
        ]
    )
    return "\n".join(lines)


def canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_artifacts(payload, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(canonical_json(payload), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")
