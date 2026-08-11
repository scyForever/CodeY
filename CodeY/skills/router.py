"""Project-local skills with deterministic, progressive routing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr

MAX_SKILL_BYTES = 128 * 1024
MAX_READ_BYTES = 128 * 1024
FRONTMATTER_KEYS = {"name", "description"}
DESCRIPTION_PREFIX = "This skill should be used when"
DESCRIPTION_NEAR_MISS_RE = re.compile(
    r"It should not activate for (?P<first>[^,]+), (?P<second>[^,]+), or (?P<third>[^.]+)\.\s*$"
)
QUOTED_TRIGGER_RE = re.compile(r'"([^"\r\n]+)"')
WORKFLOW_STEP_RE = re.compile(r"(?i)\b(first|second|third|then|next|finally|step\s+\d+)\b")
MIN_SELECTOR_CONFIDENCE = 0.5


class SkillConfigurationError(ValueError):
    """Raised when a discovered skill is present but invalid."""


@dataclass(frozen=True)
class TaskRoute:
    id: str
    label: str
    triggers: tuple[str, ...]
    reads: tuple[str, ...]
    workflow: str = ""


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    activation_phrases: tuple[str, ...]
    near_misses: tuple[str, ...]
    path: Path
    body: str
    always_text: str
    always_reads: tuple[str, ...]
    routes: tuple[TaskRoute, ...]


@dataclass(frozen=True)
class SkillCandidate:
    skill_name: str
    activation_phrases: tuple[str, ...]
    near_misses: tuple[str, ...]
    matched_activation_phrases: tuple[str, ...] = ()
    matched_near_misses: tuple[str, ...] = ()
    name_matched: bool = False
    lexical_score: int = 0

    def to_dict(self, selected=False):
        return {
            "skill_name": self.skill_name,
            "activation_phrases": list(self.activation_phrases),
            "near_misses": list(self.near_misses),
            "matched_activation_phrases": list(self.matched_activation_phrases),
            "matched_near_misses": list(self.matched_near_misses),
            "name_matched": self.name_matched,
            "lexical_score": self.lexical_score,
            "selected": bool(selected),
        }


@dataclass(frozen=True)
class SkillSelection:
    skill_name: str = ""
    confidence: float = 0.0
    reason: str = ""
    source: str = "model"


@dataclass(frozen=True)
class RouteMatch:
    routing_event_id: str = ""
    skill_name: str = ""
    route_id: str = ""
    fallback: bool = False
    matched_terms: tuple[str, ...] = ()
    loaded_paths: tuple[str, ...] = ()
    route_context: str = ""
    selection_source: str = "none"
    selector_confidence: float = 0.0
    selector_reason: str = ""
    candidates: tuple[SkillCandidate, ...] = ()

    def to_dict(self):
        return {
            "routing_event_id": self.routing_event_id,
            "skill_name": self.skill_name,
            "route_id": self.route_id,
            "fallback": self.fallback,
            "matched_terms": list(self.matched_terms),
            "loaded_paths": list(self.loaded_paths),
            "selection_source": self.selection_source,
            "selector_confidence": self.selector_confidence,
            "selector_reason": self.selector_reason,
            "candidates": [
                candidate.to_dict(selected=candidate.skill_name == self.skill_name)
                for candidate in self.candidates
            ],
        }


@dataclass(frozen=True)
class SkillContext:
    text: str = ""
    fingerprint: str = ""
    skill_names: tuple[str, ...] = ()
    loaded_paths: tuple[str, ...] = ()


class SkillRouter:
    def __init__(self, workspace_root, mode="auto"):
        self.root = Path(workspace_root).resolve()
        self.mode = str(mode or "auto")
        self.skills = self._discover()

    def _discover(self):
        if self.mode == "off":
            return ()
        if self.mode != "auto":
            candidate = self._resolve_configured_path(self.mode)
            skill_path = candidate if candidate.name == "SKILL.md" else candidate / "SKILL.md"
            return (self._parse_skill(skill_path),)

        skills_root = self.root / "skills"
        if not skills_root.is_dir():
            return ()
        paths = sorted(skills_root.glob("*/SKILL.md"), key=lambda item: item.as_posix().casefold())
        skills = tuple(self._parse_skill(path) for path in paths)
        normalized = [_normalize(skill.name) for skill in skills]
        if len(normalized) != len(set(normalized)):
            raise SkillConfigurationError("duplicate skill name")
        phrases = [
            _normalize(phrase)
            for skill in skills
            for phrase in skill.activation_phrases
        ]
        if len(phrases) != len(set(phrases)):
            raise SkillConfigurationError("activation phrases must be unique across skills")
        return skills

    def _resolve_configured_path(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.root / path).resolve()
        self._ensure_inside(resolved, raw_path)
        return resolved

    def _ensure_inside(self, resolved, display_path):
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SkillConfigurationError(f"skill path escapes workspace: {display_path}") from exc

    def _read(self, path, limit=MAX_READ_BYTES):
        resolved = Path(path).resolve()
        self._ensure_inside(resolved, path)
        if not resolved.is_file():
            raise SkillConfigurationError(f"skill file does not exist: {path}")
        data = resolved.read_bytes()
        if len(data) > limit:
            raise SkillConfigurationError(f"skill file is too large: {path}")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SkillConfigurationError(f"skill file is not UTF-8: {path}") from exc

    def _parse_skill(self, path):
        text = self._read(path, MAX_SKILL_BYTES)
        metadata, body = _parse_frontmatter(text)
        unknown = set(metadata) - FRONTMATTER_KEYS
        if unknown:
            raise SkillConfigurationError(f"unknown SKILL.md fields: {', '.join(sorted(unknown))}")
        name = str(metadata.get("name", "")).strip()
        description = str(metadata.get("description", "")).strip()
        if not name or not description or not body.strip():
            raise SkillConfigurationError(f"SKILL.md requires name, description, and body: {path}")
        if any(part in name for part in ("/", "\\", "..")):
            raise SkillConfigurationError(f"invalid skill name: {name}")
        activation_phrases, near_misses = _parse_description(description, path)

        always_block, routing_block = _parse_xml_sections(body)
        always_reads = _parse_reads(always_block)
        routes = _parse_routes(routing_block)
        if not routes or sum(route.id == "other" for route in routes) != 1:
            raise SkillConfigurationError(f"SKILL.md must define exactly one other route: {path}")
        route_ids = [_normalize(route.id) for route in routes]
        if len(route_ids) != len(set(route_ids)):
            raise SkillConfigurationError(f"duplicate task route in {path}")
        return SkillDocument(
            name=name,
            description=description,
            activation_phrases=activation_phrases,
            near_misses=near_misses,
            path=Path(path).resolve(),
            body=body.strip(),
            always_text=always_block.strip(),
            always_reads=always_reads,
            routes=routes,
        )

    def session_start(self):
        if not self.skills:
            return SkillContext()
        parts = []
        loaded = []
        for skill in self.skills:
            always_parts = []
            for relative in skill.always_reads:
                path = self._skill_file(skill, relative)
                always_parts.append(
                    f"### {xml_escape(relative)}\n{xml_escape(self._read(path))}"
                )
                loaded.append(Path(relative).name)
            route_catalog = "\n".join(
                f"- {xml_escape(route.id)}: {xml_escape(route.label)}"
                for route in skill.routes
            )
            parts.append(
                f"<skill name={xml_quoteattr(skill.name)}>\n"
                f"<description>{xml_escape(skill.description)}</description>\n"
                f"<activation-phrases>{xml_escape(' | '.join(skill.activation_phrases))}</activation-phrases>\n"
                f"<always-applicable>\n{xml_escape(skill.always_text)}\n\n"
                + "\n\n".join(always_parts)
                + "\n</always-applicable>\n\n"
                f"<task-routing>\n{route_catalog}\n</task-routing>\n</skill>"
            )
        text = "\n\n".join(parts)
        fingerprint_source = "\n\n".join(skill.body for skill in self.skills) + "\n" + text
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return SkillContext(
            text=text,
            fingerprint=fingerprint,
            skill_names=tuple(skill.name for skill in self.skills),
            loaded_paths=tuple(loaded),
        )

    def route(self, request, selector=None):
        if not self.skills:
            return RouteMatch()
        request = str(request or "")
        skill, skill_terms, selection = self._select_skill(request, selector=selector)
        candidates = tuple(self._candidate_evidence(request))
        if skill is None:
            return RouteMatch(
                selection_source=selection.source,
                selector_confidence=selection.confidence,
                selector_reason=selection.reason,
                candidates=candidates,
                route_context="<selected-skill>none</selected-skill>",
            )
        route, route_terms = self._select_route(skill, request)
        loaded = []
        sections = [
            f"<selected-skill name={xml_quoteattr(skill.name)}>{xml_escape(skill.description)}</selected-skill>",
            f"<task-route id={xml_quoteattr(route.id)} label={xml_quoteattr(route.label)}>",
        ]
        if route.workflow:
            path = self._skill_file(skill, route.workflow)
            sections.append(
                f"<workflow path={xml_quoteattr(route.workflow)}>{xml_escape(self._read(path))}</workflow>"
            )
            loaded.append(route.workflow)
        for relative in route.reads:
            path = self._skill_file(skill, relative)
            sections.append(
                f"<required-read path={xml_quoteattr(relative)}>{xml_escape(self._read(path))}</required-read>"
            )
            loaded.append(relative)
        sections.append("</task-route>")
        terms = tuple(_dedupe([*skill_terms, *route_terms]))
        if terms:
            sections.insert(1, "<matched-activation-terms>" + xml_escape(", ".join(terms)) + "</matched-activation-terms>")
        return RouteMatch(
            skill_name=skill.name,
            route_id=route.id,
            fallback=route.id == "other",
            matched_terms=terms,
            loaded_paths=tuple(loaded),
            route_context="\n\n".join(sections),
            selection_source=selection.source,
            selector_confidence=selection.confidence,
            selector_reason=selection.reason,
            candidates=tuple(
                SkillCandidate(
                    skill_name=item.skill_name,
                    activation_phrases=item.activation_phrases,
                    near_misses=item.near_misses,
                    matched_activation_phrases=item.matched_activation_phrases,
                    matched_near_misses=item.matched_near_misses,
                    name_matched=item.name_matched,
                    lexical_score=item.lexical_score,
                )
                for item in candidates
            ),
        )

    def status(self, match=None):
        match = match or RouteMatch()
        return {
            "mode": self.mode,
            "skills": [skill.name for skill in self.skills],
            **match.to_dict(),
        }

    def _select_skill(self, request, selector=None):
        explicit = re.findall(r"(?<!\S)/([\w.-]+)(?=\s|$)", request, flags=re.UNICODE)
        if len(explicit) > 1:
            raise SkillConfigurationError("multiple explicit skill invocations")
        if explicit:
            requested = _normalize(explicit[0])
            matches = [skill for skill in self.skills if _normalize(skill.name) == requested]
            if not matches:
                raise SkillConfigurationError(f"unknown skill: {explicit[0]}")
            return matches[0], (f"/{explicit[0]}",), SkillSelection(
                skill_name=matches[0].name,
                confidence=1.0,
                reason="explicit skill invocation",
                source="explicit",
            )

        candidates = self._candidate_evidence(request)
        positive = [
            candidate
            for candidate in candidates
            if candidate.lexical_score > 0 and not candidate.matched_near_misses
        ]
        positive.sort(key=lambda candidate: candidate.lexical_score, reverse=True)
        if len(positive) == 1:
            candidate = positive[0]
            skill = self._skill_by_name(candidate.skill_name)
            return skill, tuple(candidate.matched_activation_phrases), SkillSelection(
                skill_name=skill.name,
                confidence=1.0,
                reason="one skill matched an exclusive description phrase",
                source="description_phrase",
            )

        if selector is not None:
            try:
                selection = _coerce_selection(
                    selector(request, tuple(self.skills), tuple(candidates))
                )
            except Exception as exc:
                return None, (), SkillSelection(
                    confidence=0.0,
                    reason=f"Skill selector failed with {type(exc).__name__}",
                    source="description_model_error",
                )
            if selection.skill_name:
                skill = self._skill_by_name(selection.skill_name)
                if skill is None:
                    return None, (), SkillSelection(
                        confidence=selection.confidence,
                        reason="Skill selector returned an unknown Skill",
                        source="description_model_error",
                    )
                evidence = next(
                    candidate
                    for candidate in candidates
                    if candidate.skill_name == skill.name
                )
                if evidence.matched_near_misses:
                    return None, (), SkillSelection(
                        confidence=selection.confidence,
                        reason="selected Skill matched a Description near-miss exclusion",
                        source="description_model_rejected",
                    )
                if selection.confidence < MIN_SELECTOR_CONFIDENCE:
                    return None, (), SkillSelection(
                        confidence=selection.confidence,
                        reason="selected Skill confidence is below the routing threshold",
                        source="description_model_rejected",
                    )
                return skill, tuple(evidence.matched_activation_phrases), selection
            return None, (), selection

        if not positive:
            return None, (), SkillSelection(
                reason="no exclusive description phrase matched",
                source="no_description_match",
            )
        raise SkillConfigurationError("ambiguous description skill match")

    def _candidate_evidence(self, request):
        return tuple(self._candidate_for_skill(request, skill) for skill in self.skills)

    def _candidate_for_skill(self, request, skill):
        matched_activation = tuple(_matched_terms(request, skill.activation_phrases))
        matched_near_misses = tuple(_matched_terms(request, skill.near_misses))
        name_matched = bool(_matched_terms(request, [skill.name]))
        score = sum(len(_normalize(term)) for term in matched_activation)
        if matched_near_misses:
            score = 0
        return SkillCandidate(
            skill_name=skill.name,
            activation_phrases=skill.activation_phrases,
            near_misses=skill.near_misses,
            matched_activation_phrases=matched_activation,
            matched_near_misses=matched_near_misses,
            name_matched=name_matched,
            lexical_score=score,
        )

    def _skill_by_name(self, name):
        normalized = _normalize(name)
        return next((skill for skill in self.skills if _normalize(skill.name) == normalized), None)

    def _select_route(self, skill, request):
        fallback = next(route for route in skill.routes if route.id == "other")
        ranked = []
        for index, route in enumerate(skill.routes):
            if route.id == "other":
                continue
            terms = _matched_terms(request, [*route.triggers, route.label])
            ranked.append((max((len(_normalize(term)) for term in terms), default=0), -index, route, terms))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not ranked or ranked[0][0] == 0:
            return fallback, ()
        return ranked[0][2], tuple(ranked[0][3])

    def _skill_file(self, skill, relative):
        relative_path = Path(relative)
        if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
            raise SkillConfigurationError(f"invalid skill read path: {relative}")
        path = (skill.path.parent / relative_path).resolve()
        self._ensure_inside(path, relative)
        try:
            path.relative_to(skill.path.parent.resolve())
        except ValueError as exc:
            raise SkillConfigurationError(f"skill read escapes its directory: {relative}") from exc
        return path


def _parse_frontmatter(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillConfigurationError("SKILL.md must start with --- frontmatter")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise SkillConfigurationError("SKILL.md frontmatter is not closed") from exc
    metadata = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise SkillConfigurationError(f"invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in metadata:
            raise SkillConfigurationError(f"duplicate frontmatter field: {key}")
        value = value.strip()
        metadata[key] = value
    return metadata, "\n".join(lines[end + 1 :])


def _parse_description(description, path):
    if not description.startswith(DESCRIPTION_PREFIX):
        raise SkillConfigurationError(f"description must start with '{DESCRIPTION_PREFIX}': {path}")
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", description):
        raise SkillConfigurationError(f"description must use English only: {path}")
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", description)
    if len(words) < 20:
        raise SkillConfigurationError(f"description must contain at least 20 English words: {path}")
    phrases = tuple(phrase.strip() for phrase in QUOTED_TRIGGER_RE.findall(description))
    if len(phrases) != 2 or len({_normalize(phrase) for phrase in phrases}) != 2:
        raise SkillConfigurationError(f"description must contain exactly two unique quoted activation phrases: {path}")
    if any(len(re.findall(r"[A-Za-z]+", phrase)) < 3 for phrase in phrases):
        raise SkillConfigurationError(f"activation phrases must contain at least three words: {path}")
    near_miss = DESCRIPTION_NEAR_MISS_RE.search(description)
    if near_miss is None:
        raise SkillConfigurationError(f"description must end with three near-miss exclusions: {path}")
    near_misses = tuple(near_miss.group(name).strip() for name in ("first", "second", "third"))
    if any(not value for value in near_misses):
        raise SkillConfigurationError(f"description near-miss exclusions cannot be empty: {path}")
    normalized_near_misses = {_normalize(value) for value in near_misses}
    if len(normalized_near_misses) != 3:
        raise SkillConfigurationError(f"description must contain three unique near-miss exclusions: {path}")
    if normalized_near_misses & {_normalize(value) for value in phrases}:
        raise SkillConfigurationError(f"description near-misses cannot repeat activation phrases: {path}")
    if WORKFLOW_STEP_RE.search(description):
        raise SkillConfigurationError(f"description must not enumerate workflow steps: {path}")
    return phrases, near_misses


def _parse_xml_sections(body):
    try:
        root = ET.fromstring(f"<skill-root>{body}</skill-root>")
    except ET.ParseError as exc:
        raise SkillConfigurationError("SKILL.md body must be valid XML content") from exc
    children = list(root)
    tags = [child.tag for child in children]
    if tags.count("always-applicable") != 1 or tags.count("task-routing") != 1 or len(tags) != 2:
        raise SkillConfigurationError(
            "SKILL.md requires exactly one <always-applicable> and one <task-routing> block"
        )
    if tags != ["always-applicable", "task-routing"]:
        raise SkillConfigurationError("<always-applicable> must precede <task-routing>")
    if any(list(child) for child in children):
        raise SkillConfigurationError("SKILL.md XML boundary blocks cannot contain nested XML elements")
    return (children[0].text or "").strip(), (children[1].text or "").strip()


def _coerce_selection(value):
    if isinstance(value, SkillSelection):
        selection = value
    elif isinstance(value, dict) and set(value) == {"skill_name", "confidence", "reason"}:
        selection = SkillSelection(
            skill_name=str(value["skill_name"] or "").strip(),
            confidence=float(value["confidence"]),
            reason=str(value["reason"] or "").strip(),
        )
    else:
        raise SkillConfigurationError("skill selector must return SkillSelection")
    if not 0.0 <= selection.confidence <= 1.0:
        raise SkillConfigurationError("skill selector confidence must be between 0 and 1")
    if not selection.reason:
        raise SkillConfigurationError("skill selector reason is required")
    return selection


def _parse_reads(block):
    match = re.search(r"Always Read:\s*\n(?P<items>(?:\s*-\s+[^\n]+\n?)+)", block, flags=re.IGNORECASE)
    if not match:
        return ()
    return tuple(line.split("-", 1)[1].strip() for line in match.group("items").splitlines() if line.strip())


def _parse_routes(block):
    match = re.search(r"```json\s*(.*?)\s*```", block, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise SkillConfigurationError("<task-routing> must contain a JSON code block")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SkillConfigurationError("task routing JSON is invalid") from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise SkillConfigurationError("task routing JSON requires a tasks array")
    routes = []
    for item in tasks:
        if not isinstance(item, dict):
            raise SkillConfigurationError("task route must be an object")
        route_id = str(item.get("id", "")).strip()
        label = str(item.get("label", route_id)).strip()
        triggers = item.get("triggers", [])
        reads = item.get("reads", [])
        workflow = str(item.get("workflow", "")).strip()
        if not route_id or not label:
            raise SkillConfigurationError("task route requires id and label")
        if not isinstance(triggers, list) or any(not isinstance(value, str) for value in triggers):
            raise SkillConfigurationError(f"route {route_id} triggers must be strings")
        if not isinstance(reads, list) or any(not isinstance(value, str) for value in reads):
            raise SkillConfigurationError(f"route {route_id} reads must be strings")
        routes.append(TaskRoute(route_id, label, tuple(_dedupe(triggers)), tuple(reads), workflow))
    return tuple(routes)


def _normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _matched_terms(request, candidates):
    normalized_request = _normalize(request)
    matches = []
    for candidate in candidates:
        term = str(candidate).strip()
        normalized = _normalize(term)
        if not normalized:
            continue
        if _contains_term(normalized_request, normalized):
            matches.append(term)
    return _dedupe(matches)


def _contains_term(request, term):
    if any("一" <= char <= "鿿" for char in term):
        return term in request
    return re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", request, flags=re.UNICODE) is not None


def _dedupe(values):
    seen = set()
    result = []
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(str(value).strip())
    return result
