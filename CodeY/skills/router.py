"""Project-local skills with deterministic, progressive routing."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_SKILL_BYTES = 128 * 1024
MAX_READ_BYTES = 128 * 1024
FRONTMATTER_KEYS = {"name", "description", "triggers"}


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
    triggers: tuple[str, ...]
    path: Path
    body: str
    always_reads: tuple[str, ...]
    routes: tuple[TaskRoute, ...]


@dataclass(frozen=True)
class RouteMatch:
    skill_name: str = ""
    route_id: str = ""
    fallback: bool = False
    matched_terms: tuple[str, ...] = ()
    loaded_paths: tuple[str, ...] = ()
    route_context: str = ""

    def to_dict(self):
        return {
            "skill_name": self.skill_name,
            "route_id": self.route_id,
            "fallback": self.fallback,
            "matched_terms": list(self.matched_terms),
            "loaded_paths": list(self.loaded_paths),
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
        triggers = metadata.get("triggers", [])
        if not name or not description or not body.strip():
            raise SkillConfigurationError(f"SKILL.md requires name, description, and body: {path}")
        if not isinstance(triggers, list) or any(not isinstance(item, str) or not item.strip() for item in triggers):
            raise SkillConfigurationError(f"triggers must be a JSON string array: {path}")
        if any(part in name for part in ("/", "\\", "..")):
            raise SkillConfigurationError(f"invalid skill name: {name}")

        always_block = _xml_block(body, "always-applicable")
        routing_block = _xml_block(body, "task-routing")
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
            triggers=tuple(_dedupe(triggers)),
            path=Path(path).resolve(),
            body=body.strip(),
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
                always_parts.append(f"### {relative}\n{self._read(path)}")
                loaded.append(Path(relative).name)
            route_catalog = "\n".join(
                f"- {route.id}: {route.label}"
                for route in skill.routes
            )
            always_rules = _xml_block(skill.body, "always-applicable")
            parts.append(
                f"## Skill: {skill.name}\n{skill.description}\n\n"
                f"<always-applicable>\n{always_rules}\n\n" + "\n\n".join(always_parts) + "\n</always-applicable>\n\n"
                f"<task-routing>\n{route_catalog}\n</task-routing>"
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

    def route(self, request):
        if not self.skills:
            return RouteMatch()
        request = str(request or "")
        skill, skill_terms = self._select_skill(request)
        route, route_terms = self._select_route(skill, request)
        loaded = []
        sections = [f"Selected skill: {skill.name}", f"Selected task route: {route.id} ({route.label})"]
        if route.workflow:
            path = self._skill_file(skill, route.workflow)
            sections.append(f"## Workflow: {route.workflow}\n{self._read(path)}")
            loaded.append(route.workflow)
        for relative in route.reads:
            path = self._skill_file(skill, relative)
            sections.append(f"## Required read: {relative}\n{self._read(path)}")
            loaded.append(relative)
        terms = tuple(_dedupe([*skill_terms, *route_terms]))
        if terms:
            sections.insert(2, "Matched terms: " + ", ".join(terms))
        return RouteMatch(
            skill_name=skill.name,
            route_id=route.id,
            fallback=route.id == "other",
            matched_terms=terms,
            loaded_paths=tuple(loaded),
            route_context="\n\n".join(sections),
        )

    def status(self, match=None):
        match = match or RouteMatch()
        return {
            "mode": self.mode,
            "skills": [skill.name for skill in self.skills],
            **match.to_dict(),
        }

    def _select_skill(self, request):
        explicit = re.findall(r"(?<!\S)/([\w.-]+)(?=\s|$)", request, flags=re.UNICODE)
        if len(explicit) > 1:
            raise SkillConfigurationError("multiple explicit skill invocations")
        if explicit:
            requested = _normalize(explicit[0])
            matches = [skill for skill in self.skills if _normalize(skill.name) == requested]
            if not matches:
                raise SkillConfigurationError(f"unknown skill: {explicit[0]}")
            return matches[0], (f"/{explicit[0]}",)

        ranked = []
        for index, skill in enumerate(self.skills):
            terms = _matched_terms(request, [*skill.triggers, skill.name])
            ranked.append((max((len(_normalize(term)) for term in terms), default=0), -index, skill, terms))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if ranked[0][0] == 0 and len(self.skills) > 1:
            raise SkillConfigurationError("request does not select a skill")
        return ranked[0][2], tuple(ranked[0][3])

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
        if key == "triggers":
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise SkillConfigurationError("triggers must be JSON") from exc
        metadata[key] = value
    return metadata, "\n".join(lines[end + 1 :])


def _xml_block(body, tag):
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", body, flags=re.DOTALL)
    if len(matches) != 1:
        raise SkillConfigurationError(f"SKILL.md requires exactly one <{tag}> block")
    return matches[0]


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
