"""Agent 运行时核心逻辑。

CodeYAgent 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import hashlib
import os
import re
import threading
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from ..storage import checkpoint as checkpointlib
from ..memory import store as memorylib
from ..providers.clients import normalize_model_completion
from ..tools import security as securitylib
from ..context.manager import ContextManager
from ..context.transcript import (
    AsyncConversationSummarizer,
    DEFAULT_RECENT_TURNS,
    append_transcript_entry,
    new_summary_state,
    new_transcript_state,
    render_transcript_entries,
    validate_summary_against_transcript,
    validate_summary_state,
    validate_transcript_state,
)
from ..evolution import CognitiveLoop
from ..storage.checkpoint import CHECKPOINT_NONE_STATUS
from ..context.prompt_prefix import build_prompt_prefix, tool_signature
from ..storage.run import RunStore
from ..tools.security import REDACTED_VALUE
from ..storage.session import SessionStore
from ..tools.context import ToolContext
from ..tools.executor import ToolExecutor
from ..tools import registry as toolkit
from ..context.workspace import IGNORED_PATH_NAMES, MAX_TRANSCRIPT_CHARS, WorkspaceContext, clip, now
from ..skills.hooks import HookManager
from ..skills.feedback import SkillFeedbackStore
from ..skills.router import RouteMatch, SkillRouter, SkillSelection
from ..skills.semantic import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_SEMANTIC_MIN_MARGIN,
    DEFAULT_SEMANTIC_MIN_SIMILARITY,
    SkillSemanticIndex,
)

DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "prompt_cache": True,
    "self_evolution": True,
}
SESSION_SCHEMA_VERSION = 2
FORK_SCHEMA_VERSION = "fork-v2"
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")

__all__ = ["CodeYAgent", "SessionStore"]


class CodeYAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
        allowed_tools=None,
        write_allowed_paths=None,
        skill_mode="auto",
        hook_callbacks=None,
        evolution_thresholds=None,
        evolution_llm_config=None,
        evolution_llm_client=None,
        skill_model_client=None,
        skill_selection_max_new_tokens=256,
        skill_embedding_client=None,
        skill_semantic_index=None,
        skill_semantic_min_similarity=DEFAULT_SEMANTIC_MIN_SIMILARITY,
        skill_semantic_min_margin=DEFAULT_SEMANTIC_MIN_MARGIN,
        skill_embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        summary_model_client=None,
        summary_recent_turns=DEFAULT_RECENT_TURNS,
        summary_max_new_tokens=512,
        summary_max_chars=4000,
        model_client_factory=None,
        model_client_lock=None,
        graph_checkpointer=None,
        max_fork_branches=4,
        max_parallel_branches=4,
        fork_merge_checks=None,
        fork_merge_check_timeout=120,
        allow_persistent_graph_checkpointer=False,
        parent_run_id="",
        fork_id="",
        branch_id="",
        graph_thread_id="",
    ):
        self.model_client = model_client
        self.skill_model_client = skill_model_client if skill_model_client is not None else model_client
        self.skill_selection_max_new_tokens = max(64, int(skill_selection_max_new_tokens))
        self.summary_model_client = summary_model_client if summary_model_client is not None else model_client
        self.summary_max_new_tokens = max(64, int(summary_max_new_tokens))
        self.model_client_factory = model_client_factory
        self._model_client_lock = model_client_lock or threading.RLock()
        self._branch_client_locks = {}
        self._branch_client_locks_guard = threading.Lock()
        self._skill_model_client_lock = (
            self._model_client_lock
            if self.skill_model_client is self.model_client
            else threading.RLock()
        )
        self._summary_model_client_lock = (
            self._model_client_lock
            if self.summary_model_client is self.model_client
            else threading.RLock()
        )
        self._session_lock = threading.RLock()
        self.graph_checkpointer = graph_checkpointer
        self.allow_persistent_graph_checkpointer = bool(allow_persistent_graph_checkpointer)
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.max_fork_branches = max(2, int(max_fork_branches))
        self.max_parallel_branches = max(1, min(int(max_parallel_branches), self.max_fork_branches))
        self.fork_merge_checks = self._normalize_fork_merge_checks(fork_merge_checks)
        self.fork_merge_check_timeout = max(1, min(int(fork_merge_check_timeout), 1800))
        self.fork_merge_enabled = bool(self.fork_merge_checks)
        self.parent_run_id = str(parent_run_id or "")
        self.fork_id = str(fork_id or "")
        self.branch_id = str(branch_id or "")
        self.graph_thread_id = str(graph_thread_id or "")
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.write_allowed_paths = self._normalize_write_allowed_paths(write_allowed_paths)
        self.skill_mode = str(skill_mode or "auto")
        self.skill_router = SkillRouter(self.root, mode=self.skill_mode)
        self.skill_embedding_client = skill_embedding_client
        if skill_semantic_index is not None:
            self.skill_semantic_index = skill_semantic_index
            if self.skill_embedding_client is None:
                self.skill_embedding_client = getattr(skill_semantic_index, "embedding_client", None)
        elif self.skill_embedding_client is not None:
            self.skill_semantic_index = SkillSemanticIndex(
                self.skill_embedding_client,
                min_similarity=skill_semantic_min_similarity,
                min_margin=skill_semantic_min_margin,
                batch_size=skill_embedding_batch_size,
            )
        else:
            self.skill_semantic_index = None
        self.last_semantic_skill_routing = {
            "status": "not_invoked",
            "enabled": self.skill_semantic_index is not None,
        }
        self.skill_feedback_store = SkillFeedbackStore(self.root / ".codey" / "skill-routing")
        self.hook_manager = HookManager(hook_callbacks)
        self.current_route = RouteMatch()
        self._compact_hook_run_id = ""
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".codey" / "runs")
        self.cognitive_loop = CognitiveLoop(
            self.root,
            thresholds=evolution_thresholds,
            llm_client=evolution_llm_client if evolution_llm_client is not None else model_client,
            llm_config=evolution_llm_config,
        )
        self.session = session if session is not None else self._new_session()
        self._validate_session_schema()
        start_reason = "resume" if session is not None else "startup"
        self.hook_manager.session_start(self, start_reason)
        self.memory = memorylib.LayeredMemory(
            self.session["memory"],
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self._apply_tool_allowlist(self.build_tools())
        self.tool_executor = ToolExecutor(self)
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.conversation_summarizer = AsyncConversationSummarizer(
            self,
            recent_turns=summary_recent_turns,
            max_chars=summary_max_chars,
        )
        resume_summary_interrupted = False
        if session is not None:
            resume_summary_interrupted = self.conversation_summarizer.recover_persisted_pending()
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.save_session()
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self.last_cognitive_loop = {}
        self._cognitive_guidance = ""
        self._stale_memory_paths = []
        self._last_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }
        self._ask_lock = threading.Lock()
        self._used_graph_thread_ids = set()
        self._active_turn_id = ""
        if resume_summary_interrupted:
            self.conversation_summarizer.schedule()

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _new_session(self):
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "revision": 0,
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": str(self.workspace.repo_root),
            "transcript": new_transcript_state(),
            "conversation_summary": new_summary_state(),
            "memory": memorylib.default_memory_state(),
            "checkpoints": {"items": {}, "current_id": ""},
            "runtime_identity": {},
            "resume_state": {},
            "session_context": {},
            "forks": {"schema_version": FORK_SCHEMA_VERSION, "items": {}},
        }

    def _validate_session_schema(self):
        required = {
            "schema_version",
            "revision",
            "id",
            "created_at",
            "workspace_root",
            "transcript",
            "conversation_summary",
            "memory",
            "checkpoints",
            "runtime_identity",
            "resume_state",
            "session_context",
            "forks",
        }
        if not isinstance(self.session, dict) or self.session.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"session schema_version must be {SESSION_SCHEMA_VERSION}; legacy sessions are not supported"
            )
        if set(self.session) != required:
            raise ValueError("session has unknown or missing fields")
        if not isinstance(self.session["revision"], int) or self.session["revision"] < 0:
            raise ValueError("session revision is invalid")
        for key in ("id", "created_at", "workspace_root"):
            if not isinstance(self.session[key], str) or not self.session[key]:
                raise ValueError(f"session {key} is invalid")
        validate_transcript_state(self.session["transcript"])
        validate_summary_state(self.session["conversation_summary"])
        validate_summary_against_transcript(
            self.session["conversation_summary"],
            self.session["transcript"]["entries"],
        )
        memorylib.normalize_memory_state(self.session["memory"], self.root)
        checkpoints = self.session["checkpoints"]
        if not isinstance(checkpoints, dict) or set(checkpoints) != {"items", "current_id"}:
            raise ValueError("session checkpoints state is invalid")
        if not isinstance(checkpoints["items"], dict) or any(
            not isinstance(value, dict) for value in checkpoints["items"].values()
        ):
            raise ValueError("session checkpoint items are invalid")
        current_id = checkpoints["current_id"]
        if not isinstance(current_id, str) or (current_id and current_id not in checkpoints["items"]):
            raise ValueError("session current checkpoint is invalid")
        for key in ("runtime_identity", "resume_state", "session_context"):
            if not isinstance(self.session[key], dict):
                raise ValueError(f"session {key} must be an object")
        forks = self.session["forks"]
        if (
            not isinstance(forks, dict)
            or set(forks) != {"schema_version", "items"}
            or forks["schema_version"] != FORK_SCHEMA_VERSION
            or not isinstance(forks["items"], dict)
        ):
            raise ValueError("session forks state is invalid")
        return self.session

    def save_session(self):
        self._validate_session_schema()
        self.session_path = self.session_store.save(self.session)
        return self.session_path

    def current_runtime_identity(self):
        return checkpointlib.current_runtime_identity(self)

    def checkpoint_state(self):
        return checkpointlib.checkpoint_state(self)

    def current_checkpoint(self):
        return checkpointlib.current_checkpoint(self)

    def invalidate_stale_memory(self):
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self):
        return checkpointlib.evaluate_resume_state(self)

    def render_checkpoint_text(self):
        return checkpointlib.render_checkpoint_text(self)

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self):
        return toolkit.build_tool_registry(self.tool_context())

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    @staticmethod
    def _normalize_write_allowed_paths(write_allowed_paths):
        if write_allowed_paths is None:
            return None
        normalized = tuple(
            toolkit.normalize_scoped_path(path)
            for path in write_allowed_paths
        )
        if not normalized:
            raise ValueError("write_allowed_paths must be a non-empty sequence")
        if len({path.casefold() for path in normalized}) != len(normalized):
            raise ValueError("write_allowed_paths contains duplicate paths")
        return normalized

    @staticmethod
    def _normalize_fork_merge_checks(commands):
        if commands is None:
            return ()
        normalized = []
        for index, command in enumerate(commands):
            if isinstance(command, (str, bytes)) or not isinstance(command, (list, tuple)):
                raise ValueError(f"fork_merge_checks[{index}] must be an argv sequence")
            argv = tuple(str(item).strip() for item in command)
            if not argv or any(not item for item in argv):
                raise ValueError(f"fork_merge_checks[{index}] must not be empty")
            if any(len(item) > 1000 for item in argv):
                raise ValueError(f"fork_merge_checks[{index}] contains an oversized argument")
            normalized.append(argv)
        if len(normalized) > 8:
            raise ValueError("fork_merge_checks accepts at most 8 commands")
        return tuple(normalized)

    def _apply_tool_allowlist(self, tools):
        if self.allowed_tools is None:
            return tools
        legal_names = toolkit.legal_tool_names()
        unknown = [name for name in self.allowed_tools if name not in legal_names]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {
            name: tool
            for name, tool in tools.items()
            if name in allowed
        }

    def tool_signature(self):
        return tool_signature(self.tools)

    def build_prefix(self):
        context = self.session.get("session_context", {})
        return build_prompt_prefix(
            workspace=self.workspace,
            tools=self.tools,
            skill_context=str(context.get("text", "")),
            skill_fingerprint=str(context.get("skill_fingerprint", "")),
        )

    def route_task(self, user_message, *, run_id, task_id):
        self.last_semantic_skill_routing = {
            "status": "not_invoked",
            "enabled": self.skill_semantic_index is not None,
        }
        route = self.skill_router.route(
            user_message,
            selector=self.select_skill,
        )
        event_id = self.skill_feedback_store.start(
            session_id=self.session["id"],
            run_id=run_id,
            task_id=task_id,
            request=self.redact_text(user_message),
            route=route.to_dict(),
        )
        self.current_route = replace(route, routing_event_id=event_id)
        return self.current_route

    def select_skill(self, request, skills, candidates):
        if self.skill_semantic_index is None:
            self.last_semantic_skill_routing = {
                "status": "disabled",
                "enabled": False,
            }
            return self.select_skill_from_descriptions(request, skills, candidates)

        excluded = tuple(
            candidate.skill_name
            for candidate in candidates
            if candidate.matched_near_misses
        )
        try:
            match = self.skill_semantic_index.select(
                self.redact_text(request),
                skills,
                excluded_skill_names=excluded,
            )
        except Exception as exc:
            self.last_semantic_skill_routing = {
                "status": "error",
                "enabled": True,
                "accepted": False,
                "error_type": type(exc).__name__,
            }
            fallback = self.select_skill_from_descriptions(request, skills, candidates)
            return replace(
                fallback,
                reason=(
                    f"{fallback.reason} Semantic vector routing failed with "
                    f"{type(exc).__name__}; used the Description-model fallback."
                ).strip(),
            )

        self.last_semantic_skill_routing = {
            "enabled": True,
            **match.to_dict(),
        }
        if match.accepted:
            return SkillSelection(
                skill_name=match.skill_name,
                confidence=max(0.0, min(1.0, match.score)),
                reason=match.reason,
                source="semantic_vector",
            )

        fallback = self.select_skill_from_descriptions(request, skills, candidates)
        return replace(
            fallback,
            reason=(
                f"{fallback.reason} Semantic vector route was not accepted "
                f"({match.status}); used the Description-model fallback."
            ).strip(),
        )

    def select_skill_from_descriptions(self, request, skills, candidates):
        catalog = [
            {
                "skill_name": skill.name,
                "description": skill.description,
                "activation_phrases": list(skill.activation_phrases),
                "near_misses": list(skill.near_misses),
                "lexical_evidence": next(
                    candidate.to_dict()
                    for candidate in candidates
                    if candidate.skill_name == skill.name
                ),
            }
            for skill in skills
        ]
        prompt = (
            "Select at most one project Skill for the user request. Use each Description as the "
            "activation boundary and respect its near-miss exclusions. Return exactly one JSON "
            "object with keys skill_name, confidence, and reason. Use an empty skill_name when no "
            "Skill applies.\n\n"
            f"Request:\n{request}\n\n"
            "Skill catalog:\n"
            + json.dumps(catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        with self._skill_model_client_lock:
            completion = normalize_model_completion(
                self.skill_model_client.complete(prompt, self.skill_selection_max_new_tokens)
            )
        try:
            payload = json.loads(completion.text.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("skill selector output must be strict JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"skill_name", "confidence", "reason"}:
            raise ValueError("skill selector output has invalid fields")
        return SkillSelection(
            skill_name=str(payload["skill_name"] or "").strip(),
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"] or "").strip(),
            source="description_model",
        )

    def route_context_text(self):
        return self.current_route.route_context

    def route_status(self):
        return {
            **self.skill_router.status(self.current_route),
            "semantic_route": dict(self.last_semantic_skill_routing),
        }

    def complete_skill_routing_feedback(self, task_state):
        event_id = str(getattr(task_state, "routing_event_id", "") or "")
        if not event_id:
            return None
        outcome = str(
            self.last_cognitive_loop.get("outcome", {}).get("label", "")
            or ("correct" if task_state.status == "completed" else "incorrect")
        )
        return self.skill_feedback_store.complete(
            event_id,
            status=task_state.status,
            stop_reason=task_state.stop_reason,
            outcome=outcome,
            tool_steps=task_state.tool_steps,
        )

    def submit_skill_feedback(self, correct, expected_skill_name="", note="", event_id=""):
        if not isinstance(correct, bool):
            raise TypeError("correct must be a boolean")
        expected_skill_name = str(expected_skill_name).strip()
        if expected_skill_name and expected_skill_name.casefold() not in {
            skill.name.casefold()
            for skill in self.skill_router.skills
        }:
            raise ValueError(f"unknown expected Skill: {expected_skill_name}")
        resolved = str(event_id or getattr(self.current_task_state, "routing_event_id", "") or "")
        if not resolved:
            raise ValueError("no Skill routing event is available")
        return self.skill_feedback_store.submit_user_feedback(
            resolved,
            correct=correct,
            expected_skill_name=expected_skill_name,
            note=note,
        )

    def propose_skill_description_patch(self, skill_name, min_samples=3):
        return self.skill_feedback_store.propose_description_patch(
            skill_name,
            min_samples=min_samples,
        )

    def restore_session_context(self, reason):
        payload = self.hook_manager.session_start(self, reason)
        if hasattr(self, "tools"):
            self._apply_prefix_state(self.build_prefix())
        self.save_session()
        return payload

    def compact_session_context(self, run_id):
        run_id = str(run_id or "")
        if run_id and self._compact_hook_run_id == run_id:
            return False
        self.restore_session_context("compact")
        self._compact_hook_run_id = run_id
        return True

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self):
        text = self.memory.render_memory_text()
        if self.feature_enabled("self_evolution") and self._cognitive_guidance:
            text += "\n\n" + self._cognitive_guidance
        return text

    def prepare_cognitive_context(self, task_state):
        self._stale_memory_paths = []
        self.last_cognitive_loop = {}
        if not self.feature_enabled("self_evolution"):
            task_state.evolution_context = {"enabled": False}
            self._cognitive_guidance = ""
            return dict(task_state.evolution_context)
        try:
            context, guidance = self.cognitive_loop.prepare_run(task_state)
        except Exception as exc:
            context = {
                "enabled": False,
                "error_type": type(exc).__name__,
            }
            guidance = ""
        task_state.evolution_context = context
        self._cognitive_guidance = guidance
        return dict(context)

    def process_cognitive_loop(self, task_state, tool_events):
        if not self.feature_enabled("self_evolution"):
            self.last_cognitive_loop = {"status": "disabled"}
            return dict(self.last_cognitive_loop)
        try:
            result = self.cognitive_loop.complete_run(
                task_state,
                tool_events,
                stale_paths=self._stale_memory_paths,
                redactor=self.redact_artifact,
            )
            self.last_cognitive_loop = self.redact_artifact(result)
            self.emit_trace(
                task_state,
                "cognitive_loop_completed",
                {
                    "status": self.last_cognitive_loop.get("status", ""),
                    "outcome": self.last_cognitive_loop.get("outcome", {}),
                    "root_cause": self.last_cognitive_loop.get("root_cause"),
                    "generated_patches": self.last_cognitive_loop.get("generated_patches", []),
                    "patch_transitions": self.last_cognitive_loop.get("patch_transitions", []),
                    "decision_audit": self.last_cognitive_loop.get("decision_audit", {}),
                },
            )
        except Exception as exc:
            self.last_cognitive_loop = {
                "status": "error",
                "error_type": type(exc).__name__,
            }
            try:
                self.emit_trace(task_state, "cognitive_loop_failed", dict(self.last_cognitive_loop))
            except Exception:
                pass
        return dict(self.last_cognitive_loop)

    def approve_cognitive_patch(self, patch_id):
        """Activate one human-reviewed patch and refresh its knowledge view."""
        return self.cognitive_loop.approve_patch(patch_id)

    def transcript_entries(self):
        return list(self.session["transcript"]["entries"])

    def transcript_text(self):
        entries = self.transcript_entries()
        if not entries:
            return "- empty"
        return clip(render_transcript_entries(entries), MAX_TRANSCRIPT_CHARS)

    def conversation_summary_state(self):
        return self.session["conversation_summary"]

    def conversation_summary_text(self):
        return str(self.conversation_summary_state()["committed"]["text"])

    def refresh_conversation_summary_async(self):
        return self.conversation_summarizer.schedule()

    def wait_for_conversation_summary(self, timeout=None):
        return self.conversation_summarizer.wait(timeout)

    def close(self, timeout=30.0):
        """Wait for asynchronous summary work, cancelling pending state on timeout."""
        completed = self.wait_for_conversation_summary(timeout)
        if not completed:
            self.conversation_summarizer.cancel_pending("SummaryFlushTimeout")
        return completed

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record_transcript(self, item, turn_id=None):
        with self._session_lock:
            resolved_turn_id = str(turn_id or self._active_turn_id or "").strip()
            if not resolved_turn_id:
                raise ValueError("manual transcript entries require an explicit turn_id")
            entry = append_transcript_entry(self.session["transcript"], item, resolved_turn_id)
            self.save_session()
            return dict(entry)

    @staticmethod
    def looks_sensitive_env_name(name):
        return securitylib.looks_sensitive_env_name(name)

    def is_secret_env_name(self, name):
        return securitylib.is_secret_env_name(name, secret_env_names=self.secret_env_names)

    def configured_secret_env_items(self):
        return securitylib.configured_secret_env_items(secret_env_names=self.secret_env_names)

    def detected_secret_env_items(self):
        return securitylib.detected_secret_env_items(secret_env_names=self.secret_env_names)

    def secret_env_summary(self):
        return securitylib.secret_env_summary(secret_env_names=self.secret_env_names)

    def detected_secret_env_summary(self):
        return securitylib.detected_secret_env_summary(secret_env_names=self.secret_env_names)

    def redact_text(self, text):
        return securitylib.redact_text(text, secret_env_names=self.secret_env_names)

    def redact_artifact(self, value, key=None):
        return securitylib.redact_artifact(value, key=key, secret_env_names=self.secret_env_names)

    def shell_env(self):
        return securitylib.shell_env(allowlist=self.shell_env_allowlist, root=self.root)

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def _build_prompt_and_metadata(self, user_message):
        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_manager.build(user_message)
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "transcript_chars": len(self.transcript_text()),
                "conversation_summary": {
                    "committed_generation": self.conversation_summary_state()["committed"]["generation"],
                    "covered_through_sequence": self.conversation_summary_state()["committed"]["covered_through_sequence"],
                    "pending_generation": int(
                        (self.conversation_summary_state().get("pending") or {}).get("generation", 0)
                    ),
                    "last_error": dict(self.conversation_summary_state().get("last_error") or {}),
                },
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "skill_fingerprint": self.prefix_state.skill_fingerprint,
                "skill_route": self.current_route.to_dict(),
                "skill_semantic_route": dict(self.last_semantic_skill_routing),
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        return checkpointlib.create_checkpoint(self, task_state, user_message, trigger)

    def infer_next_step(self, task_state):
        return checkpointlib.infer_next_step(task_state)

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `transcript`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整 transcript，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            summaries = dict(self.memory.to_dict().get("file_summaries", {}))
            if canonical_path in summaries and canonical_path not in self._stale_memory_paths:
                self._stale_memory_paths.append(canonical_path)
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def reject_durable_reason(self, note_text):
        text = str(note_text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
            return "secret_shaped"
        checkpoint_like_prefixes = (
            "current goal",
            "current blocker",
            "next step",
            "current phase",
            "key files",
            "freshness",
            "当前目标",
            "当前卡点",
            "下一步",
            "当前阶段",
            "关键文件",
            "已完成",
            "已排除",
        )
        if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
            return "transient_task_state"
        if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
            return "noisy_output"
        return ""

    def extract_durable_promotions(self, user_message, final_answer):
        user_text = str(user_message or "")
        if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text) or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
            return [], []
        promotions = []
        rejections = []
        for line in str(final_answer or "").splitlines():
            text = line.strip()
            if not text or REDACTED_VALUE in text:
                continue
            for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
                match = pattern.match(text)
                if not match:
                    continue
                note_text = match.group(1).strip()
                if note_text:
                    reason = self.reject_durable_reason(note_text)
                    if reason:
                        rejections.append(f"{topic}:{reason}")
                        break
                    promotions.append((topic, note_text))
                break
        return promotions, rejections

    def promote_durable_memory(self, user_message, final_answer):
        promotions, rejections = self.extract_durable_promotions(user_message, final_answer)
        promoted, superseded = self.memory.promote_durable(promotions)
        self.session["memory"] = self.memory.to_dict()
        self.last_durable_promotions = promoted
        self.last_durable_rejections = rejections
        self.last_durable_superseded = superseded
        return promoted, rejections, superseded

    def ask(self, user_message, *, thread_id=None):
        from .agent_loop import AgentLoop

        requested_thread_id = str(thread_id or "")
        with self._ask_lock:
            if requested_thread_id and requested_thread_id in self._used_graph_thread_ids:
                raise ValueError("graph thread_id must be unique per CodeYAgent run")
            if requested_thread_id:
                self._used_graph_thread_ids.add(requested_thread_id)
            final = AgentLoop(self).run(user_message, thread_id=requested_thread_id or None)
            if self.current_task_state is not None:
                self._used_graph_thread_ids.add(self.current_task_state.graph_thread_id)
            return final

    def complete_model(self, prompt, **kwargs):
        """Bind provider text and metadata inside one concurrency boundary."""

        with self._model_client_lock:
            return normalize_model_completion(
                self.model_client.complete(prompt, self.max_new_tokens, **kwargs)
            )

    def complete_summary_model(self, prompt):
        with self._summary_model_client_lock:
            return normalize_model_completion(
                self.summary_model_client.complete(prompt, self.summary_max_new_tokens)
            )

    def summary_model_name(self):
        return str(
            getattr(self.summary_model_client, "model", "")
            or self.summary_model_client.__class__.__name__
        )

    def execute_tool(self, name, args):
        result = self.tool_executor.execute(name, args)
        self._last_tool_result_metadata = dict(result.metadata)
        return result

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        return self.execute_tool(name, args).content

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.transcript_entries() if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "durable_promotions": list(self.last_durable_promotions),
            "durable_rejections": list(self.last_durable_rejections),
            "durable_superseded": list(self.last_durable_superseded),
            "cognitive_loop": dict(self.last_cognitive_loop),
            "redacted_env": self.detected_secret_env_summary(),
            "skill_route": self.current_route.to_dict(),
            "skill_semantic_route": dict(self.last_semantic_skill_routing),
            "session_context": dict(self.session.get("session_context", {})),
            "conversation_summary": dict(self.conversation_summary_state()),
            "fork_summary": dict(task_state.fork_summary),
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self.tool_context(), name, args)
        if name in {"write_file", "patch_file"}:
            self.authorize_write_path(args["path"])

    def tool_context(self):
        return ToolContext(
            root=self.root,
            path_resolver=self.path,
            shell_env_provider=self.shell_env,
            depth=self.depth,
            max_depth=self.max_depth,
            spawn_fork=self.spawn_fork,
            spawn_fork_merge=self.spawn_fork_merge,
            max_fork_branches=self.max_fork_branches,
            fork_merge_enabled=self.fork_merge_enabled,
        )

    def save_fork_state(self, fork_state):
        with self._session_lock:
            forks = self.session["forks"]
            forks["items"][fork_state["fork_id"]] = dict(fork_state)
            self.save_session()

    def _branch_model_client(self, spec):
        if self.model_client_factory is None:
            return self.model_client, self._model_client_lock
        client = self.model_client_factory(spec)
        if client is self.model_client:
            return client, self._model_client_lock
        # A Python API factory can accidentally return the same mutable client
        # for several branches.  Keying locks by object identity preserves real
        # concurrency for distinct clients while serializing unsafe reuse.
        with self._branch_client_locks_guard:
            lock = self._branch_client_locks.setdefault(id(client), threading.RLock())
        return client, lock

    def _create_child(
        self,
        task,
        max_steps,
        *,
        spec=None,
        workspace=None,
        approval_policy="never",
        read_only=True,
        allowed_tools=None,
        write_allowed_paths=None,
        parent_run_id="",
        fork_id="",
        branch_id="",
        graph_thread_id="",
    ):
        model_client, model_client_lock = self._branch_model_client(spec)
        child_workspace = workspace or self.workspace
        child = CodeYAgent(
            model_client=model_client,
            model_client_factory=self.model_client_factory,
            model_client_lock=model_client_lock,
            graph_checkpointer=self.graph_checkpointer,
            workspace=child_workspace,
            session_store=self.session_store,
            run_store=self.run_store,
            approval_policy=approval_policy,
            max_steps=int(max_steps),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=read_only,
            secret_env_names=self.secret_env_names,
            shell_env_allowlist=self.shell_env_allowlist,
            feature_flags={**self.feature_flags, "self_evolution": False},
            allowed_tools=allowed_tools,
            write_allowed_paths=write_allowed_paths,
            skill_mode=self.skill_mode,
            skill_embedding_client=self.skill_embedding_client,
            skill_semantic_index=self.skill_semantic_index,
            max_fork_branches=self.max_fork_branches,
            max_parallel_branches=self.max_parallel_branches,
            fork_merge_checks=(),
            fork_merge_check_timeout=self.fork_merge_check_timeout,
            allow_persistent_graph_checkpointer=self.allow_persistent_graph_checkpointer,
            parent_run_id=parent_run_id,
            fork_id=fork_id,
            branch_id=branch_id,
            graph_thread_id=graph_thread_id,
        )
        child.memory.set_task_summary(task)
        child.memory.append_note(self.redact_artifact(clip(self.transcript_text(), 300)), source="fork-parent")
        child.session["memory"] = child.memory.to_dict()
        child.save_session()
        return child

    def _create_read_only_child(
        self,
        task,
        max_steps,
        *,
        spec=None,
        parent_run_id="",
        fork_id="",
        branch_id="",
        graph_thread_id="",
    ):
        return self._create_child(
            task,
            max_steps,
            spec=spec,
            approval_policy="never",
            read_only=True,
            allowed_tools=self.allowed_tools,
            parent_run_id=parent_run_id,
            fork_id=fork_id,
            branch_id=branch_id,
            graph_thread_id=graph_thread_id,
        )

    def create_fork_child(self, spec, fork_id, thread_id, max_steps):
        parent_run_id = self.current_task_state.run_id if self.current_task_state else ""
        return self._create_read_only_child(
            spec.objective,
            max_steps,
            spec=spec,
            parent_run_id=parent_run_id,
            fork_id=fork_id,
            branch_id=spec.branch_id,
            graph_thread_id=thread_id,
        )

    def create_fork_merge_child(self, spec, fork_id, thread_id, max_steps, workspace):
        from .worktree_fork import SCOPED_WRITE_TOOLS

        parent_run_id = self.current_task_state.run_id if self.current_task_state else ""
        return self._create_child(
            spec.objective,
            max_steps,
            spec=spec,
            workspace=workspace,
            approval_policy="auto",
            read_only=False,
            allowed_tools=SCOPED_WRITE_TOOLS,
            write_allowed_paths=spec.allowed_paths,
            parent_run_id=parent_run_id,
            fork_id=fork_id,
            branch_id=spec.branch_id,
            graph_thread_id=thread_id,
        )

    def spawn_fork(self, args):
        from .fork import ForkCoordinator

        summary = ForkCoordinator(self).run(args)
        return json.dumps(summary, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def spawn_fork_merge(self, args):
        from .worktree_fork import WorktreeForkCoordinator

        summary = WorktreeForkCoordinator(self).run(args)
        return json.dumps(summary, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self.tool_context(), args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self.tool_context(), args)

    def tool_search(self, args):
        return toolkit.tool_search(self.tool_context(), args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self.tool_context(), args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self.tool_context(), args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self.tool_context(), args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = CodeYAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", CodeYAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", CodeYAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", CodeYAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", CodeYAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = CodeYAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", CodeYAgent.retry_notice()
        if "<final>" in raw:
            final = CodeYAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", CodeYAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", CodeYAgent.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = CodeYAgent.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = CodeYAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.conversation_summarizer.invalidate()
        with self._session_lock:
            self.session["transcript"] = new_transcript_state()
            self.session["conversation_summary"] = new_summary_state()
            self.session["memory"] = memorylib.default_memory_state()
            self._active_turn_id = ""
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.current_route = RouteMatch()
        self.restore_session_context("reset")

    def authorize_write_path(self, raw_path):
        if self.write_allowed_paths is None:
            return
        resolved = self.path(raw_path)
        relative = resolved.relative_to(self.root).as_posix()
        normalized = toolkit.normalize_scoped_path(relative)
        # Writable Fork leases are exact repo-relative paths.  The tool-schema
        # validator remains conservative and rejects case-fold collisions, but
        # authorization must never widen ``foo`` into ``Foo`` on a
        # case-sensitive filesystem.
        allowed = set(self.write_allowed_paths)
        if normalized not in allowed:
            raise ValueError(
                f"write path is not allowed by branch scope: {normalized}"
            )

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        try:
            inside_root = os.path.commonpath([str(self.root), str(resolved)]) == str(self.root)
        except ValueError:
            inside_root = False
        if not inside_root:
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved
