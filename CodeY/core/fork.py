"""Bounded homogeneous fork/join orchestration."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import time
import uuid

from ..context.workspace import clip, now


_BRANCH_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
MAX_JOIN_CONTENT = 3600


def _safe_branch_id(value, index):
    normalized = _BRANCH_ID_PATTERN.sub("-", str(value or "").strip()).strip("-.")
    return normalized[:48] or f"branch-{index + 1}"


def _shorten(value, limit):
    text = str(value or "")
    return text if len(text) <= limit else text[:limit]


def compact_join_summary(summary, limit=MAX_JOIN_CONTENT):
    """Keep a fork result valid JSON after the generic tool-output limit."""

    payload = {key: value for key, value in summary.items() if key != "branches"}
    payload["branches"] = []
    for branch in summary.get("branches", []):
        item = dict(branch)
        original_answer = str(item.get("final_answer", ""))
        item["objective"] = _shorten(item.get("objective", ""), 160)
        item["final_answer"] = _shorten(original_answer, 600)
        if len(item["final_answer"]) < len(original_answer):
            item["answer_truncated"] = True
            item["answer_chars"] = len(original_answer)
        payload["branches"].append(item)

    def encoded_length():
        return len(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))

    while encoded_length() > limit:
        candidates = [
            branch
            for branch in payload["branches"]
            if len(str(branch.get("final_answer", ""))) > 80
        ]
        if not candidates:
            break
        branch = max(candidates, key=lambda item: len(str(item.get("final_answer", ""))))
        answer = str(branch.get("final_answer", ""))
        branch["final_answer"] = answer[: max(80, len(answer) // 2)]
        branch["answer_truncated"] = True

    if encoded_length() > limit:
        payload["branches"] = [
            {
                "branch_id": branch.get("branch_id", ""),
                "status": branch.get("status", ""),
                "stop_reason": branch.get("stop_reason", ""),
                "error_type": branch.get("error_type", ""),
                "final_answer": _shorten(branch.get("final_answer", ""), 120),
                "answer_truncated": True,
                "result_path": branch.get("result_path", ""),
            }
            for branch in payload["branches"]
        ]
    return payload


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    objective: str
    index: int

    @classmethod
    def from_payload(cls, item, index):
        return cls(
            branch_id=_safe_branch_id(item.get("id"), index),
            objective=str(item.get("objective", "")).strip(),
            index=index,
        )


@dataclass
class BranchResult:
    branch_id: str
    objective: str
    index: int
    status: str
    final_answer: str = ""
    run_id: str = ""
    thread_id: str = ""
    stop_reason: str = ""
    error_type: str = ""
    duration_ms: int = 0
    result_path: str = ""

    def to_dict(self):
        return {
            "branch_id": self.branch_id,
            "objective": self.objective,
            "index": self.index,
            "status": self.status,
            "final_answer": self.final_answer,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "stop_reason": self.stop_reason,
            "error_type": self.error_type,
            "duration_ms": self.duration_ms,
            "result_path": self.result_path,
        }


class ForkCoordinator:
    """Run sibling objectives with isolated child agents and join their results."""

    def __init__(self, parent):
        self.parent = parent

    def run(self, args):
        parent = self.parent
        parent_state = parent.current_task_state
        if parent_state is None:
            raise RuntimeError("fork_join requires an active parent run")

        fork_id = "fork_" + uuid.uuid4().hex[:10]
        specs = [BranchSpec.from_payload(item, index) for index, item in enumerate(args["tasks"])]
        if len({spec.branch_id for spec in specs}) != len(specs):
            raise ValueError("branch ids collide after normalization")
        max_steps = int(args.get("max_steps", 3))
        join_policy = str(args.get("join_policy", "all_settled"))
        started_at = time.monotonic()
        fork_state = {
            "fork_id": fork_id,
            "source_run_id": parent_state.run_id,
            "source_task_id": parent_state.task_id,
            "status": "running",
            "join_policy": join_policy,
            "branch_ids": [spec.branch_id for spec in specs],
            "created_at": now(),
            "updated_at": now(),
        }
        parent.save_fork_state(fork_state)
        parent.emit_trace(
            parent_state,
            "fork_started",
            {
                "fork_id": fork_id,
                "branch_ids": list(fork_state["branch_ids"]),
                "join_policy": join_policy,
                "max_steps": max_steps,
            },
        )
        try:
            return self._execute(
                specs=specs,
                fork_id=fork_id,
                fork_state=fork_state,
                max_steps=max_steps,
                join_policy=join_policy,
                started_at=started_at,
            )
        except Exception as exc:
            fork_state.update(
                {
                    "status": "failed",
                    "updated_at": now(),
                    "error_type": type(exc).__name__,
                }
            )
            try:
                parent.save_fork_state(fork_state)
                parent.emit_trace(
                    parent_state,
                    "join_failed",
                    {"fork_id": fork_id, "error_type": type(exc).__name__},
                )
            except Exception:
                pass
            raise

    def _execute(self, specs, fork_id, fork_state, max_steps, join_policy, started_at):
        parent = self.parent
        parent_state = parent.current_task_state
        prepared = []
        for spec in specs:
            thread_id = f"{parent_state.graph_thread_id}/{fork_id}/{spec.branch_id}"
            child = parent.create_fork_child(
                spec=spec,
                fork_id=fork_id,
                thread_id=thread_id,
                max_steps=max_steps,
            )
            prepared.append((spec, thread_id, child))
            parent.emit_trace(
                parent_state,
                "branch_started",
                {"fork_id": fork_id, "branch_id": spec.branch_id, "thread_id": thread_id},
            )

        worker_count = min(len(prepared), parent.max_parallel_branches)
        results = {}
        safe_results = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="codey-fork") as executor:
            futures = {
                executor.submit(self._run_branch, spec, thread_id, child): (spec, thread_id)
                for spec, thread_id, child in prepared
            }
            for future in as_completed(futures):
                spec, thread_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = BranchResult(
                        branch_id=spec.branch_id,
                        objective=spec.objective,
                        index=spec.index,
                        status="failed",
                        thread_id=thread_id,
                        error_type=type(exc).__name__,
                    )
                self._persist_result(parent_state, fork_id, result, results, safe_results)

        ordered = [results[spec.branch_id] for spec in specs]
        ordered_safe = [safe_results[spec.branch_id] for spec in specs]
        succeeded = sum(result.status == "completed" for result in ordered)
        failed = len(ordered) - succeeded
        status = "completed" if failed == 0 else "partial"
        summary = compact_join_summary(
            {
                "fork_id": fork_id,
                "status": status,
                "join_policy": join_policy,
                "branch_count": len(ordered),
                "succeeded": succeeded,
                "failed": failed,
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "branches": ordered_safe,
            }
        )
        fork_state.update(
            {
                "status": status,
                "updated_at": now(),
                "succeeded": succeeded,
                "failed": failed,
                "result_paths": [result.result_path for result in ordered],
            }
        )
        parent.save_fork_state(fork_state)
        parent_state.fork_count += 1
        parent_state.fork_summary = {
            key: value
            for key, value in summary.items()
            if key != "branches"
        }
        parent.run_store.write_task_state(parent_state)
        parent.emit_trace(parent_state, "join_completed", dict(parent_state.fork_summary))
        return summary

    def _persist_result(self, parent_state, fork_id, result, results, safe_results):
        parent = self.parent
        result_path = parent.run_store.branch_result_path(
            parent_state.run_id,
            fork_id,
            result.branch_id,
        )
        result.result_path = result_path.relative_to(parent.current_run_dir).as_posix()
        safe_result = parent.redact_artifact(result.to_dict())
        parent.run_store.write_branch_result(
            parent_state.run_id,
            fork_id,
            result.branch_id,
            safe_result,
        )
        results[result.branch_id] = result
        safe_results[result.branch_id] = safe_result
        event = "branch_finished" if result.status == "completed" else "branch_failed"
        parent.emit_trace(
            parent_state,
            event,
            {
                "fork_id": fork_id,
                "branch_id": result.branch_id,
                "child_run_id": result.run_id,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "error_type": result.error_type,
                "result_path": result.result_path,
                "duration_ms": result.duration_ms,
            },
        )

    @staticmethod
    def _run_branch(spec, thread_id, child):
        started_at = time.monotonic()
        final = child.ask(spec.objective, thread_id=thread_id)
        task_state = child.current_task_state
        return BranchResult(
            branch_id=spec.branch_id,
            objective=spec.objective,
            index=spec.index,
            status=task_state.status,
            final_answer=clip(final, 1200),
            run_id=task_state.run_id,
            thread_id=thread_id,
            stop_reason=task_state.stop_reason,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
