"""一次 ask() 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"
TASK_STATE_SCHEMA_VERSION = 2


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""
    skill_name: str = ""
    route_id: str = ""
    route_fallback: bool = False
    route_match_terms: tuple[str, ...] = ()
    routing_event_id: str = ""
    evolution_context: dict = field(default_factory=dict)
    graph_thread_id: str = ""
    phase: str = "think"
    parent_run_id: str = ""
    fork_id: str = ""
    branch_id: str = ""
    fork_count: int = 0
    fork_summary: dict = field(default_factory=dict)

    @classmethod
    def create(cls, task_id, user_request, run_id=""):
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data):
        required = {
            "schema_version",
            "run_id",
            "task_id",
            "user_request",
            "status",
            "tool_steps",
            "attempts",
            "last_tool",
            "stop_reason",
            "final_answer",
            "checkpoint_id",
            "resume_status",
            "skill_name",
            "route_id",
            "route_fallback",
            "route_match_terms",
            "routing_event_id",
            "evolution_context",
            "graph_thread_id",
            "phase",
            "parent_run_id",
            "fork_id",
            "branch_id",
            "fork_count",
            "fork_summary",
        }
        if not isinstance(data, dict) or data.get("schema_version") != TASK_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"task state schema_version must be {TASK_STATE_SCHEMA_VERSION}; legacy task states are not supported"
            )
        if set(data) != required:
            raise ValueError("task state has unknown or missing fields")
        return cls(
            run_id=str(data["run_id"]),
            task_id=str(data["task_id"]),
            user_request=str(data["user_request"]),
            status=str(data["status"]),
            tool_steps=int(data["tool_steps"]),
            attempts=int(data["attempts"]),
            last_tool=str(data["last_tool"]),
            stop_reason=str(data["stop_reason"]),
            final_answer=str(data["final_answer"]),
            checkpoint_id=str(data["checkpoint_id"]),
            resume_status=str(data["resume_status"]),
            skill_name=str(data["skill_name"]),
            route_id=str(data["route_id"]),
            route_fallback=bool(data["route_fallback"]),
            route_match_terms=tuple(str(item) for item in data["route_match_terms"]),
            routing_event_id=str(data["routing_event_id"]),
            evolution_context=dict(data["evolution_context"]),
            graph_thread_id=str(data["graph_thread_id"]),
            phase=str(data["phase"]),
            parent_run_id=str(data["parent_run_id"]),
            fork_id=str(data["fork_id"]),
            branch_id=str(data["branch_id"]),
            fork_count=int(data["fork_count"]),
            fork_summary=dict(data["fork_summary"]),
        )

    def record_attempt(self):
        # attempt 统计的是“模型被调用了几轮”，不等于 tool_steps。
        self.attempts += 1
        return self

    def record_tool(self, name):
        # tool_steps 只统计真正进入执行阶段的工具调用次数。
        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def stop(self, stop_reason, status=STATUS_STOPPED, final_answer=""):
        # stop_reason 和 status 分开存，是为了区分“怎么停的”和“停下时是什么状态”。
        self.status = status
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer=""):
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer=""):
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_model_error(self, final_answer=""):
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def finish_success(self, final_answer):
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
        return self

    def to_dict(self):
        return {
            "schema_version": TASK_STATE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
            "skill_name": self.skill_name,
            "route_id": self.route_id,
            "route_fallback": self.route_fallback,
            "route_match_terms": list(self.route_match_terms),
            "routing_event_id": self.routing_event_id,
            "evolution_context": dict(self.evolution_context),
            "graph_thread_id": self.graph_thread_id,
            "phase": self.phase,
            "parent_run_id": self.parent_run_id,
            "fork_id": self.fork_id,
            "branch_id": self.branch_id,
            "fork_count": self.fork_count,
            "fork_summary": dict(self.fork_summary),
        }
