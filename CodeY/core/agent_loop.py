"""LangGraph-backed homogeneous agent control loop."""

import time
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..context.workspace import clip, now
from ..storage.checkpoint import (
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_PARTIAL_STALE_STATUS,
    CHECKPOINT_WORKSPACE_MISMATCH_STATUS,
)
from .task_state import STATUS_RUNNING, TaskState


class AgentGraphState(TypedDict, total=False):
    user_message: str
    run_id: str
    task_id: str
    graph_thread_id: str
    run_started_at: float
    max_attempts: int
    task_state: dict
    tool_events: list[dict]
    raw: str
    payload: object
    completion_metadata: dict
    model_duration_ms: int
    decision: str
    final: str
    finalized: bool


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent
        if agent.graph_checkpointer is None:
            agent.graph_checkpointer = InMemorySaver()
        elif (
            not isinstance(agent.graph_checkpointer, InMemorySaver)
            and not agent.allow_persistent_graph_checkpointer
        ):
            raise ValueError(
                "persistent graph checkpointers require explicit opt-in because graph state may contain task content"
            )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)
        builder.add_node("initialize", self._initialize)
        builder.add_node("think", self._think)
        builder.add_node("reflect", self._reflect)
        builder.add_node("act", self._act)
        builder.add_node("fork", self._fork)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "think")
        builder.add_edge("think", "reflect")
        builder.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "think": "think",
                "act": "act",
                "fork": "fork",
                "finalize": "finalize",
            },
        )
        builder.add_edge("act", "think")
        builder.add_edge("fork", "think")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.agent.graph_checkpointer)

    def run(self, user_message, thread_id=None):
        agent = self.agent
        run_id = agent.new_run_id()
        task_id = agent.new_task_id()
        graph_thread_id = str(
            thread_id
            or agent.graph_thread_id
            or f"{agent.session['id']}/{run_id}"
        )
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)
        result = self.graph.invoke(
            {
                "user_message": str(user_message),
                "run_id": run_id,
                "task_id": task_id,
                "graph_thread_id": graph_thread_id,
                "run_started_at": time.monotonic(),
                "max_attempts": max_attempts,
                "tool_events": [],
                "decision": "think",
                "final": "",
                "finalized": False,
            },
            config={
                "configurable": {"thread_id": graph_thread_id},
                "recursion_limit": max(50, max_attempts * 4 + 20),
            },
        )
        return str(result.get("final", ""))

    def _initialize(self, state):
        agent = self.agent
        user_message = state["user_message"]
        agent.memory.set_task_summary(user_message)
        route = agent.route_task(
            user_message,
            run_id=state["run_id"],
            task_id=state["task_id"],
        )

        task_state = TaskState.create(
            run_id=state["run_id"],
            task_id=state["task_id"],
            user_request=user_message,
        )
        agent._active_turn_id = task_state.task_id
        agent.record_transcript({"role": "user", "content": user_message, "created_at": now()})
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        task_state.skill_name = route.skill_name
        task_state.route_id = route.route_id
        task_state.route_fallback = route.fallback
        task_state.route_match_terms = route.matched_terms
        task_state.routing_event_id = route.routing_event_id
        task_state.graph_thread_id = state["graph_thread_id"]
        task_state.phase = "think"
        task_state.parent_run_id = agent.parent_run_id
        task_state.fork_id = agent.fork_id
        task_state.branch_id = agent.branch_id
        agent.current_task_state = task_state
        agent.prepare_cognitive_context(task_state)
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
                "skill_route": route.to_dict(),
                "evolution_context": dict(task_state.evolution_context),
                "graph_thread_id": task_state.graph_thread_id,
                "parent_run_id": task_state.parent_run_id,
                "fork_id": task_state.fork_id,
                "branch_id": task_state.branch_id,
            },
        )
        return {"task_state": task_state.to_dict()}

    def _think(self, state):
        agent = self.agent
        task_state = self._task_state(state)
        task_state.phase = "think"
        if task_state.tool_steps >= agent.max_steps:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
            agent.record_transcript({"role": "assistant", "content": final, "created_at": now()})
            agent.run_store.write_task_state(task_state)
            return {
                "task_state": task_state.to_dict(),
                "decision": "finalize",
                "final": final,
            }
        if task_state.attempts >= state["max_attempts"]:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
            agent.record_transcript({"role": "assistant", "content": final, "created_at": now()})
            agent.run_store.write_task_state(task_state)
            return {
                "task_state": task_state.to_dict(),
                "decision": "finalize",
                "final": final,
            }

        task_state.record_attempt()
        agent.run_store.write_task_state(task_state)
        prompt_started_at = time.monotonic()
        prompt, prompt_metadata = agent._build_prompt_and_metadata(state["user_message"])
        agent.emit_trace(
            task_state,
            "prompt_built",
            {
                "prompt_metadata": prompt_metadata,
                "phase": "think",
                "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
            },
        )
        if self._checkpoint_for_prompt_state(task_state, state["user_message"], prompt_metadata):
            prompt, prompt_metadata = agent._build_prompt_and_metadata(state["user_message"])
            agent.emit_trace(
                task_state,
                "prompt_rebuilt_after_compact",
                {
                    "prompt_metadata": prompt_metadata,
                    "phase": "think",
                },
            )
        agent.emit_trace(
            task_state,
            "model_requested",
            {
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
                "phase": "think",
                "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
            },
        )
        prompt_cache_key = None
        prompt_cache_retention = None
        if getattr(agent.model_client, "supports_prompt_cache", False):
            prompt_cache_key = prompt_metadata.get("prompt_cache_key")
            prompt_cache_retention = "in_memory"
        agent.last_prompt_metadata = prompt_metadata
        model_started_at = time.monotonic()
        try:
            completion = agent.complete_model(
                prompt,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
        except Exception as exc:
            final = f"Task failed because the model provider raised {type(exc).__name__}."
            task_state.stop_model_error(final)
            agent.record_transcript({"role": "assistant", "content": final, "created_at": now()})
            agent.emit_trace(
                task_state,
                "model_failed",
                {"error_type": type(exc).__name__, "phase": "think"},
            )
            return {
                "task_state": task_state.to_dict(),
                "decision": "finalize",
                "final": final,
            }

        completion_metadata = dict(completion.metadata)
        if completion_metadata:
            prompt_metadata.update(completion_metadata)
        agent.last_prompt_metadata = prompt_metadata
        return {
            "task_state": task_state.to_dict(),
            "raw": completion.text,
            "completion_metadata": completion_metadata,
            "decision": "reflect",
            "model_duration_ms": int((time.monotonic() - model_started_at) * 1000),
        }

    def _reflect(self, state):
        agent = self.agent
        task_state = self._task_state(state)
        task_state.phase = "reflect"
        if state.get("decision") == "finalize":
            agent.run_store.write_task_state(task_state)
            return {"task_state": task_state.to_dict(), "decision": "finalize"}

        kind, payload = agent.parse(state.get("raw", ""))
        agent.emit_trace(
            task_state,
            "model_parsed",
            {
                "kind": kind,
                "phase": "reflect",
                "completion_metadata": dict(state.get("completion_metadata", {})),
                "duration_ms": int(state.get("model_duration_ms", 0)),
            },
        )
        if kind == "tool":
            name = str(payload.get("name", ""))
            decision = "fork" if name in {"fork_join", "fork_merge"} else "act"
        elif kind == "retry":
            agent.record_transcript({"role": "assistant", "content": payload, "created_at": now()})
            agent.run_store.write_task_state(task_state)
            decision = "think"
        else:
            decision = "finalize"

        agent.emit_trace(
            task_state,
            "graph_routed",
            {
                "phase": "reflect",
                "next_node": decision,
            },
        )
        update = {
            "task_state": task_state.to_dict(),
            "payload": payload,
            "decision": decision,
        }
        if kind == "final":
            update["final"] = str(payload or state.get("raw", "")).strip()
        return update

    @staticmethod
    def _route_after_reflect(state):
        return state.get("decision", "finalize")

    def _act(self, state):
        return self._execute_action(state, "act")

    def _fork(self, state):
        return self._execute_action(state, "fork")

    def _execute_action(self, state, phase):
        agent = self.agent
        task_state = self._task_state(state)
        task_state.phase = phase
        payload = dict(state.get("payload", {}) or {})
        name = str(payload.get("name", ""))
        args = dict(payload.get("args", {}) or {})
        task_state.record_tool(name)
        tool_started_at = time.monotonic()
        tool_result = agent.execute_tool(name, args)
        result = tool_result.content
        tool_events = list(state.get("tool_events", []))
        tool_events.append(
            {
                "name": name,
                "args": dict(args),
                "content": result,
                "metadata": dict(tool_result.metadata or {}),
            }
        )
        agent.record_transcript(
            {
                "role": "tool",
                "name": name,
                "args": args,
                "content": result,
                "created_at": now(),
            }
        )
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "tool_executed",
            {
                "name": name,
                "args": args,
                "result": clip(result, 500),
                "phase": phase,
                "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                **dict(tool_result.metadata or {}),
            },
        )
        checkpoint = agent.create_checkpoint(task_state, state["user_message"], trigger="tool_executed")
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": "tool_executed",
            },
        )
        return {
            "task_state": task_state.to_dict(),
            "tool_events": tool_events,
            "decision": "think",
        }

    def _finalize(self, state):
        agent = self.agent
        task_state = self._task_state(state)
        final = str(state.get("final", ""))
        if task_state.status == STATUS_RUNNING:
            agent.record_transcript({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
        task_state.phase = "finalize"
        agent.current_task_state = task_state
        final = self._finish_run(
            task_state,
            state["user_message"],
            final,
            state["run_started_at"],
            list(state.get("tool_events", [])),
        )
        return {
            "task_state": task_state.to_dict(),
            "final": final,
            "finalized": True,
        }

    def _checkpoint_for_prompt_state(self, task_state, user_message, prompt_metadata):
        agent = self.agent
        rebuild_prompt = False
        if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
            trigger = "freshness_mismatch"
        elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
            agent.emit_trace(
                task_state,
                "runtime_identity_mismatch",
                {"fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", []))},
            )
            trigger = "workspace_mismatch"
        elif prompt_metadata.get("budget_reductions"):
            rebuild_prompt = agent.compact_session_context(task_state.run_id)
            trigger = "context_budget"
        else:
            return False
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=trigger)
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": trigger},
        )
        return rebuild_prompt

    def _task_state(self, state):
        task_state = TaskState.from_dict(state["task_state"])
        self.agent.current_task_state = task_state
        return task_state

    def _finish_run(self, task_state, user_message, final, run_started_at, tool_events):
        agent = self.agent
        agent.promote_durable_memory(user_message, final)
        checkpoint_trigger = "run_finished" if task_state.status == "completed" else task_state.stop_reason or "run_stopped"
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=checkpoint_trigger)
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": checkpoint_trigger,
            },
        )
        agent.process_cognitive_loop(task_state, tool_events)
        agent.complete_skill_routing_feedback(task_state)
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "graph_thread_id": task_state.graph_thread_id,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.refresh_conversation_summary_async()
        return final
