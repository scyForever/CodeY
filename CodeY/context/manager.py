"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、checkpoint、memory、相关笔记、摘要、transcript
以及当前用户请求送进模型。
"""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import DEFAULT_RECENT_TURNS, render_transcript_entries, split_for_context


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3200,
    "checkpoint": 1200,
    "route_context": 1800,
    "memory": 1300,
    "relevant_memory": 900,
    "conversation_summary": 1800,
    "transcript": 3600,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "checkpoint": 500,
    "route_context": 500,
    "memory": 400,
    "relevant_memory": 300,
    "conversation_summary": 500,
    "transcript": 1500,
}
# 当前请求永不裁剪；稳定约束和最近对话晚于可重建的记忆与摘要被压缩。
DEFAULT_REDUCTION_ORDER = (
    "relevant_memory",
    "memory",
    "conversation_summary",
    "transcript",
    "checkpoint",
    "route_context",
    "prefix",
)
SECTION_ORDER = (
    "prefix",
    "checkpoint",
    "route_context",
    "memory",
    "relevant_memory",
    "conversation_summary",
    "transcript",
    "current_request",
)
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 3


def _tail_clip(text, limit):
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 工作记忆 +
        相关笔记 + 摘要 + transcript + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `CodeYAgent.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，`LayeredMemory`
        提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        memory_enabled = True
        relevant_memory_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "checkpoint": "",
            "route_context": str(getattr(getattr(self.agent, "current_route", None), "route_context", "")) or "Selected task route:\n- none",
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "conversation_summary": self._conversation_summary_raw(),
            "transcript": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["checkpoint"] = checkpoint_text
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 如果 prompt 超预算，就按固定顺序不断压缩。
        # 这里的顺序体现了平台偏好：先牺牲可重建记忆，再压缩摘要和 transcript。
        # 最新用户请求永远不裁剪，因为那是本轮最重要的输入。
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    def _compute_section_floors(self):
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "conversation_summary":
                raw = section_texts[section]
                rendered[section] = SectionRender(
                    raw=raw,
                    budget=int(budget or 0),
                    rendered=_tail_clip(raw, int(budget or 0)),
                    details=self._conversation_summary_details(),
                )
            elif section == "transcript":
                rendered[section] = self._render_transcript_section(int(budget or 0))
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        header = "Relevant memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )

        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉。
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _conversation_summary_raw(self):
        committed = self.agent.conversation_summary_state()["committed"]
        text = committed["text"] or "- none"
        return (
            "Conversation summary "
            f"(generation={committed['generation']}, covered_through={committed['covered_through_sequence']}):\n"
            f"{text}"
        )

    def _conversation_summary_details(self):
        state = self.agent.conversation_summary_state()
        committed = state["committed"]
        pending = state.get("pending") or {}
        return {
            "generation": committed["generation"],
            "covered_through_sequence": committed["covered_through_sequence"],
            "source_hash": committed["source_hash"],
            "pending_generation": int(pending.get("generation", 0)),
            "pending_covered_through_sequence": int(pending.get("covered_through_sequence", 0)),
            "last_error": dict(state.get("last_error") or {}),
        }

    def _render_transcript_section(self, budget, enforce_budget=True):
        entries = self.agent.transcript_entries()
        active_turn_id = str(getattr(self.agent, "_active_turn_id", ""))
        older, recent, current = split_for_context(
            entries,
            recent_turns=getattr(self.agent.conversation_summarizer, "recent_turns", DEFAULT_RECENT_TURNS)
            if hasattr(self.agent, "conversation_summarizer")
            else DEFAULT_RECENT_TURNS,
            current_turn_id=active_turn_id,
        )
        covered = self.agent.conversation_summary_state()["committed"]["covered_through_sequence"]
        pending_fallback = [entry for entry in older if entry["sequence"] > covered]
        current_visible = [entry for entry in current if entry["role"] != "user"]
        groups = self._transcript_groups(pending_fallback, recent, current_visible)
        raw = self._render_transcript_groups(groups)
        rendered_groups = list(groups)
        rendered = raw
        truncated = False
        if enforce_budget and budget > 0 and len(rendered) > budget:
            truncated = True
            selected = []
            for group in reversed(groups):
                candidate = [group, *selected]
                candidate_text = self._render_transcript_groups(candidate)
                if len(candidate_text) <= budget:
                    selected = candidate
                    continue
                if not selected:
                    available = max(20, budget - len("Transcript:\n"))
                    selected = [{**group, "text": _tail_clip(group["text"], available)}]
                break
            rendered_groups = selected
            rendered = self._render_transcript_groups(rendered_groups)
            if len(rendered) > budget:
                rendered = _tail_clip(rendered, budget)
        rendered_turn_ids = [group["turn_id"] for group in rendered_groups]
        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "recent_turn_limit": getattr(
                    getattr(self.agent, "conversation_summarizer", None),
                    "recent_turns",
                    DEFAULT_RECENT_TURNS,
                ),
                "pending_fallback_entries": len(pending_fallback),
                "recent_turn_count": len({entry["turn_id"] for entry in recent}),
                "current_turn_entries": len(current_visible),
                "rendered_turn_ids": rendered_turn_ids,
                "rendered_turn_count": len(rendered_turn_ids),
                "truncated": truncated,
            },
        )

    @staticmethod
    def _transcript_groups(pending_fallback, recent, current):
        groups = []
        for scope, entries in (
            ("summary-pending-fallback", pending_fallback),
            ("recent-original", recent),
            ("current-turn", current),
        ):
            by_turn = []
            current_id = None
            bucket = []
            for entry in entries:
                if entry["turn_id"] != current_id:
                    if bucket:
                        by_turn.append((current_id, bucket))
                    current_id = entry["turn_id"]
                    bucket = []
                bucket.append(entry)
            if bucket:
                by_turn.append((current_id, bucket))
            for turn_id, turn_entries in by_turn:
                groups.append(
                    {
                        "scope": scope,
                        "turn_id": turn_id,
                        "text": render_transcript_entries(turn_entries),
                    }
                )
        return groups

    @staticmethod
    def _render_transcript_groups(groups):
        if not groups:
            return "Transcript:\n- empty"
        parts = ["Transcript:"]
        for group in groups:
            parts.append(f"[turn:{group['turn_id']} scope:{group['scope']}]\n{group['text']}")
        return "\n".join(parts)

    def _assemble_prompt(self, rendered):
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            rendered[section].rendered
            for section in SECTION_ORDER
            if rendered[section].rendered
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts):
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["relevant_memory"].raw_chars,
                "rendered_chars": rendered["relevant_memory"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "route": {
                **(self.agent.current_route.to_dict() if getattr(self.agent, "current_route", None) else {}),
                "raw_chars": rendered["route_context"].raw_chars,
                "rendered_chars": rendered["route_context"].rendered_chars,
            },
            "conversation_summary": {
                "raw_chars": rendered["conversation_summary"].raw_chars,
                "rendered_chars": rendered["conversation_summary"].rendered_chars,
                **dict(rendered["conversation_summary"].details or {}),
            },
            "transcript": {
                "raw_chars": rendered["transcript"].raw_chars,
                "rendered_chars": rendered["transcript"].rendered_chars,
                **dict(rendered["transcript"].details or {}),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            },
        }
