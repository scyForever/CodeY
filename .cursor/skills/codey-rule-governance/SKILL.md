---
name: codey-rule-governance
description: >-
  Entry point for CodeY repository-rule governance. Use when scanning distributed
  AGENTS.md, CLAUDE.md, Cursor, or Codex rules; creating a reviewed rule patch;
  running an isolated canary; or delegating a bounded edit to local Codex or Claude.
primary: true
---

# CodeY rule governance - Cursor entry

This is a thin Cursor registration entry. The canonical maintenance skill is
[skills/codey/SKILL.md](../../../skills/codey/SKILL.md).

1. Read `../../../skills/codey/SKILL.md`.
2. Select the `rule-governance` route for scan, plan, trial, apply, rollback, or external-agent delegation work.
3. Read only the selected route's required files and follow its workflow.
4. Keep repository rules read-only until an explicit reviewed Patch action.

Do not duplicate the implementation or a generated rule bundle here. Keep this
entry thin so Cursor loads the canonical source.
