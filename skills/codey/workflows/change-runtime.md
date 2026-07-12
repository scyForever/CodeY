# Change the agent runtime

1. Trace CLI assembly through `CodeYAgent`, `ContextManager`, and `AgentLoop`.
2. Separate stable context from per-task context before editing.
3. Preserve deterministic prefix hashing and current-request placement.
4. Exercise the change with a fake model client.
5. Inspect prompt metadata, trace events, and the final report.
