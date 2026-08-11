# Sessions and evaluation constraints

- Session and task-state schemas are strict contracts. Reject legacy versions and unknown or missing fields; perform an explicit offline migration when old artifacts must be retained.
- Resume reuses conversation state but routes the next user task afresh.
- Checkpoints describe recovery state; traces describe events; reports summarize one run.
- Evaluation defaults to deterministic fake clients. Live provider experiments must be optional and explicitly configured.
