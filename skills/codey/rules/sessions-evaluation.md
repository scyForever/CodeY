# Sessions and evaluation constraints

- Old session and task-state JSON must remain loadable when new fields are added.
- Resume reuses conversation state but routes the next user task afresh.
- Checkpoints describe recovery state; traces describe events; reports summarize one run.
- Evaluation defaults to deterministic fake clients. Live provider experiments must be optional and explicitly configured.
