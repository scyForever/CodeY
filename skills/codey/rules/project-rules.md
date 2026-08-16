# Project rules

- `CodeY/` is the canonical Python package. Do not restore the deleted `code/` tree.
- Reject legacy `.codey/` session, task-state, memory, and evolution schemas instead of repairing them implicitly.
- Keep stable prompt material before per-task context and the current request last.
- Preserve workspace path confinement, approval checks, bounded joins, and secret redaction. Keep `fork_join` children read-only; keep `fork_merge` as a separate opt-in protocol with clean-base Git worktrees, exact disjoint path leases, no child shell/Git, validation gates bound to the integration commit, and a locally verified `ff-only` target update. Do not describe this as crash-atomic across the final Git update and state persistence.
- Route each new task independently; do not promote task-specific files into Always Read.
- Use deterministic fake clients and temporary workspaces for tests.
