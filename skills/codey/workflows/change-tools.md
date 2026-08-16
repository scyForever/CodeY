# Change tools safely

1. Read the registry, validation, execution, and security boundary.
2. Update schema and behavior together.
3. Test valid, invalid, denied, and path-escape inputs.
4. Verify structured metadata and secret redaction.
5. Keep `fork_join` children read-only and bounded. For `fork_merge`, verify disabled-by-default exposure, parent approval, exact path scopes, worktree isolation, no child shell/Git, pre-stage secret rejection, Git-status failure closure, validation `HEAD` identity, cleanup-pending semantics, target drift, and `ff-only` behavior.
