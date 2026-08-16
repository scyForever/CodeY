# Tool and security constraints

- Update tool registration, schema, validation, and implementation together.
- Resolve every model-supplied path under the workspace after symlink resolution.
- Keep risky writes and shell execution behind the existing approval policy.
- `fork_join` child agents stay read-only, bounded, and unable to approve risky operations.
- `fork_merge` stays disabled without coordinator-configured validation commands. When enabled, require parent approval, a readable clean Git status, independent worktrees, exact disjoint file leases, scoped file tools only, pre-stage environment-secret checks, candidate-diff verification, validation bound to an unchanged integration commit, target-drift checks, and `ff-only`; never expose shell/Git to writable children or partially apply successful branches after another branch fails.
- Redact secret-shaped values from traces, reports, and tool output metadata.
