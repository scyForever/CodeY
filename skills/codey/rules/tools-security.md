# Tool and security constraints

- Update tool registration, schema, validation, and implementation together.
- Resolve every model-supplied path under the workspace after symlink resolution.
- Keep risky writes and shell execution behind the existing approval policy.
- Delegates stay read-only, bounded, and unable to approve risky operations.
- Redact secret-shaped values from traces, reports, and tool output metadata.
