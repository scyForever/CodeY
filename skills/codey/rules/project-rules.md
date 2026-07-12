# Project rules

- `CodeY/` is the canonical Python package. Do not restore the deleted `code/` tree.
- Keep `.codey/` session and run artifacts compatible unless a migration is explicitly requested.
- Keep stable prompt material before per-task context and the current request last.
- Preserve workspace path confinement, approval checks, read-only delegation, and secret redaction.
- Route each new task independently; do not promote task-specific files into Always Read.
- Use deterministic fake clients and temporary workspaces for tests.
