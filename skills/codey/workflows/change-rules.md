# Change repository rule governance

1. Scan the relevant repository and inspect source scopes, config warnings, duplicates, secrets, and path-boundary errors.
2. Change discovery, Patch, adapter, runner, or CLI code at its owning layer; keep cognitive-evolution Patch behavior separate.
3. Add deterministic tests for source/target drift, explicit approval, rollback, canary assignment, isolation, and diff budgets.
4. Use fake/scripted runners in pytest. Live Codex/Claude trials are explicit smoke tests, never default test dependencies.
5. Validate `codey rules --help`, `scan`, `plan`, `diff`, `status`, and agent probing before broader regression tests.
