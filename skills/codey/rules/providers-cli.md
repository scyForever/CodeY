# Provider and CLI constraints

- Keep provider protocol differences behind `providers/clients.py`.
- Treat OpenAI-compatible and Anthropic-compatible clients as compatibility adapters, not full official-SDK feature surfaces.
- Configuration precedence is explicit CLI value, project/shell environment, then code default.
- Keep `codey` as the canonical command and `codey` as a compatibility alias while `.codey/` remains the persistence root.
