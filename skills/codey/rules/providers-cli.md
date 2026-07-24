# Provider and CLI constraints

- Keep provider protocol differences behind `providers/clients.py`.
- Treat OpenAI-compatible and Anthropic-compatible clients as compatibility adapters, not full official-SDK feature surfaces.
- Configuration precedence is explicit CLI value, project/shell environment, then code default.
- Keep `codey` as the canonical command. Preserve legacy one-shot/REPL arguments while routing `codey rules ...` before model-provider construction; `.codey/` remains the local persistence root.
