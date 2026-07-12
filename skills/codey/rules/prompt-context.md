# Prompt and context constraints

- Keep tool schemas and universal operating rules in the stable prefix.
- Keep selected route files in the per-task route section.
- A route change must not invalidate the stable prefix; a Skill core change must.
- Reduce older memory and history before the selected route. Never truncate the current user request.
- Record section sizes, reductions, Skill fingerprint, route, and loaded paths in metadata.
