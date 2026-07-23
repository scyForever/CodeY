# Runtime architecture

`cli.py` builds `WorkspaceContext`, provider client, `SessionStore`, and `CodeYAgent`. `CodeYAgent` owns tools, memory, skills, hooks, stable prompt state, and the rule-supervised `CognitiveLoop`. `ContextManager` adds the selected route, memory, active or shadow patch guidance, transcript, and current request. `AgentLoop` calls the model, executes validated tools through `ToolExecutor`, then finalizes trace/outcome/root-cause/patch processing after each top-level task. Sessions, checkpoints, traces, reports, durable memory, and evolution patches live below `.codey/`.
