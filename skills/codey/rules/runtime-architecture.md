# Runtime architecture

`cli.py` builds `WorkspaceContext`, provider client, `SessionStore`, and `CodeYAgent`. `CodeYAgent` owns tools, memory, skills, hooks, and stable prompt state. `ContextManager` adds the selected route, memory, transcript, and current request. `AgentLoop` calls the model and executes validated tools through `ToolExecutor`. Sessions, checkpoints, traces, reports, and durable memory live below `.codey/`.
