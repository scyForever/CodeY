---
name: codey
description: This skill should be used when the user's primary objective is to maintain or extend the CodeY local coding-agent runtime, and the request is best characterized as "CodeY agent runtime engineering" or "local coding assistant harness". It should not activate for generic source-code questions, unrelated agent products, or ordinary application feature work.
---

<always-applicable>
Always Read:
- rules/project-rules.md
- rules/runtime-architecture.md

Keep changes small, preserve workspace confinement, and verify observable behavior.
</always-applicable>

<task-routing>
Match every new task independently. Read only the selected route's workflow and reads.

```json
{
  "tasks": [
    {
      "id": "prompt-context",
      "label": "Prompt and context runtime / 提示词与上下文",
      "triggers": ["prompt", "context", "上下文", "提示词", "cache key"],
      "workflow": "workflows/change-runtime.md",
      "reads": ["rules/prompt-context.md"]
    },
    {
      "id": "tools-security",
      "label": "Tools and security / 工具与安全",
      "triggers": ["tool", "security", "approval", "工具", "安全", "审批"],
      "workflow": "workflows/change-tools.md",
      "reads": ["rules/tools-security.md"]
    },
    {
      "id": "providers-cli",
      "label": "Providers, CLI, and packaging / 模型后端与命令行",
      "triggers": ["provider", "CLI", "package", "模型后端", "命令行", "打包"],
      "reads": ["rules/providers-cli.md"]
    },
    {
      "id": "sessions-evaluation",
      "label": "Sessions, recovery, and evaluation / 会话恢复与评测",
      "triggers": ["session", "checkpoint", "memory", "evaluation", "会话", "检查点", "记忆", "评测"],
      "reads": ["rules/sessions-evaluation.md"]
    },
    {
      "id": "other",
      "label": "Other CodeY work / 其他任务",
      "triggers": [],
      "reads": []
    }
  ]
}
```
</task-routing>
