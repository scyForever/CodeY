# CodeY

CodeY 是一个面向本地代码仓库的小型 Coding Agent Runtime。它将模型调用、工具执行、上下文预算、会话恢复、结构化 Skill 路由和运行审计组合在一个可读、可测试的 Python 控制循环中。

> 项目仍处于开发阶段。当前 provider 层以 Ollama、OpenAI-compatible、Anthropic-compatible 和 DeepSeek-compatible HTTP 适配器为主；“兼容”不代表已经接入对应厂商 SDK 的全部原生能力。

## 核心能力

- **受控 Agent 循环**：模型每轮只能返回一个 `<tool>` 或 `<final>`，Runtime 负责解析、校验、执行和停止。
- **结构化 Skill 管理**：先进行领域 Skill 粗路由，再按 `SKILL.md` 的 Tasks 表选择任务路由。
- **渐进式按需加载**：SessionStart 只加载 Skill 导航和 Always-read 核心约束；只有命中的任务才读取 workflow 和 route-specific 文件。
- **XML 核心边界**：`<always-applicable>` 与 `<task-routing>` 将核心约束和路由协议分开，便于压缩后重新注入。
- **Python Runtime Hooks**：支持 `startup`、`resume`、`reset`、`compact` 四种 SessionStart 原因。
- **分段上下文预算**：稳定 prefix、route context、memory、相关记忆、历史和当前请求分别管理；当前请求不会被裁剪。
- **工具安全边界**：工作区路径约束、参数校验、重复调用防护、危险操作审批、只读子 Agent 和 secret redaction。
- **可恢复与可审计**：会话、检查点、工作记忆、trace、task state 和 report 持久化到 `.codey/`。
- **规则监督的认知闭环**：任务结束后基于结构化 trace 完成自省、结果归一化、根因归类、Patch 灰度和知识路由；可选的受限 LLM Advisor 只能消歧和精炼规则候选，不会从模型自由文本中猜测性学习。
- **评测工具**：包含固定任务评测、上下文/记忆/恢复/安全实验与 provider 实验脚本。

## 架构

```text
CLI
 ├─ WorkspaceContext
 ├─ Provider Client
 ├─ SkillRouter ── 双层路由 + 渐进加载
 └─ SessionStart HookManager
          │
          ▼
      CodeYAgent
       ├─ PromptPrefix（稳定规则、工具、Skill core）
       ├─ ContextManager（route、memory、history、request）
       ├─ AgentLoop
       │    ├─ ModelClient
       │    ├─ ToolExecutor
       │    └─ CognitiveLoop（trace、outcome、root cause、patch gate）
       └─ Session / Checkpoint / Run / Memory stores
```

主要目录：

```text
CodeY/
├─ core/          # Runtime、AgentLoop、TaskState
├─ context/       # Workspace、稳定 prefix、上下文预算
├─ skills/        # SkillRouter 与 Python SessionStart hooks
├─ tools/         # 工具注册、执行器、安全和上下文
├─ storage/       # Session、Checkpoint、Run artifacts
├─ memory/        # 工作记忆与 durable memory
├─ evolution/     # 规则监督的任务后认知闭环与 Patch 状态机
├─ providers/     # 模型后端适配器
└─ evaluation/    # 固定评测与指标实验
skills/codey/     # 当前仓库自身的示例/维护 Skill
tests/            # Runtime Skill、Hooks、Context 和包入口测试
scripts/          # 指标与 provider 实验命令
```

## 结构化 Skill

### 目录约定

CodeY 默认扫描目标工作区的 `skills/*/SKILL.md`：

```text
skills/my-skill/
├─ SKILL.md
├─ rules/
└─ workflows/
```

`SKILL.md` 使用一个很小的 frontmatter：

```yaml
---
name: my-skill
description: Maintain this project's API and data pipeline.
triggers: ["API", "data pipeline", "数据管道"]
---
```

正文必须恰好包含一组兄弟 XML 边界：

````md
<always-applicable>
Always Read:
- rules/project-rules.md
- rules/architecture.md
</always-applicable>

<task-routing>
```json
{
  "tasks": [
    {
      "id": "fix-api",
      "label": "Fix API / 修复接口",
      "triggers": ["API", "接口"],
      "workflow": "workflows/fix-api.md",
      "reads": ["rules/api.md"]
    },
    {
      "id": "other",
      "label": "Other",
      "triggers": [],
      "reads": []
    }
  ]
}
```
</task-routing>
````

### 双层路由

1. **Skill 路由**：根据每个 Skill 的 `name` 与 `triggers` 选出领域 Skill。`description` 用于导航与展示，当前不参与确定性匹配；显式 `/skill-name` 优先。
2. **Task 路由**：只在已选 Skill 内按任务 label/trigger 匹配；无匹配时使用唯一的 `other`。

一个 `ask()` 运行期间会固定路由，避免模型重试或工具循环中途切换工作流；同一 session 的下一条用户请求会重新路由。

### 渐进式加载

- SessionStart：读取 `SKILL.md`、Always-read 文件和紧凑任务索引。
- 当前任务：只读取命中 route 的 `workflow` 与 `reads`。
- 不会把所有 workflow/rules 一次性塞进稳定 prefix。
- Skill core 改变会更新 Skill fingerprint 和 prefix hash；单纯切换任务 route 不会。

### 路径与格式安全

Skill 文件必须是工作区内的 UTF-8 普通文件。Runtime 在解析符号链接后的真实路径上校验边界，并拒绝路径逃逸、重复 route、缺失 `other`、未知 frontmatter 字段和超限文件。

## Runtime SessionStart Hooks

这里的 Hook 是 **CodeY Python Runtime 内部生命周期接口**，不是 Claude Code 的 shell hook。

| 原因 | 触发时机 |
|---|---|
| `startup` | 创建新 CodeYAgent |
| `resume` | 从已有 session 恢复 |
| `reset` | REPL `/reset` 清空会话 |
| `compact` | ContextManager 发生预算压缩后恢复 Skill core |

内置 Hook 将结构化结果写入 session 的 `session_context`：

- 原因与 generation
- Skill fingerprint
- 已发现 Skill
- 已加载的 Always-read 路径
- 可重新构造稳定 prefix 的核心文本

`compact` 对同一 run 幂等，避免一次运行中的每轮 prompt 都重复触发。Python API 可以通过 `hook_callbacks` 注册额外 callback；Runtime 不会从项目 Skill 自动执行任意 shell 命令。

## 安装

要求 Python 3.10+：

```bash
python -m venv .venv
python -m pip install -e .
codey --help
```

`codey` 是正式命令。包目录为 `CodeY/`。

## 配置

复制示例配置：

```bash
copy .env.example .env
```

或在 PowerShell 中：

```powershell
Copy-Item .env.example .env
```

常用变量：

| Provider | 主要变量 |
|---|---|
| DeepSeek | `CODEY_DEEPSEEK_API_BASE`、`CODEY_DEEPSEEK_API_KEY`、`CODEY_DEEPSEEK_MODEL` |
| OpenAI-compatible | `CODEY_OPENAI_API_BASE`、`CODEY_OPENAI_API_KEY`、`CODEY_OPENAI_MODEL` |
| Anthropic-compatible | `CODEY_ANTHROPIC_API_BASE`、`CODEY_ANTHROPIC_API_KEY`、`CODEY_ANTHROPIC_MODEL` |
| Ollama | `--host`、`--model` |

选择优先级是：显式 CLI 参数 → 项目 `.env`/shell 环境 → 代码默认值。不要提交真实 `.env`。CodeY 会在 trace/report 中清理配置为 secret 的环境变量，但这不是凭据管理系统的替代品。

## 使用

### 单次任务

```bash
codey --cwd . --provider deepseek "解释这个仓库的入口"
```

### 交互模式

```bash
codey --cwd .
```

REPL 命令：

- `/help`：显示帮助
- `/memory`：显示提炼后的工作记忆
- `/route`：显示已发现 Skill 与最近一次路由
- `/session`：显示 session 文件路径
- `/reset`：清空历史与工作记忆，并触发 `SessionStart(reset)`
- `/exit`：退出

### Skill 控制

```bash
# 默认：自动扫描 skills/*/SKILL.md
codey --skill auto

# 禁用 Runtime Skill
codey --skill off

# 加载工作区内指定 Skill 文件或目录
codey --skill skills/codey/SKILL.md
```

### 恢复会话

```bash
codey --resume latest
codey --resume <session-id>
```

恢复只复用会话、记忆和检查点；新的用户任务仍会重新匹配 route。

### 审批策略

```bash
codey --approval ask   # 默认，危险操作询问
codey --approval auto  # 自动批准
codey --approval never # 拒绝危险操作
```

## 工具与安全模型

内置工具包括：

- `list_files`
- `read_file`
- `search`
- `run_shell`
- `write_file`
- `patch_file`
- `delegate`

关键边界：

- 所有模型提供的路径解析后必须仍位于 workspace root。
- `write_file`、`patch_file`、`run_shell` 等风险动作遵守审批策略。
- `patch_file` 要求旧文本唯一匹配。
- 连续重复且无进展的工具调用会被拒绝。
- delegate 是步数受限的只读子 Agent，不能批准风险动作。
- shell 只继承 allowlist 环境；trace/report 做 secret redaction。

模型输出仍属于不可信输入。若把 CodeY 用于不可信仓库或高风险执行环境，应额外使用容器、受限系统用户和网络隔离。

## Prompt 与上下文管理

最终 prompt 的顺序是：

1. 稳定 prefix：Runtime 规则、工具协议、Skill core、workspace 基线
2. route context：当前任务 workflow 与按需读取文件
3. working memory
4. relevant memory
5. 压缩后的 transcript
6. 当前用户请求

超预算时优先压缩相关记忆和旧历史，再压缩 working memory、route context 与 prefix。当前请求永不裁剪。每轮 metadata 记录 section 大小、压缩步骤、prefix/workspace/tool/skill fingerprint 和路由加载证据。

`prompt_cache_key` 是 CodeY 的稳定 prefix 身份。只有 provider adapter 明确声明支持时才会下发；当前 Anthropic-compatible 适配器并不等于官方 Anthropic SDK，也不会因此自动获得所有原生 prompt caching 能力。

## 会话、检查点和工件

默认目录：

```text
.codey/
├─ sessions/<session-id>.json
├─ runs/<run-id>/
│  ├─ task_state.json
│  ├─ trace.jsonl
│  └─ report.json
├─ memory/
   ├─ MEMORY.md
   └─ topics/*.md
└─ evolution/
   ├─ patches/patch_*.json
   ├─ behavior/policies.md
   ├─ decisions.md
   └─ knowledge/
      ├─ definition/*.md
      └─ experience/*.md
```

- **session**：可恢复的历史、memory、checkpoint 和 session context。
- **task_state**：一次 ask 的状态、route、工具步数和停止原因。
- **trace**：逐事件时间线。
- **report**：一次运行的结果与关键 metadata。
- **checkpoint**：用于新鲜度、工作区不匹配和压缩后的恢复。

## 规则监督的自进化认知闭环

每个顶层 `ask()` 终结后都会执行一条确定性链路：

```text
Trace Collector
  -> Outcome Evaluator (correct / incorrect / partial / harmful)
  -> Root Cause Analyzer (policy / strategy / chain / execution)
  -> Patch Generator
  -> Safety Gate
  -> patch store + knowledge views
```

自省结果固定回答三个问题，并写入 `report.json` 的 `cognitive_loop.reflection`：本次是否观察到新知识、是否出现错误路线、是否确认有过时知识。没有结构化证据时返回 `no` 或 `not_observed`，不会从最终回答的自由文本里提取猜测性结论。Collector 只保留工具名、状态、错误码、路径和 workspace-change 信号；命令、工具输出、代码 diff、临时 checkpoint 和模型推理都不进入认知 Patch。

### 可选的规则 + LLM 模式

默认 `rules` 模式完全使用确定性规则。显式启用 `hybrid` 后，CodeY 会为 Collector 产生的终止事件、工具事件和 stale-path 信号分配稳定 `evidence_id`，并只把这份裁剪后的 Evidence Packet 交给 LLM Advisor：

```bash
python -m CodeY --evolution-mode hybrid "检查并修改运行时"
```

混合模式仍由规则掌握最终权限：

- Outcome 的 `harmful`、明确失败和无结构化失败的成功是硬规则；LLM 只可把歧义的 `partial` 保持为 `partial` 或提升为 `incorrect`。
- Root Cause 先由规则给出候选集合；LLM 只能在候选层级中消歧，安全或权限证据产生的 `policy` 不能降级。
- Patch 类型、scope、correction kind 和生命周期状态由规则固定；LLM 只能精炼 `correction.action`，并把触发条件缩小到原规则条件的子集。
- 每个被接受的建议必须引用真实 `evidence_id` 并达到置信度阈值；非法 JSON、伪造证据、秘密形态内容、diff 片段、provider 异常或低置信度都会回退到原规则候选。
- Safety Gate 始终是纯规则组件，LLM 不能把 Patch 直接设为 `active`，也不能绕过 Policy/Definition 的人工审核。

诊断只在 Outcome 或 Root Cause 存在规则歧义时调用；Patch Advisor 只在已有规则候选时调用。审计结果写入 `report.json` 的 `cognitive_loop.decision_audit`，仅保存 prompt 版本/hash、结构化建议状态、证据引用和回退错误码，不保存原始 prompt 或模型响应。Python API 可以复用主模型，也可以注入独立 critic：

```python
agent = CodeYAgent(
    ...,
    evolution_llm_config={
        "mode": "hybrid",
        "min_confidence": 0.8,
        "max_new_tokens": 800,
    },
    evolution_llm_client=critic_client,  # 省略时复用 model_client
)
```

> **启用前必读**：现有护栏不能消除语义误判、shadow 因果归因、prompt injection、外部 provider 隐私、人工治理、并发存储和长尾延迟风险。完整清单、状态和修复优先级见 [规则 + LLM 自进化闭环风险登记](docs/evolution-hybrid-risk-register.md)。

Patch 是 JSON 对象，至少包含 `type`、`scope`、`correction`、`trigger_conditions`、`status`、`metrics` 和来源 run。状态转换为：

- `strategy`、`action_chain`、`knowledge_experience`：`draft -> shadow -> active -> expired`，达到灰度阈值后可自动激活。
- `policy`：`draft -> review_required -> active -> expired`，绝不自动升级；只能调用 `agent.approve_cognitive_patch(patch_id)` 完成人工批准。
- `knowledge_definition`：CodeY 采用更保守的仓库级取舍，也进入 `review_required`，避免仅凭文件名自动改写架构边界。

`shadow` 不是被动计数：同 scope 的候选按稳定哈希进入少量任务 prompt，并标记为 shadow guidance；只有本次 trace 实际触发 Patch 声明的目标工具或路径才计为 hit，无关任务的成功不会抬高成功率。默认参数为 20% 灰度流量、至少 3 次命中、命中率不低于 10%、成功率不低于 80% 才激活；累计 100 次命中后成功率低于 40% 会过期，任何 `harmful` 灰度结果会立即过期。激活后仍持续统计，后续退化也会触发过期；过期 Patch 保留 JSON 审计记录，但会从 active 知识视图移除。

激活 Patch 的知识视图按类型路由：Policy 写入 `behavior/policies.md`，Strategy/Action Chain 写入 `decisions.md`，定义与经验分别写入 `knowledge/definition/` 和 `knowledge/experience/`。这些路径都位于 workspace 的 `.codey/evolution/` 下，Patch JSON 是审计事实源。Python API 可通过 `evolution_thresholds={...}` 覆盖阈值，或用 `feature_flags={"self_evolution": False}` 关闭闭环。

## Provider 支持

| Provider | 接口形态 | 说明 |
|---|---|---|
| Ollama | `/api/generate` | 本地模型；不使用 CodeY prompt cache 参数 |
| OpenAI-compatible | `/v1/responses` | 支持 JSON/SSE 兼容响应，能力取决于服务端 |
| Anthropic-compatible | `/v1/messages` | 兼容 HTTP 适配器，不等同官方 SDK 完整能力 |
| DeepSeek | Anthropic-compatible endpoint | 使用独立 DeepSeek 配置 |

## Evaluation 与指标

主要模块：

- `CodeY/evaluation/evaluator.py`：固定 benchmark、fixture、verifier、预算和停止原因检查。
- `CodeY/evaluation/metrics.py`：记忆/上下文/安全/恢复实验与报告渲染。
- `CodeY/evaluation/real_skill_routing.py`：真实外部模型的多 Skill 选择、解析和计分。
- `scripts/run_real_skill_routing_experiment.py`：运行 5/15/25/50/100 Skill 对照实验。
- `scripts/run_provider_experiments.py`：provider 实验。
- `scripts/collect_resume_metrics.py`：聚合 benchmark 与 run artifacts。
- `scripts/run_large_scale_experiments.py`：生成完整实验产物。

### 真实模型多 Skill 命中率

真实评测数据位于 `benchmarks/real-skill-routing/`：

- `skills.json`：100 个真实软件工程 Skill，覆盖架构、前后端、数据、测试、交付、运维、安全和专项平台。
- `requests.json`：100 条单 Skill 请求、10 条多 Skill 请求和 5 条拒识请求，包含中文、英文和混合技术表达。
- `description-rules.md`：领域互斥 Description 的七项编写指南、标准模板与自动校验边界。
- 前 5 条单 Skill 请求固定为 anchor，在所有规模重复使用，用来观察新增干扰 Skill 导致的退化。

实验固定覆盖 5、15、25、50、100 个 Skill，并使用 `.env` 中 `CODEY_PROVIDER` 对应的真实外部模型。每个规模运行两种条件：

- **flat_full**：向模型一次性提供所有完整 Skill 定义和通用工作说明。
- **structured_index**：按领域组织同一批 Skill 的紧凑索引，只保留选择阶段需要的职责信息。

```bash
# 使用 .env 中的 provider/model 执行完整对照实验
python scripts/run_real_skill_routing_experiment.py

# 先验证最小规模，或显式覆盖 provider
python scripts/run_real_skill_routing_experiment.py --scales 5
python scripts/run_real_skill_routing_experiment.py --provider anthropic
```

脚本默认启用断点续跑，并把大规模请求按最多 25 条一批提交；可以使用 `--no-resume`、`--batch-size`、`--timeout` 和 `--delay-seconds` 调整运行策略。

结果写入：

- `artifacts/real-skill-routing/results.json`：原始模型响应、逐请求预测、usage、延迟和完整计分。
- `artifacts/real-skill-routing/results.md`：按规模与模式汇总的可读报告。
- `artifacts/real-skill-routing/skills/*/SKILL.md`：由固定目录物化出的 100 个可解析 Skill 文件。

核心指标包括 exact set match、固定 anchor accuracy、单 Skill accuracy、多 Skill exact match、拒识 accuracy、micro precision/recall/F1、误触发数、漏召回数、prompt 字符数和调用延迟。外部模型调用不进入默认 pytest；测试只使用固定 JSON 验证数据、提示词、解析器和计分逻辑。

其他实验脚本仍需要相应 benchmark、fixture、run root 和输出路径。

## 开发与验证

```bash
python -m pytest -q
python -m ruff check CodeY tests scripts
python -m compileall -q CodeY
python -m CodeY --help
```

测试使用临时 workspace 与 `FakeModelClient`，覆盖双层路由、渐进加载、非法路径、SessionStart 生命周期、XML 边界、上下文预算、同 session 重路由和包入口。
