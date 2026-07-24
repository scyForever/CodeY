# 规则治理与隔离试验

## 产品边界

CodeY 的主场景是现有 Coding Agent 的辅助层：读取仓库规则、生成可审查 Patch、记录灰度 exposure，并把受限任务交给内置 CodeY 或本机已有 Codex/Claude。它不实现 Codex/Claude 的完整替代品，不覆盖用户全局配置，也不把某个工具的权限语义假定为另一个工具的权限语义。

仓库规则治理与原有任务后认知演进是两个独立系统：

| 维度 | Rule Patch | Cognitive Patch |
|---|---|---|
| 输入 | 仓库规则文件与配置 | CodeY 结构化 runtime trace |
| 目的 | 跨 Agent 规则整理、试验、发布 | 内置 Runtime 行为/知识演进 |
| 存储 | `.codey/rules/` | `.codey/evolution/` |
| 发布 | 始终显式人工批准 | Policy/Definition 人审，部分类型可 canary |
| 外部 Agent 证据 | 独立 trial observation | 不接收 |

## 工作流

```text
scan (read-only)
  -> RuleInventory
  -> plan
  -> review_required RulePatch + adapter diffs
  -> baseline / candidate / deterministic canary trial
  -> explicit apply
  -> active
  -> explicit rollback when needed
```

`scan` 识别层级化 `AGENTS.md`、`CLAUDE.md`、`GEMINI.md`、Copilot instructions、Cursor rules/skills、Claude rules/skills 和 Codex/Claude 项目配置。每份来源包含真实仓库相对路径、生态、类型、scope、优先级、原始字节 hash、大小和 `repository_untrusted` 标记。

`plan` 只选择 instruction/rule/skill，不把 hook 或执行配置混入提示词。相同内容会去重，但跨来源的语义冲突不会假装由字符串规则解决；生成块保留 source path、scope 和 hash，等待人工审查。secret-shaped 内容、UTF-8 错误、路径逃逸、单文件或总量超限会阻断计划。

四个固定 adapter 目标是：

- CodeY：`.codey/rules/active.md`，仅在 active v2 Patch 的 candidate hash 精确匹配时作为独立、可审计的 prompt section 加载；手工写入或事后篡改会显示为 unavailable。
- Codex：根 `AGENTS.md` 的 CodeY managed block。
- Claude：根 `CLAUDE.md` 的 CodeY managed block。
- Cursor：`.cursor/rules/codey-managed.mdc` 的 always-apply managed block。

Patch schema v2 保存目标修改前的原始字节、候选 UTF-8 内容、统一 diff 和双方 hash。32 位十六进制 Patch ID 是 128-bit 内容身份，绑定 revision、计划时 dirty 快照、来源元数据、inventory issues 以及每个目标的 before/candidate hash；加载时会重算内容、diff 与身份。`apply` 先重新验证所有来源与目标，再在仓库级跨进程锁内原子写入；`rollback` 只在目标仍精确等于候选 hash 时恢复原始字节。

apply/rollback 在写入前保存 write-ahead journal。中断恢复只接受目标仍处于该 Patch 的 before 或 candidate 状态；若恢复期间出现第三种内容，会保留 journal 并报告冲突，不覆盖新修改。Patch 状态已进入终态时仍会核验所有目标 hash，再清理 journal。

## 灰度与交付

`trial` 是 inspect-only。候选规则会在 detached worktree 中物化；进程退出后若出现任何 Git 可见文件变化，结果为 `unexpected_changes`。`baseline` 不物化候选，`candidate` 总是物化，`canary` 使用 `patch_id + runner + cohort_key` 的稳定哈希按比例选择 exposure。这个选择只证明实际 exposure，不证明因果改进。

`delegate` 只支持 Codex/Claude edit，仍在 detached worktree 中运行。结果按 changed files、diff lines 和 diff bytes 限额；超限或 secret-shaped diff 不会保存可应用补丁。成功 diff 位于 `.codey/rules/trials/<trial-id>/changes.patch`，当前工作树不会自动应用它。

| Runner | Inspect | Edit/delegate | 关键限制 |
|---|---|---|---|
| CodeY | `approval=never`、Skill off | 使用原有 `codey` 单次任务，不走 rule delegate | 内置 `trial` 只读 |
| Codex | `codex exec --sandbox read-only --ephemeral` | `workspace-write` detached worktree | 忽略用户配置，显式 `approval_policy=never` |
| Claude | `permission-mode=plan`、只允许 Read/Glob/Grep | `acceptEdits`、只允许 Read/Glob/Grep/Edit/Write | 项目 hook 置空，不允许 Bash/MCP/Chrome，不持久化 session |

Runner 使用参数数组与 `shell=False`，不会复用通用 `run_shell`；Codex/Claude task 经 stdin 传递，不出现在命令行参数中。子进程环境使用显式 allowlist，只保留启动、临时目录和本机 CLI 认证配置所需变量。timeout 会终止进程树；stdout+stderr 使用合计字节上限并做环境 secret 与 secret-shaped 文本清理。审计记录 runner/version、base revision、variant、mode、cohort/task hash、退出状态、时延、diff 统计、预算违规和 artifact 路径，不保存 cohort key 原文。

detached worktree 来自 Patch 记录的提交 revision，不包含当前未提交代码。若 `dirty_at_plan=true`，trial/delegate 默认拒绝；`--allow-dirty-base` 只表示用户明确接受“规则来自计划快照、代码来自已提交 revision”这一差异，不会复制未提交代码。只要任一 adapter 目标在计划时 dirty，即使提供该开关也会拒绝，以免 candidate 混入未提交目标内容。

## 安全边界

- 仓库文本只是 untrusted evidence，不能授予修改用户全局 Codex/Claude 配置、凭据、插件或工作区外路径的权限。
- CodeY 的内部 Git 命令强制空 hooks、忽略 system/global Git 配置并中和已配置的本地 filter driver；仓库 `.codex/config.toml` 会在副本中替换为固定安全配置。
- apply/rollback 是唯一修改真实规则目标的 rules 命令，并要求 `--approve` 与 exact hash。
- 外部 Agent 的模型调用仍可能访问其厂商服务。CodeY 不能替代 provider 的保留策略、网络隔离或组织策略。
- Patch identity/fingerprint 用于去重和 stale 检查，不是密码学签名或审批人身份认证。
- `.codey/rules` 是本地审计工件，不应当作多人并发数据库；apply/rollback 已有跨进程锁与 journal，但仍没有签名、RBAC 或双人批准。

## 已知残余风险

1. 当前“整理”保留并按 scope 标注原始规则，不做可靠的自然语言矛盾证明；冲突仍需人工审查。
2. Cursor/Codex/Claude 的加载顺序与版本会变化；CLI `--version` 探测成功不等于每个 flag 在未来版本语义不变。
3. Claude edit 模式没有 OS 级 sandbox，因此禁用 Bash、MCP 和网络工具；这也意味着它不能在 delegate 内运行测试命令。
4. Codex workspace sandbox、Claude permission mode 和 CodeY `approval=never` 不是等价控制，试验结果必须按 runner 分开解释。
5. 输出清理只能处理已知环境 secrets 与常见 secret 形态，不能保证识别仓库中的所有凭据或个人信息。
6. 一次成功 trial 只是一条 observation；当前没有自动 evaluator、置信区间、因果对照结论或自动 promote。
7. 外部 Agent 可能创建被 Git ignore 的文件，当前 diff 只统计 Git 可见的 baseline 增量；试验副本会销毁，但该行为不会进入 changes.patch。
8. CodeY 管理的 apply/rollback 会串行化，但外部编辑器仍可能在 hash 校验与 replace 之间改变路径；普通冲突会 fail closed，符号链接/目录替换的 TOCTOU 仍是本机对抗场景下的残余风险。
9. 本机 Codex/Claude 为调用厂商服务仍需访问各自的认证配置；环境 allowlist、禁用 Bash/MCP 和 detached worktree 不能等价为凭据沙箱。对高度不可信仓库，应使用一次性本机 profile/凭据与额外系统级隔离。

因此当前发布结论是：适合本机、单操作者、明确仓库边界下的规则盘点、审查与隔离试验；不应表述为已解决企业级策略分发、强沙箱或自动因果优化。
