# 规则 + LLM 自进化闭环风险登记

更新日期：2026-08-11
适用范围：`CodeY/evolution/cognitive.py`、`CodeY/evolution/hybrid.py`、Runtime 接入、Patch 存储与知识视图。

本文记录本轮实现与复核中识别出的全部已知风险。它不是“功能已经失效”的结论，而是对当前证据边界、残余风险和后续治理工作的诚实说明。状态含义如下：

- **未解决**：当前实现没有直接控制措施。
- **部分缓解**：已有护栏，但不能消除该风险。
- **已解决**：原缺陷已有代码和回归测试覆盖；条目保留用于审计追溯。
- **设计边界**：出于保守性主动接受的能力限制。
- **验证缺口**：代码已有实现，但尚未获得相应环境或规模的验证。

## 已落地的主要护栏

当前实现已经具备以下基础保护，但这些保护不应被解释为“风险已清零”：

- 默认使用纯规则模式；只有显式启用 `hybrid` 才调用 Advisor。
- Evidence Packet 不包含原始命令、工具输出、代码 diff 或模型推理。
- Outcome 的硬规则、Root Cause 候选集合、Patch 类型/scope/kind 和 Safety Gate 状态迁移由规则掌握。
- LLM 建议必须通过严格 JSON schema、置信度、真实 evidence id、证据相关性和触发条件子集校验。
- 已知 secret、diff、控制字符和命令形态文本会被拒绝；非法输出和 provider 异常回退到规则结果。
- Policy 和 Definition 必须进入人工审核；Strategy、Action Chain 和 Experience 只能先进入 shadow。
- Patch 使用规则候选指纹稳定身份，避免仅因 LLM 改写措辞而无限生成重复 Patch。
- Patch 分开记录 eligible、exposed、triggered、success 和 harmful；只有实际渲染的 Patch 才属于 exposed。
- 审计只保存模型标识、prompt 版本/hash、结构化结果、证据引用和错误码，不保存 Advisor 原始响应。

## A. 结果正确性与认识论风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| EPI-01 | 高 | 未解决 | `correct` 当前表示任务正常结束且没有结构化工具失败，不代表用户目标、业务约束或外部验收真正通过。错误答案可能被当作成功。 | Outcome 使用确定性终止信号，但 Evidence Packet 尚未接入 acceptance criteria、测试/verifier 结果或用户确认。 |
| EPI-02 | 高 | 部分缓解 | `harmful` 主要依赖 `partial_success + workspace_changed`。未上报 metadata 的数据损坏、隐私泄露、错误外部操作或语义伤害可能漏检。 | 已有结构化 harmful 硬规则，但覆盖面受 Tool metadata 完整性限制，LLM 也不能补写没有结构化证据的 harmful。 |
| EPI-03 | 中 | 部分缓解 | Root Cause 候选由启发式规则产生：安全码倾向 Policy、重复调用倾向 Strategy、多工具失败倾向 Chain、单工具失败倾向 Execution。复杂故障可能被粗分或错分。 | LLM 可在候选中消歧，但不能选择规则没有生成的层级，因此候选召回率是诊断上限。 |
| EPI-04 | 高 | 部分缓解 | evidence id 有效、且与规则证据集合有交集，只证明“引用了现场证据”，不证明修正与成功之间存在因果关系。 | 已拒绝伪造和明显无关引用；尚无反事实验证、因果图或独立 verifier。 |
| EPI-05 | 高 | 部分缓解 | LLM 仍可能用真实证据生成语义错误、与规则意图相反或过度概括的 `correction.action`。结构合法不等于内容正确。 | 类型、scope、kind、trigger 和状态不可改，Strategy/Chain/Experience 需 shadow；但 correction 的语义尚无形式化 contract。 |
| EPI-06 | 中 | 设计边界 | 混合 Outcome 只能把歧义的 `partial` 保持为 `partial` 或收紧为 `incorrect`，不能根据 LLM 建议升级为 `correct`，也不能仅凭语义判断为 `harmful`。 | 这是为避免猜测性学习而采取的保守限制，会降低 Advisor 覆盖率。 |
| EPI-07 | 中 | 未解决 | 新知识仍主要等价于“成功访问过的仓库路径”，且每次最多保留前两个事实；Definition 依赖文件名中的 architecture/design/boundary/overview/system 标记。它不能证明接口语义、架构边界或实现价值，并可能遗漏同次任务后面的有效事实。 | Advisor 只看到裁剪后的结构化 trace，不读取文件内容，因此无法可靠弥补该语义缺口。 |
| EPI-08 | 低 | 设计边界 | 最终回答、模型推理和原始工具输出不作为学习证据，因此一些真实但未结构化的新知识不会被沉淀。 | 这是避免幻觉、secret 和临时状态进入知识库的主动取舍。 |
| EPI-09 | 中 | 部分缓解 | 稳定规则候选指纹会冻结同一候选首次入库的 correction。后续模型给出更好的同义改写不会自动更新；旧 schema Patch 也不会被新 Advisor 静默重写。 | 避免 Patch 爆炸和无监督改写，但需要显式 Patch 修订/版本升级工作流。 |
| EPI-10 | 中 | 设计边界 | LLM 返回 `patch_eligible=false` 只会阻止 Advisor 精炼，不会否决确定性规则已经生成的 Patch。调用方若把该字段理解为“不会创建 Patch”会产生治理误判。 | 这是“LLM 无权否决规则”的结果，需要在 API contract 和 UI 中明确区分 `advisor_refinement_eligible` 与 `rule_patch_eligible`。 |
| EPI-11 | 中 | 未解决 | 规则候选身份不包含 source run、evidence 或完整 root-cause trigger。不同故障如果落到相同 type/scope/correction/trigger，可能被合并成同一 Patch，丢失故障差异。 | 稳定身份抑制重复 Patch，但当前缺少显式的“可合并等价性”定义。 |

## B. Shadow、命中统计与因果评估风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| EVA-01 | 高 | 未解决 | shadow 不是离线观察：被选中的 guidance 会真实进入 prompt，在激活前就可能改变 Agent 行为并造成伤害。 | harmful 命中后可立即 expired，但首次有害暴露无法预先撤销。 |
| EVA-02 | 高 | 未解决 | “暴露”和“命中”定义不一致。Patch 已注入 prompt，但若执行 trace 未出现声明的工具/路径则不计 hit；反之，出现触发器也不能证明模型实际遵循了 guidance。 | 当前 hit 是事后 trace 匹配，不是可观测的策略采用信号。 |
| EVA-03 | 高 | 已解决 | 旧实现曾把未渲染的第 9 条及后续 Patch 当作 exposed/hit 候选。 | 当前 `exposed_patch_ids`、`active_patch_ids` 和 `shadow_patch_ids` 只来自实际渲染的前 8 条；eligible 但未 exposed 的 Patch 只增加 eligible 计数。 |
| EVA-04 | 高 | 未解决 | 多个 Patch 同时注入时，只要各自触发，就会共享同一个任务 Outcome；系统无法区分成功或失败由哪条 Patch 导致，也无法识别 Patch 交互。 | 尚无单 Patch 隔离实验、互斥分桶或多变量归因。 |
| EVA-05 | 高 | 未解决 | 激活依据是命中率和命中任务成功率，没有未使用 Patch 的同 scope 对照组、uplift 或置信区间。简单任务、时间趋势和其他改动都可能造成伪提升。 | 当前指标适合运行监控，不足以单独证明因果有效性。 |
| EVA-06 | 中 | 未解决 | 默认最少 3 次命中即可激活，样本很小；全局阈值也没有按 Patch 类型、风险或 scope 难度校准。 | 可通过配置提高阈值，但配置校验只检查范围，不检查统计合理性。 |
| EVA-07 | 高 | 部分缓解 | 触发条件已经改为严格 AND，可表达“工具 + 路径 + 错误码”同时满足；但 `task_scope=workspace` 仍是显式全工作区通配条件，过宽规则会扩大命中范围。 | 未知 signal 会判为 false，Advisor 仍只能把条件缩小到规则条件子集；需要继续限制通配 scope 的生成条件。 |
| EVA-08 | 中 | 未解决 | 运行前 scope 只匹配 `skill_name` 和 `route_id`，`target_tool` 尚不可用，因此同 route 的无关任务也可能收到工具相关 guidance。 | 事后可能不计 hit，但 prompt 已经受到影响。 |
| EVA-09 | 中 | 部分缓解 | 每轮最多展示 8 条 Patch，仍没有风险优先级、语义互斥或近期表现调度。 | 已采用 `least_exposed_first_v1`，优先选择累计 exposed 次数更少的 eligible Patch，消除固定文件名排序导致的长期饥饿；冲突和风险排序仍未解决。 |
| EVA-10 | 中 | 未解决 | Knowledge Experience 在“再次读取同一路径且任务成功”后可能激活，但这不能证明该知识指导对成功有贡献。 | 与 EVA-02、EVA-05 相同，需要可观测采用信号和对照评估。 |
| EVA-11 | 中 | 未解决 | 低频、低质量但未达到 100 次命中的 shadow/active Patch 可能长期不 expired；route 重构后永远不再命中的 Patch 也没有 TTL。 | 当前只有命中次数、成功率和 harmful 驱动的过期，没有时间衰减。 |

## C. LLM、Prompt、安全与隐私风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| SEC-01 | 高 | 部分缓解 | 省略独立 critic 时复用主模型，相同模型可能重复自身偏差，形成“自己执行、自己解释”的相关性错误。 | 支持注入独立 `evolution_llm_client`，但 CLI 默认 hybrid 路径仍复用主模型。 |
| SEC-02 | 高 | 部分缓解 | 文件名、路径、错误码和 metadata 都是不可信字符串，可能包含 prompt injection；active/shadow correction 本身也会重新进入后续 prompt。 | Prompt 明确把 JSON 字符串视为不可信数据，输出有严格 schema；但自然语言模型不能提供形式化隔离保证。 |
| SEC-03 | 高 | 部分缓解 | blacklist/redaction 只能识别已知 secret、diff、命令和控制字符模式。编码、拆分、Unicode 混淆、未知凭证格式、个人信息和新型注入仍可能漏过；同时 blacklist 也会误拒绝合法建议。 | 需要 allowlist、DLP/secret scanner 和对抗测试，不能把正则当作完整安全边界。 |
| SEC-04 | 高 | 未解决 | Evidence Packet 包含仓库路径、工具名和运行拓扑。使用外部 provider 时，这些信息会离开本机；CodeY 不落盘原始 Advisor prompt，不代表 provider 不记录请求。 | 当前没有 provider 级数据保留保证、路径匿名化模式或用户确认门。 |
| SEC-05 | 高 | 未解决 | `.codey/evolution/patches/*.json` 和物化 Markdown 没有签名、MAC、权限验证或完整性检查。本地其他进程/用户篡改后可把恶意 guidance 注入 Agent prompt。 | 原子写只防半写，不防恶意修改、符号链接攻击或离线篡改。 |
| SEC-06 | 高 | 部分缓解 | correction 虽限制为单行且过滤常见危险模式，仍可能包含语义诱导、隐藏编码或与任务无关的指令；Markdown 物化也没有完整转义。 | shadow/人工审核降低风险，但自动激活类型仍存在残余风险。 |
| SEC-07 | 中 | 部分缓解 | Provider 不支持严格 JSON/模型不遵循 schema 时会频繁回退；不同模型、模型升级和温度会改变建议分布。prompt hash 可审计，但不能保证可复现。 | 默认一次 Advisor 尝试且失败即回退，安全但可能使 hybrid 名义启用、实际覆盖率很低。 |
| SEC-08 | 高 | 未解决 | 不同 Patch 之间没有冲突检测、优先级或 supersession。两条分别合法的 correction 可能互相矛盾，并同时进入 prompt。 | 稳定 fingerprint 只解决同一规则候选的措辞重复，不解决语义冲突。 |

## D. Safety Gate 与人工治理风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| GOV-01 | 高 | 部分缓解 | Policy/Definition 必须人工批准，但当前 API 只有 `approve_cognitive_patch(patch_id)`：没有 reviewer 身份、角色授权、审批理由、签名、双人复核、驳回 API 或 UI。 | 状态历史能记录批准发生，但不能回答“谁、为什么、依据什么批准”。 |
| GOV-02 | 高 | 未解决 | Strategy/Action Chain/Experience 是否高风险取决于语义，但 Safety Gate 主要按 Patch type 决策。错误类型归因可能让本应审核的约束进入自动 shadow/active。 | 需要独立 risk classifier 的规则上界和“任何高风险 scope 强制审核”。 |
| GOV-03 | 中 | 未解决 | Policy/Definition 激活需要人工审核，但激活后与其他类型共用自动 expiry 逻辑，可能因低成功率或 harmful 自动失效；这与“宪法层修改必须人工”的治理语义不完全对称。 | 需要明确“自动熔断”和“正式废止”是否应分成两个状态。 |
| GOV-04 | 中 | 未解决 | 阈值配置只做范围校验，允许 `canary_fraction=0`、极低置信度、过小样本量或激活/过期阈值关系不合理。错误配置可让 Patch 永不评估或过早激活。 | 尚无安全配置 profile、关系约束或启动警告。 |
| GOV-05 | 中 | 未解决 | active Patch 没有人工暂停、回滚到 shadow、重新审核或带理由废止的公开工作流；当前状态机过期后也不能恢复。 | 可创建新 Patch，但审计与运维成本较高。 |

## E. 存储、并发、生命周期与审计风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| STO-01 | 高 | 未解决 | Store 使用原子文件替换，但没有跨线程/跨进程锁或 compare-and-swap。并发任务可能丢失 metrics、覆盖状态历史，或在同一候选创建时发生竞态。 | 原子写只能保证单次文件完整，不能保证 read-modify-write 隔离。 |
| STO-02 | 中 | 未解决 | `observed_run_ids` 对每个 eligible run 持续增长；Patch JSON、过期审计记录和物化视图也没有 retention/compaction，长期运行会增加 I/O 和存储。 | 尚无滚动窗口、聚合计数、归档或容量上限。 |
| STO-03 | 中 | 未解决 | 只有命中表现触发过期，没有基于代码版本、route 版本、文件 freshness 或时间的失效机制。代码重构后，旧 Patch 可能继续 active 或成为永久孤儿。 | stale-path 只覆盖当前任务发现的部分文件摘要。 |
| STO-04 | 中 | 部分缓解 | Patch 的 evidence ref 如 `tool_001` 只在 run 内有意义；当前 cognitive report 保存引用，但没有持久化一份显式 `evidence_id -> 规范化事件` 映射。删除或缺失 run trace 后，来源难以独立复核。 | trace 顺序可人工推断，但不是稳定的机器可解析 provenance contract。 |
| STO-05 | 中 | 设计边界 | Patch schema 已升级到 v3，并严格拒绝旧版本、未知/缺失字段和旧 `hit_count` 指标；没有内置迁移器。 | 这是“不隐式兼容旧实现”的主动边界；如需保留历史 Patch，必须在运行时之外显式迁移并重新验证。 |
| STO-06 | 低 | 未解决 | Patch id 只使用规则候选 SHA-256 的前 16 个十六进制字符。碰撞概率很低，但 path 已存在时当前不会再比较完整 rule fingerprint，理论上可能错误合并。 | Patch 同时保存完整 fingerprint，但创建路径尚未做碰撞拒绝。 |
| STO-07 | 中 | 部分缓解 | 物化视图刷新/删除失败会让 cognitive loop 报错并被主任务隔离，主任务仍返回成功；此时 JSON 事实源和 Markdown active view 可能短暂不一致。 | 失败会进入 cognitive error/trace，但没有自动修复队列。 |
| STO-08 | 中 | 设计边界 | 为降低隐私风险，CodeY 不保存 Advisor 原始 prompt/response，只保存 hash 和结构化结果；因此事后无法逐字符重放或判断模型是否输出过被 validator 丢弃的其他内容。 | 可依赖 provider 日志或受控调试采样，但这又会引入新的隐私与保留风险。 |

## F. 性能、成本与可用性风险

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| OPS-01 | 高 | 未解决 | Advisor 在 `_finish_run` 内同步执行，主结果要等认知闭环完成才返回。一次任务最多增加诊断和 Patch 两次调用，且 provider 内部还可能重试/等待长超时。 | Advisor 异常不会改变主任务结果，但可能显著拖延结果交付。 |
| OPS-02 | 中 | 未解决 | 没有独立的 evolution 调用超时、总 token/费用预算、日配额、熔断器或失败退避。持续 schema 失败会在每个歧义任务重复付出成本和延迟。 | 只有单次 `max_new_tokens` 和 Advisor `max_attempts` 上限。 |
| OPS-03 | 中 | 未解决 | 没有相同 Evidence/规则候选的诊断缓存，也没有批量 Advisor；重复故障会反复调用模型。 | 稳定 Patch fingerprint 只去重存储，不去重推理费用。 |
| OPS-04 | 低 | 部分缓解 | guidance 限制为 8 条控制 prompt 体积，但 Patch 数量、每条 correction 长度和 active view 重建仍会随时间增长。 | 有单条长度限制，没有全局 Patch 数或 view 大小预算。 |

## G. 验证与发布证据缺口

| ID | 严重度 | 状态 | 风险与影响 | 当前控制与剩余问题 |
|---|---|---|---|---|
| VAL-01 | 高 | 验证缺口 | 当前 hybrid 测试使用 `FakeModelClient`，没有真实 provider/真实模型的严格 JSON 遵循率、建议质量、token 成本和尾延迟数据。 | 单元与集成回归可以证明边界逻辑，不足以证明线上模型表现。 |
| VAL-02 | 高 | 验证缺口 | 尚无并发、多进程、长时间运行、Patch 数量膨胀和故障恢复压力测试。 | Store 的竞态和容量风险未被实测量化。 |
| VAL-03 | 高 | 验证缺口 | 尚无系统性红队测试覆盖 Unicode/编码 secret、路径 prompt injection、Markdown 注入、有效证据上的错误结论和互相冲突 Patch。 | 已有伪造 evidence、secret-shaped 文本和状态越权测试，只覆盖有限样例。 |
| VAL-04 | 中 | 验证缺口 | 尚无离线 replay benchmark 比较 rules 与 hybrid 的 Outcome 准确率、Root Cause 混淆矩阵、Patch 接受质量和有害建议率。 | 无法用当前测试回答“hybrid 是否比 rules 更准确”。 |
| VAL-05 | 低 | 验证缺口 | Windows pytest 在全部断言通过后仍出现临时目录清理 `PermissionError`。它不影响本轮测试结果，但会污染 CI/本地日志并可能掩盖真正的 teardown 问题。 | 需要单独修复 pytest 临时目录权限/编码环境。 |

## 优先处理顺序

### P0：在扩大 hybrid 使用范围前

1. 修正 `guidance[:8]` 与实际 exposure id 集合不一致的问题，只统计真实渲染的 Patch。
2. 把 exposure、adoption 和 trigger 分成三个信号；未观察到采用时不能把任务结果归因给 Patch。
3. 将规范化 Evidence Packet 或可解析映射持久化，保证 evidence ref 可独立复核。
4. 为 Patch Store 增加文件锁/CAS，并验证多进程 metrics 更新。
5. 为外部 provider 增加路径匿名化/隐私模式和显式数据发送说明。
6. 补充 reviewer 身份、理由、驳回、暂停、废止和权限控制。

### P1：在允许自动激活前强化

1. 引入 acceptance criteria、真实 verifier 和用户反馈，收紧 `correct` 的定义。
2. 建立未暴露对照组、uplift、置信区间和单 Patch 隔离实验。
3. 支持 AND trigger、target-tool exposure gating、Patch 冲突检测与优先级。
4. 对高风险 scope 强制 `review_required`，不要只依赖 Patch type。
5. 增加 TTL、代码/route 版本绑定、retention 和 observed-run 聚合。

### P2：工程化与证据建设

1. 增加 Advisor 专用超时、成本预算、熔断、缓存和异步后处理选项。
2. 建立真实模型 replay benchmark、红队集和 provider 兼容矩阵。
3. 增加 Patch 签名/完整性检查、schema 迁移和事实源/物化视图修复工具。
4. 解决 Windows pytest 临时目录清理警告，保持验证日志可判读。

## 当前发布结论

当前实现适合作为**默认关闭 LLM、显式 opt-in、规则兜底的实验性混合闭环**。在 P0 风险关闭前，不应把 shadow/active 指标解释为因果收益，也不应把自动激活 Patch 视为已获得生产级安全证明。Policy/Definition 的人工审核闸门应继续保留，优先使用独立 critic，并避免向不具备数据保留保证的外部 provider 发送敏感仓库 Evidence Packet。
