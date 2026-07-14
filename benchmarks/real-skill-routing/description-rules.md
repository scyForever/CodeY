# Skill Description 编写与校验规则

本目录的 Description 用于模型选择 Skill，而不是描述执行步骤。目标是在多 Skill 目录中形成领域互斥、自然语言化、可审计的激活边界。

## 标准模板

```text
This skill should be used when <user context and domain condition>, and the core intent matches "<exclusive English trigger phrase A>" or "<exclusive English trigger phrase B>". <Optional scope boundary.> It should not activate for <near miss A>, <near miss B>, or <near miss C>.
```

## 七项指南

| 序号 | 规则 | 本数据集的约束 |
|---:|---|---|
| 1 | 足够的长度 | 至少 20 个英文单词。 |
| 2 | 使用约定语言 | Description 统一使用英文且不得包含中日韩字符；真实请求可使用中文，用于测量跨语言激活。 |
| 3 | 包含带引号的触发短语 | 每条描述恰好包含两条 ASCII 双引号包裹的英文短语；全目录 200 条短语必须唯一。 |
| 4 | 第三人称格式 | 描述必须以 `This skill should be used when` 开头。 |
| 5 | 包含激活条件 | 描述必须说明用户请求的上下文和核心领域意图，不能只写“用于某类工作”。 |
| 6 | 不要枚举工作流 | 禁止首先、其次、然后、最后或第几步等步骤化表达；工作流信息应留在 route 表和 workflow 文件中。 |
| 7 | 命名近似未命中反触发 | 描述必须以 `It should not activate for A, B, or C.` 格式列出三个近似但不应激活的场景。 |

## 自动校验与人工审阅

`CodeY.evaluation.real_skill_routing.validate_skill_descriptions` 校验长度、双语短语、句式、反触发、步骤枚举和短语重复。自动校验不能判断语义是否真正互斥；修改后仍应大声朗读，并人工检查相邻 Skill 是否争夺同一种用户意图。

物化为 `skills/*/SKILL.md` 后，可用下列命令辅助检查重复触发短语：

```bash
grep -h '"' skills/*/SKILL.md | sort | uniq -d
```

该命令只能发现字面重复，不能代替语义边界审阅。
