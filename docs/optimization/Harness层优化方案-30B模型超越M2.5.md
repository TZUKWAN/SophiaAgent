# Sophia + Aurora Harness 层优化方案：30B模型超越M2.5+Claude Code

## Context

Sophia Agent（学术研究垂域，146+工具）和 Aurora Agent（竞赛咨询垂域，40+工具）目前都针对大模型（GPT-4/Claude级）设计Harness层。关键问题：
- 每次请求发送所有工具schema（Sophia ~20000 tokens，Aurora ~7000 tokens）
- 系统提示过长（Aurora ~5000 tokens含完整白玉方法论）
- 全局temperature=0.7，对工具调用场景过高
- 上下文窗口未利用GLM-4-Flash的128K能力
- 工具schema嵌套3-4层，小模型生成困难
- BP生成10次串行LLM调用，小模型在后续调用中丢失上下文

**目标**：通过Harness层优化，使GLM-4.7-Flash（30B MoE，~3-10B激活参数，128K上下文）在Sophia/Aurora各自的垂域场景中，整体表现超过MiniMax M2.5（230B MoE，10B激活）配合Claude Code的组合。

**核心策略**：Harness承担推理编排，模型承担执行。通过动态工具过滤、提示压缩、Schema扁平化、温度策略、批量生成、异构路由等手段，将每次请求的token开销降低60-70%，同时提升工具调用准确率和输出质量。

---

## 理论依据

1. arXiv 2512.15943v2：350M参数模型经SFT后ToolBench通过率77.55%，远超ChatGPT的26%。**针对性训练 > 暴力缩放**
2. Galileo研究：小模型prompt敏感度192%，大模型仅7%。**prompt工程ROI对小模型极高**
3. NVIDIA Research (arXiv 2506.02153)：推荐异构架构——SLM处理80%常规任务，LLM回退20%复杂推理
4. EACL 2024：小模型通过few-shot CoT可以学会大模型的推理模式
5. Google web.dev：小模型每次最多暴露3-5个工具，参数层级<=2，temperature 0.0-0.1用于工具调用

---

## Token预算对比

### Aurora（优化前 vs 优化后）

| 组件 | 当前 | 优化后 | 节省 |
|------|------|--------|------|
| System Prompt | ~5000 | ~1500 | 3500 |
| Tool Schemas (40个) | ~7000 | ~1000 | 6000 |
| Instruction Files | ~300 | ~300 | 0 |
| Conversation History | ~2000 | ~4000 (keep_recent=20) | -2000 |
| **Total per request** | **~14300** | **~6800** | **~7500 (52%)** |

### Sophia（优化前 vs 优化后）

| 组件 | 当前 | 优化后 | 节省 |
|------|------|--------|------|
| System Prompt + Appendix | ~1500 | ~800 | 700 |
| Tool Schemas (146个) | ~20000 | ~1500 | 18500 |
| Task Harness | ~3000 | ~500 | 2500 |
| Workspace Context | ~30000 | ~30000 | 0 |
| **Total per request** | **~54500** | **~32800** | **~21700 (40%)** |

---

## Phase 0：共享优化框架（新包 `slm_harness/`）

**新建** `D:/SophiaAgentWork/slm_harness/`，6个模块：

| 模块 | 职责 |
|------|------|
| `intent_router.py` | 关键词意图分类，返回意图类别+推荐工具+温度+模型层级 |
| `tool_filter.py` | 根据意图类别筛选3-8个相关工具schema |
| `schema_flattener.py` | 将嵌套JSON Schema扁平化到2层以内 |
| `temperature_strategy.py` | 工具调用0.0 / 分类0.1 / 生成0.4 / 综合0.3 |
| `output_parser.py` | 5级容错JSON解析（直解->去代码块->正则提取->KV提取->空dict） |
| `context_optimizer.py` | 128K窗口管理+按需RAG检索 |

两个项目各自维护领域映射配置（Aurora: 竞赛/BP/评估/答辩/团队/可视化/大创；Sophia: 文献/写作/分析/引用/数据/NLP/设计/伦理/评审/演示/翻译）。

---

## Phase 1：动态工具过滤（P0，影响最大）

**原理**：30B模型看到40+工具选项会决策瘫痪，嵌套JSON Schema导致参数生成错误。每次只发送3-8个相关工具。

**Aurora修改**：
- `aurora/agent.py`：`_process_message()`中，`self.tools.get_schemas()` -> `ToolFilter.select_tools(user_input, self.tools.get_schemas())`
- `aurora/config.py`：`ModelConfig`新增`tool_filter_enabled: bool = True`
- 新建`aurora/tool_domains.py`：定义8个领域->工具映射

**Sophia修改**：
- `sophia/agent.py`：`run()`中，当`allowed_tools is None`时调用ToolFilter
- `sophia/config.py`：新增`tool_filter_enabled: bool = True`
- 新建`sophia/tool_domains.py`：定义11个领域->工具映射

**预期效果**：Aurora从7000 token降至~1000 token。Sophia从20000 token降至~1500 token。

---

## Phase 2：Schema扁平化（P0）

**原理**：嵌套3-4层的JSON Schema对30B模型是灾难。扁平化到2层以内，参数上限6个。

**实现**：
- `slm_harness/schema_flattener.py`：`flatten_schema(schema) -> schema`
  - 嵌套object -> 扁平字符串字段（`project_info.tech` -> `tech_summary`）
  - 移除`oneOf`/`anyOf`，改为description中的文字说明
  - 截断description到40字符
  - 每个工具最多6个参数
- Aurora `aurora/tools/registry.py`：`get_schemas(compact=True)`时自动扁平化
- Sophia `sophia/tools/registry.py`：同上
- `compact_mode`由config控制，大模型用原始schema，小模型用扁平schema

**具体schema简化示例**（Aurora `dachuang_generate`）：
- 当前：12个参数，含嵌套object
- 优化后：6个核心参数（project_name, project_type, innovation_point, budget_range, team_info, additional_requirements），均为string类型

---

## Phase 3：温度策略（P0，改动最小收益最大）

**原理**：工具调用temperature=0.0-0.1，内容生成=0.3-0.5。当前全局0.7对小模型是灾难性的。

**Aurora修改**：
| 文件 | 当前 | 优化后 |
|------|------|--------|
| `agent.py` `_call_llm` | 0.7 | 工具轮0.0，最终轮0.4 |
| `agent.py` `_synthesize_tool_response` | 0.7 | 0.3 |
| `business_plan/generator.py` | 0.7 | 0.4 |
| `evaluation/engine.py` | 0.5 | 0.3 |
| `swarm/decomposer.py` | 0.3 | 0.1 |
| `context.py` | 0.3 | 0.1 |

**Sophia修改**：
- `providers/openai_compat.py`：`chat()`方法根据`tools`参数是否存在自动选择temperature
- `agent.py`：传递温度覆盖值

---

## Phase 4：上下文窗口优化（P1）

**Aurora**：
- `config.py` `ContextConfig`：新增`context_window: int = 128000`
- `context.py`：使用config值替代硬编码32000
- `keep_recent`从5增加到20
- `compress_threshold`从0.65增加到0.80

**Sophia**：
- `config.py`：`max_context_tokens`默认128000
- `context.py`：读取config值
- `keep_recent`增加到20

---

## Phase 5：输出解析鲁棒性（P1）

**原理**：30B模型比大模型更容易生成格式错误的JSON参数。需要5级容错解析。

**新建** `slm_harness/output_parser.py`：
```
RobustToolCallParser.parse_arguments(raw):
  1. json.loads() 直解
  2. 去markdown代码块 ```json...```
  3. 正则提取 {.*} JSON块
  4. key=value键值对提取
  5. 返回空dict + error flag
```

**修改**：
- Aurora `agent.py` `_execute_tool_calls`：`json.loads(arguments)` -> `RobustToolCallParser.parse_arguments(arguments)`
- Sophia `providers/openai_compat.py`：同上

---

## Phase 6：提示压缩与RAG迁移（P2）

**Aurora**（SYSTEM_PROMPT从~5000 token压至~1500 token）：
- `prompts/system.py`：创建`SYSTEM_PROMPT_COMPACT`（~40行）
  - 保留核心身份、红线、基本规则
  - 删除详细白玉方法论、评分标准、五道槛、避坑指南（移至RAG）
  - 删除完整工具列表（Phase 1已动态过滤）
- 新建`aurora/knowledge/methodologies/baiyu_rag.json`：按需检索的知识块
- `agent.py` `_build_messages`：根据config选择compact/full版本

**Sophia**（压缩Autopilot附录和Task Harness）：
- `autopilot.py`：`AUTOPILOT_SYSTEM_APPENDIX`从20行压至5行
- `task_harness.py`：复杂任务不再注入完整harness文本，改为2行摘要+工具调用检索

---

## Phase 7：批量生成替代串行调用（P2，仅Aurora）

**当前**：`BusinessPlanGenerator.generate()` 10次串行LLM调用，temperature=0.7

**优化为3次调用**：
1. **大纲生成**（1次调用）：temperature=0.1, max_tokens=4096 -> 生成10个章节的要点大纲
2. **批量生成Part 1**（1次调用）：前5章（executive_summary -> marketing_strategy），temperature=0.4, max_tokens=16384
3. **批量生成Part 2**（1次调用）：后5章（operation_plan -> risk_assessment），temperature=0.4, max_tokens=16384

**修改**：`aurora/business_plan/generator.py`：`generate()`方法重构，保留旧方法作为`_generate_section_by_section()`供大模型使用。

---

## Phase 8：异构路由（P3，小模型主用+大模型兜底）

**原理**：80%请求（工具调用、信息检索、格式化）用GLM-4.7-Flash，20%（完整BP、深度评估、复杂推理）用DeepSeek-V3或Qwen3-235B。

**实现**：
- `aurora/config.py` `ModelConfig`：新增`fallback_name: str = ""`
- `sophia/config.py`：同上
- Intent Router的输出包含`model_tier: "small" | "large"`
- Agent的`_call_llm`根据tier选择model
- Swarm编排和BP生成自动使用large tier

---

## Phase 9：意图路由统一（P2）

**替换**两个项目中现有的碎片化意图检测（Aurora的Swarm trigger + Sophia的AutopilotRouter + TaskHarness），统一为`slm_harness/intent_router.py`。

输出：`IntentResult(category, relevant_tools, temperature, model_tier, system_prompt_variant)`

---

## 实施顺序与依赖

```
Phase 0 (共享框架) <- 一切的前提
  |
  +-> Phase 1 (工具过滤) + Phase 2 (Schema扁平) + Phase 3 (温度策略) <- 可并行，P0级
  |
  +-> Phase 4 (上下文窗口) + Phase 5 (输出解析) <- 可并行，P1级
  |
  +-> Phase 6 (提示压缩) + Phase 9 (意图路由) <- 可并行，P2级
  |
  +-> Phase 7 (批量生成) <- 仅Aurora，P2级
  |
  +-> Phase 8 (异构路由) <- P3级，需要两个API key
```

---

## 验证策略

### 每Phase完成后
1. `pytest tests/ -v` 全量回归
2. 小模型（GLM-4.7-Flash）工具调用准确率测试
3. 对比优化前后的token消耗

### 全部完成后
1. 创建A/B评估套件：20个典型prompt（Aurora 10个 + Sophia 10个）
2. 对比矩阵：GLM-4.7-Flash优化Harness vs GLM-4.7-Flash原始Harness vs MiniMax M2.5 vs Claude Code
3. 指标：工具调用准确率、任务完成率、token消耗、延迟、输出质量（人工评估）

---

## 关键文件清单

| 文件 | 改动类型 |
|------|----------|
| `slm_harness/*.py` (6个新文件) | 新建 |
| `aurora/agent.py` | 修改（工具过滤+温度+解析） |
| `aurora/config.py` | 修改（新增配置项） |
| `aurora/tools/registry.py` | 修改（compact mode） |
| `aurora/prompts/system.py` | 修改（compact版提示词） |
| `aurora/business_plan/generator.py` | 修改（批量生成） |
| `aurora/context.py` | 修改（128K窗口） |
| `aurora/tool_domains.py` | 新建 |
| `sophia/agent.py` | 修改（工具过滤+温度） |
| `sophia/config.py` | 修改（新增配置项） |
| `sophia/tools/registry.py` | 修改（compact mode） |
| `sophia/providers/openai_compat.py` | 修改（温度+解析） |
| `sophia/autopilot.py` | 修改（压缩附录） |
| `sophia/task_harness.py` | 修改（RAG迁移） |
| `sophia/context.py` | 修改（128K窗口） |
| `sophia/tool_domains.py` | 新建 |
