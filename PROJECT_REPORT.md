# SophiaAgent 项目报告

> **版本**: 0.1.0  
> **定位**: 面向人文社科领域的 AI 研究助手  
> **代码规模**: 91 个 Python 模块，约 36,500 行源码；65 个测试文件，约 18,800 行测试；总计 1622+ 测试用例全部通过

---

## 一、项目概述

SophiaAgent 是一个专为**人文与社会科学**研究设计的 AI 智能体系统。与通用对话助手不同，它内置了完整的研究工作流——从文献检索、研究设计、数据分析到论文写作与评审——所有环节均可通过自然语言指令驱动完成。

核心设计理念：将人文社科研究中高度重复、规则明确但耗时费力的环节（格式排版、统计检验、文献管理、引用格式化）交给 AI 自动处理，让研究者专注于思想与洞见。

---

## 二、功能全景

### 2.1 八大工具域（84+ 个工具）

| 工具域 | 核心能力 | 代表工具 |
|--------|---------|---------|
| **文件管理** | 工作空间文件读写、目录浏览 | `file_read`, `file_write`, `file_list` |
| **文献检索** | Semantic Scholar / arXiv / Crossref 语义搜索 | `literature_search` |
| **引用管理** | BibTeX 库维护、GB/T 7714 & APA 格式化、引用网络图 | `ref_add`, `ref_format`, `ref_network` |
| **学术写作** | 论文/报告/专著/基金申请书全流程 | `doc_create`, `doc_write_section`, `doc_export_docx` |
| **数据分析** | Pandas + Matplotlib 沙箱，CSV/Excel/SPSS/Stata 支持 | `data_load`, `data_describe`, `data_visualize` |
| **网络采集** | 网页搜索、内容提取、批量爬取 | `web_search`, `web_extract` |
| **学术评审** | 六维度加权评审（真实性、逻辑、引用、语言、统计、伦理） | `doc_review`, `systematic_review` |
| **数据收集** | 宏观经济面板数据、A股数据、新闻采集、网页爬取 | `data_macro`, `data_china_finance`, `data_news` |

### 2.2 九大研究方法论引擎

SophiaAgent 内置了覆盖人文社科主流研究方法的计算引擎：

- **统计推断引擎** (`StatisticalEngine`) — 描述统计、T检验、方差分析、非参数检验、正态性检验、效应量计算
- **研究设计引擎** (`ResearchDesignEngine`) — 实验设计、抽样方案、变量操作化
- **因果推断引擎** (`CausalEngine`) — DID、RDD、PSM、IV、Synthetic Control、Mediation/Moderation
- **调查研究引擎** (`SurveyEngine`) — 问卷设计、信效度分析、因子分析、结构方程模型
- **定性研究引擎** (`QualitativeEngine`) — 编码分析、主题分析、扎根理论、话语分析
- **元分析引擎** (`MetaAnalysisEngine`) — 效应量合并、森林图、发表偏倚检验、亚组分析
- **计算社会科学引擎** (`ComputationalEngine`) — 社会网络分析、文本挖掘、情感分析、主题模型
- **机器学习引擎** (`MLEngine`) — AutoML、特征工程、模型解释 (SHAP)、超参优化
- **可视化引擎** (`VisualizationEngine`) — 学术级图表生成，支持 APA 三线表、OMML 公式

### 2.3 文档导出体系

支持完整的学术文档生产链路：

| 格式 | 技术实现 | 特性 |
|------|---------|------|
| **Markdown** | 原生渲染 | 快速预览、版本控制友好 |
| **LaTeX** | pylatex / 自定义模板 | 数学公式、交叉引用、学术排版 |
| **DOCX** | python-docx + OMML Builder | 原生 Word 公式 (OMML)、APA 三线表、样式模板 |
| **PDF** | XeLaTeX 编译 | 中文支持、矢量图表、印刷级输出 |

DOCX 导出是推荐格式——它支持原生 OMML 数学公式（Word 中可直接编辑），以及符合 APA 规范的三线表，可直接投稿。

### 2.4 学术评审系统

六维度自动化评审，基于 PRISMA 2020 系统综述框架：

1. **真实性评审** — 检测数据造假、异常值、不可复制的结果
2. **逻辑评审** — 论证链完整性、因果推断合理性
3. **引用评审** — 幽灵引用检测、引用格式一致性
4. **语言评审** — 学术用语规范、口语化表达标记
5. **统计评审** — p值矛盾、缺失效应量、样本量不足
6. **伦理评审** — 研究伦理合规性检查

### 2.5 ChatGPT 风格 Web 界面

基于 FastAPI + WebSocket 的现代化 Web UI：

- **流式输出** — WebSocket 实时推送 token，响应零等待
- **消息渲染** — Markdown + KaTeX 数学公式 + highlight.js 代码高亮
- **深色/浅色模式** — 一键切换，localStorage 持久化
- **会话管理** — 左侧边栏按工作空间分组，支持历史会话切换与删除
- **设置面板** — 运行时切换 API Provider、模型、工作空间，无需重启
- **Token 用量** — Header 实时显示累计 token 消耗
- **文件上传** — 拖拽上传，自动存入工作空间

### 2.6 CLI 交互界面

基于 prompt_toolkit + rich 的终端交互：

- 斜杠命令系统（`/help`, `/sessions`, `/checkpoint`, `/resume`）
- 状态栏实时显示 token 用量与模型信息
- 会话编号交互式恢复
- 彩色 Markdown 渲染与进度条

---

## 三、技术路线

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        交互层                                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│   │   CLI 终端    │  │   Web 界面    │  │   Python API     │ │
│   │ prompt_toolkit │  │ FastAPI+WS   │  │   直接调用        │ │
│   └──────────────┘  └──────────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Agent 核心层                            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SophiaAgent (对话循环)                   │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │   │
│   │  │  Hook   │ │  Goal   │ │  SubAgent│ │   Loop    │ │   │
│   │  │ Manager │ │ Manager │ │ Manager  │ │ Manager   │ │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └───────────┘ │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │   │
│   │  │ Memory  │ │ Context │ │Recovery │ │Guardrails │ │   │
│   │  │ Manager │ │Compressor│ │Manager │ │           │ │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └───────────┘ │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │   │
│   │  │Scheduler│ │ Kanban  │ │ Plugins │ │ Security  │ │   │
│   │  │         │ │  Board  │ │ Manager │ │ Manager   │ │   │
│   │  └─────────┘ └─────────┘ └─────────┘ └───────────┘ │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                       工具层                                 │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ │
│   │  File  │ │Research│ │Citation│ │Writing │ │ Analysis│ │
│   └────────┘ └────────┘ └────────┘ └────────┘ └─────────┘ │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────────┐ │
│   │  Web   │ │ Review │ │DataColl│ │   Discovery (自演化) │ │
│   └────────┘ └────────┘ └────────┘ └────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      引擎层                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ │
│   │Statistics│ │ Design │ │ Causal │ │ Survey │ │Qualitative│
│   └────────┘ └────────┘ └────────┘ └────────┘ └─────────┘ │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ │
│   │  Meta  │ │Computational│ │  ML   │ │Visualization│ │
│   └────────┘ └────────┘ └────────┘ └────────┘ └─────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     基础设施层                               │
│   ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │
│   │  Provider  │  │  Session   │  │  ResultStore       │   │
│   │(OpenAI/    │  │ (SQLite)   │  │ (WorkspaceGuard)   │   │
│   │ Anthropic) │  └────────────┘  └────────────────────┘   │
│   └────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心设计决策

#### 3.2.1 多 Provider 抽象层

不绑定单一模型供应商，通过 `BaseProvider` 抽象接口统一封装：

- **OpenAI 兼容** (`openai_compat.py`) — 支持 DeepSeek、Ollama、vLLM、Qwen 等所有 OpenAI API 格式的服务
- **Anthropic** (`anthropic.py`) — 原生 Claude API 支持
- **运行时热切换** — Web 界面中可实时切换 Provider 和模型，无需重启服务

```python
# 统一接口
data = provider.chat(messages, tools=tools, stream=True)
# 底层自动处理不同供应商的 tool calling 格式差异
```

#### 3.2.2 Tool Registry 动态注册

所有工具通过装饰器自动注册到 `ToolRegistry`，schema 自动生成：

```python
@tool_registry.register(
    name="research_did",
    description="Run Difference-in-Differences analysis",
    parameters={...}
)
def research_did(...) -> dict:
    ...
```

这种设计使得新增工具零配置——写好函数即可自动暴露给 LLM。

#### 3.2.3 Workspace 隔离机制

每个工作空间拥有独立的文件系统沙箱：

- **WorkspaceGuard** — 文件访问权限校验，禁止越界读写
- **ResultStore** — 研究结果结构化存储，支持跨工具传递 `result_id`
- **Session DB** — SQLite WAL 模式，按工作空间隔离会话历史

#### 3.2.4 上下文压缩

长对话场景下自动压缩历史消息：

- 保留最近 N 条完整消息
- 旧消息通过 `ContextCompressor` 生成摘要
- Tool 结果和系统指令始终保留
- 阈值触发（默认 65% 上下文占用率）

#### 3.2.5 自演化发现系统

研究方法的"自动生成"能力：

- **MethodCatalog** — 内置方法论知识库（覆盖定量/定性/混合方法）
- **MethodSearcher** — 根据研究问题语义检索最适配的方法
- **MethodBuilder** — 根据已有方法组合生成新方法（依赖管理自动解析）
- **DependencyManager** — Python/R 包依赖自动安装与沙箱隔离

### 3.3 数据流设计

```
用户输入
   │
   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ System Prompt │───▶│  Context     │───▶│   Provider   │
│  + Tools      │    │  Assembler   │    │  (LLM API)   │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ Tool Call?   │──No──▶ 直接输出
                                        └──────────────┘
                                               │ Yes
                                               ▼
                                        ┌──────────────┐
                                        │ ToolRegistry │
                                        │  路由执行     │
                                        └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ ResultStore  │
                                        │  结果存储     │
                                        └──────────────┘
                                               │
                                               ▼
                                        返回结果给 LLM 继续推理
                                        （循环直到无新 tool call）
```

---

## 四、关键特性

### 4.1 写作流水线（Writing Pipeline）

学术研究写作被拆解为可追踪的 7 阶段流水线：

1. **Outline** — 与用户确认大纲结构
2. **Draft** — 逐节撰写（此阶段禁用文献检索，专注写作）
3. **Assemble** — 自动从 ResultStore 组装 Methods 和 Results
4. **Review** — 六维度自动化评审
5. **Revise** — 基于评审结果自动修订
6. **Refine** — 最终润色
7. **Export** — 导出 DOCX / LaTeX / PDF / Markdown

支持一键全自动化：`doc_pipeline_run` 执行 assemble → review → revise → export。

### 4.2 研究方法推荐引擎

用户提出研究问题后，系统自动执行：

1. 调用 `methodology_advise` 分析研究问题特征
2. 返回排序后的方法推荐列表（含适用性评分、前提假设、替代方案）
3. 用户确认后，按推荐顺序自动调用对应研究工具链
4. 最终输出 APA 格式的结果解读（含效应量、置信区间、实际显著性）

### 4.3 技能自动挖掘与进化

系统观察用户的工具使用模式：

- **SkillMiner** — 检测重复出现的 3+ 工具序列
- **SkillFactory** — 将序列封装为可复用模板
- **SkillEvolution** — 当模板多次失败时自动调参优化

### 4.4 安全与防护

多层安全机制：

- **WorkspaceGuard** — 文件系统沙箱，防止越界访问
- **ToolGuardrails** — 工具调用频率限制（每分钟 60 次、连续 5 次上限）
- **SecurityManager** — 命令注入检测、敏感信息过滤
- **CredentialPool** — API 密钥加密存储，内存中脱敏

### 4.5 实验可复现性

- **GlobalSeed** — 全局随机种子控制，确保统计结果可复现
- **ExperimentPipeline** — 实验参数、数据版本、结果全流程追踪
- **SnapshotManager** — 工作空间快照，支持任意时刻回滚
- **Checkpoint** — 会话级检查点，支持对话历史恢复

---

## 五、技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| **语言** | Python 3.10+ | 类型提示、dataclass、match-case |
| **LLM 接口** | openai-sdk, anthropic-sdk | 统一抽象，支持流式/非流式 |
| **Web 框架** | FastAPI + Uvicorn | 异步端点、自动 OpenAPI 文档 |
| **前端** | 原生 JS + marked.js + KaTeX + highlight.js | 零构建工具，CDN 直引 |
| **CLI** | prompt_toolkit + rich | 交互式终端、彩色渲染 |
| **数据科学** | pandas, numpy, scipy, scikit-learn | 统计分析与机器学习 |
| **可视化** | matplotlib, seaborn | 学术级图表 |
| **文档导出** | python-docx, pylatex | DOCX / LaTeX / PDF |
| **数据库** | SQLite (WAL 模式) | 零配置、会话持久化 |
| **测试** | pytest | 1622+ 测试用例 |
| **代码规范** | ruff | 格式化与 lint |

---

## 六、项目结构

```
sophia/
├── agent.py              # Agent 核心，集成所有子系统
├── config.py             # 配置管理（YAML + 环境变量）
├── session.py            # SQLite 会话持久化
├── cli.py                # CLI 入口与命令实现
├── web/                  # Web 服务（FastAPI + 静态资源）
│   ├── __init__.py       # API 端点定义
│   ├── static/           # CSS + JS
│   └── templates/        # HTML 模板
├── prompts/
│   └── system.py         # 系统提示词模板
├── providers/
│   ├── base.py           # Provider 抽象接口
│   ├── openai_compat.py  # OpenAI 兼容实现
│   └── anthropic.py      # Anthropic 实现
├── tools/                # 工具实现（8 个域）
│   ├── registry.py       # 工具注册中心
│   ├── files.py          # 文件操作
│   ├── research.py       # 研究方法工具
│   ├── citation.py       # 引用管理
│   ├── writing.py        # 学术写作
│   ├── analysis.py       # 数据分析
│   ├── web.py            # 网络工具
│   ├── review.py         # 评审系统
│   └── data_collection.py # 数据采集
├── research/             # 研究方法引擎（9 个引擎）
│   ├── statistics.py     # 统计推断
│   ├── design.py         # 研究设计
│   ├── causal.py         # 因果推断
│   ├── survey.py         # 调查研究
│   ├── qualitative.py    # 定性研究
│   ├── meta_analysis.py  # 元分析
│   ├── computational.py  # 计算社会科学
│   ├── ml.py             # 机器学习
│   ├── visualization.py  # 可视化
│   ├── advisor.py        # 方法论推荐
│   ├── discovery/        # 自演化发现系统
│   └── ...
├── review/               # 六维度评审引擎
│   ├── authenticity.py   # 真实性
│   ├── citations.py      # 引用
│   ├── ethics.py         # 伦理
│   ├── language.py       # 语言
│   ├── logic.py          # 逻辑
│   └── statistics.py     # 统计
├── exporters/            # 文档导出
│   ├── docx_engine.py    # DOCX 引擎（含 OMML）
│   ├── latex_export.py   # LaTeX 导出
│   └── pdf_export.py     # PDF 导出
├── hooks.py              # 钩子系统（事件驱动扩展）
├── goal.py               # 目标管理与分解
├── subagent.py           # 子智能体调度
├── loop.py               # 循环任务管理
├── memory.py             # 记忆系统
├── context.py            # 上下文压缩
├── recovery.py           # 故障恢复
├── guardrails.py         # 工具调用防护
├── scheduler.py          # 定时任务
├── kanban.py             # 看板任务管理
├── plugins.py            # 插件系统
├── security.py           # 安全管理
├── skills/               # 技能挖掘与进化
├── learning.py           # 学习系统
├── autopilot.py          # 自动驾驶编排
├── experiment.py         # 实验管理
├── snapshot.py           # 快照管理
├── trajectory.py         # 轨迹记录
├── browser.py            # 浏览器工具
├── credentials.py        # 凭据管理
└── pipeline/             # 流水线编排
    ├── assembler.py      # 文档组装
    └── loop.py           # 流水线循环
```

---

## 七、部署方式

### 7.1 本地安装

```bash
pip install -e ".[all]"
sophia chat        # CLI 模式
sophia web         # Web 模式（默认端口 8892）
```

### 7.2 Docker

```bash
docker compose up -d
```

### 7.3 配置方式

支持三层配置优先级（高到低）：

1. 环境变量（`SOPHIA_API_KEY`, `SOPHIA_BASE_URL`, `SOPHIA_MODEL`）
2. `config.yaml` 文件
3. 内置默认值

---

## 八、总结

SophiaAgent 不是一个简单的"聊天机器人套壳"项目。它的核心差异化在于：

1. **领域深度** — 内置 9 大研究方法引擎 + 84+ 个专业工具，覆盖人文社科从研究设计到论文发表的全链路
2. **工程完备** — 20+ 个子系统（Hook、Goal、Memory、Recovery、Guardrails 等）构成生产级 Agent 架构
3. **自演化能力** — 技能挖掘、方法论发现、模板进化，系统会随着使用越来越"懂"用户
4. **学术规范** — GB/T 7714 / APA 引用、APA 三线表、OMML 公式、六维度评审，从底层设计上保证学术合规
5. **多模态交互** — CLI、Web、API 三种接口，适配不同使用场景

项目在 1622+ 个测试用例的保障下保持零回归迭代，代码结构清晰、模块职责单一，具备持续演进的技术基础。
