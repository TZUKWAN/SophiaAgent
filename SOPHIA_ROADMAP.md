# SophiaAgent 功能扩充路线图 — 详细任务分解

> 版本：v1.0 | 日期：2026-05-21
> 本文档将 12 个功能模块拆解为可执行的具体任务，每个任务包含：目标、具体工作内容、涉及文件、验收标准、依赖关系。

---

## 目录

- [Phase 1：核心能力补全（P0）](#phase-1核心能力补全p0)
  - [模块 A：中文 NLP 与质性分析引擎](#模块-a中文-nlp-与质性分析引擎)
  - [模块 B：论文精读与笔记系统](#模块-b论文精读与笔记系统)
  - [模块 C：英文学术写作润色与投稿支持](#模块-c英文学术写作润色与投稿支持)
- [Phase 2：学科深度适配（P1）](#phase-2学科深度适配p1)
  - [模块 D：学科专属模板与规范库](#模块-d学科专属模板与规范库)
  - [模块 E：理论脉络与概念史工具](#模块-e理论脉络与概念史工具)
  - [模块 F：论证逻辑检查器增强](#模块-f论证逻辑检查器增强)
- [Phase 3：体验增强与差异化（P2）](#phase-3体验增强与差异化p2)
  - [模块 G：访谈与问卷数据采集管线](#模块-g访谈与问卷数据采集管线)
  - [模块 H：研究设计与方法论顾问增强](#模块-h研究设计与方法论顾问增强)
  - [模块 I：研究伦理与 IRB 支持](#模块-i研究伦理与-irb-支持)
  - [模块 J：期刊匹配与投稿指南](#模块-j期刊匹配与投稿指南)
  - [模块 K：学术汇报 PPT 生成](#模块-k学术汇报-ppt-生成)
  - [模块 L：多语言学术翻译](#模块-l多语言学术翻译)

---

## Phase 1：核心能力补全（P0）

---

### 模块 A：中文 NLP 与质性分析引擎

**目标**：补全中文文本分析能力，使系统的质性研究工具能真正处理中文访谈、政策文本、社交媒体数据。

**现状**：`sophia/research/qualitative.py` 中的 `QualitativeEngine` 仅包含英文停用词集 `_EN_STOPWORDS` 和简单的英文正/负面情感词库。`content()` 方法基于英文空格分词。`sentiment()` 方法的回退词库全部是英文。中文文本进入后会因为无法分词而产出无意义结果。

#### A-1：中文分词基础设施

| 项 | 内容 |
|---|------|
| **目标** | 为所有中文质性分析方法提供统一分词能力 |
| **具体工作** | (1) 新建 `sophia/research/chinese_nlp.py`，实现 `ChineseTokenizer` 类。(2) 优先尝试 jieba，回退到 pkuseg，最终回退到基于字符的简单分词。(3) 内置中文学术领域自定义词典（社会学、教育学、政治学、心理学常用术语约 2000 词）。(4) 支持用户自定义词典文件加载（从 workspace `/.sophia/user_dict.txt` 读取）。(5) 提供统一接口 `tokenize(text, mode) -> List[str]`，mode 支持 `'default'` / `'search'` / `'all'`。(6) 提供停用词过滤接口 `remove_stopwords(tokens) -> List[str]`，内置中文停用词表（哈工大/百度停用词表合并去重）。|
| **涉及文件** | 新建 `sophia/research/chinese_nlp.py` |
| **依赖** | jieba（可选），pkuseg（可选），无硬依赖 |
| **验收标准** | ① 输入「社会资本对城市居民幸福感的影响机制研究」能正确分词输出 `['社会', '资本', '对', '城市', '居民', '幸福感', '的', '影响', '机制', '研究']`。② 无 jieba/pkuseg 时退化为字符级分词不报错。③ 自定义词典加载后术语「内卷化」不被切开。④ 100 次调用的延迟 < 2 秒（对 5000 字文本）。|

#### A-2：中文关键词提取与主题建模

| 项 | 内容 |
|---|------|
| **目标** | 支持从中文文本中自动提取关键词和主题 |
| **具体工作** | (1) 在 `chinese_nlp.py` 中实现 `extract_keywords(text, top_n=20) -> List[Tuple[str, float]]`，基于 TF-IDF + TextRank 混合算法。(2) 实现 `extract_topics(texts, n_topics=5) -> List[Dict]`，支持 LDA 和 BERTopic 两种模式（BERTopic 为可选依赖）。(3) 每个主题返回 `{topic_id, keywords: List[str], weight: float, representative_docs: List[int]}`。(4) 结果支持 ResultStore 持久化。|
| **涉及文件** | `sophia/research/chinese_nlp.py`（追加方法） |
| **依赖** | 任务 A-1，gensim（可选），bertopic（可选） |
| **验收标准** | ① 输入 10 段教育学研究访谈文本，能输出 3-5 个主题，每个主题有 5+ 关键词。② 关键词排名合理，高频无意义词被过滤。③ 无 gensim 时退化为 TF-IDF 聚类不报错。|

#### A-3：中文情感分析

| 项 | 内容 |
|---|------|
| **目标** | 对中文文本进行情感极性和情绪维度分析 |
| **具体工作** | (1) 在 `chinese_nlp.py` 中实现 `analyze_sentiment_cn(text) -> Dict`。(2) 优先使用 snownlp，回退到自带的中文情感词典（正面词 800+、负面词 800+、程度副词 100+、否定词 30+）。(3) 输出 `{sentiment: 'positive'/'negative'/'neutral', score: float, confidence: float, dimensions: {joy, anger, sadness, fear, surprise}, key_phrases: List[str]}`。(4) 支持批量处理接口 `analyze_sentiment_batch(texts) -> List[Dict]`。(5) 内置社会舆情常见情感模式模板（如舆情发酵-高潮-衰减曲线）。|
| **涉及文件** | `sophia/research/chinese_nlp.py`（追加方法），新建 `sophia/research/data/sentiment_cn_dict.json` |
| **依赖** | 任务 A-1，snownlp（可选） |
| **验收标准** | ① 「这项政策极大改善了农民工的就业环境」判定为 positive，score > 0.6。② 「形式主义严重，基层苦不堪言」判定为 negative。③ 批量处理 100 条微博文本延迟 < 5 秒。|

#### A-4：话语分析引擎

| 项 | 内容 |
|---|------|
| **目标** | 识别文本中的权力关系、话语策略、意识形态框架 |
| **具体工作** | (1) 新建 `sophia/research/discourse.py`，实现 `DiscourseEngine` 类。(2) 实现 `analyze_discourse(text, framework='foucault') -> Dict`，支持三种分析框架：福柯式（权力/知识/话语）、批判话语分析（CDA，Fairclough 三维模型）、叙事话语分析。(3) 具体分析维度：① 话语主体识别（谁在说话、谁被描述）② 权力关系标注（支配/服从/抵抗/协商）③ 话语策略检测（模糊化、权威引用、数据修辞、情感动员、他者化）④ 意识形态框架识别（新自由主义/国家主义/民粹主义等 10 种预设框架）。(4) 支持 LLM 辅助路径和纯规则回退路径。(5) 输出包含原文标注（行号 + 标签）和总结报告。(6) 注册为工具 `research_discourse_analysis`。|
| **涉及文件** | 新建 `sophia/research/discourse.py`，修改 `sophia/research/register.py`（新增 `_register_discourse_tools`），修改 `sophia/agent.py`（初始化 DiscourseEngine） |
| **依赖** | 任务 A-1 |
| **验收标准** | ① 输入一段政府工作报告节选，能识别出「权威引用」和「数据修辞」两种话语策略。② 输入一段新闻评论，能标注出至少一个权力关系（如「政府-公民：支配」）。③ 无 LLM 时回退路径产出基本可用的结果。④ 工具注册后能通过 `sophia tools list` 看到。|

#### A-5：叙事分析引擎

| 项 | 内容 |
|---|------|
| **目标** | 从访谈/文本中识别叙事结构、转折点、角色建构 |
| **具体工作** | (1) 新建 `sophia/research/narrative.py`，实现 `NarrativeEngine` 类。(2) 实现 `analyze_narrative(text, mode='structure') -> Dict`，支持三种模式：① `structure`：识别叙事结构（Labov 六要素模型：摘要、导向、纠葛、评价、解决、结尾）② `turning_point`：识别关键转折点（时间线 + 事件分类 + 情感转向）③ `identity`：角色/身份建构分析（自我定位、他者建构、角色转换）。(3) 支持 LLM 辅助和纯规则回退。(4) 输出格式包含：`{narrative_elements: List[Dict], timeline: List[Dict], characters: List[Dict], turning_points: List[Dict], coherence_score: float}`。(5) 注册为工具 `research_narrative_analysis`。|
| **涉及文件** | 新建 `sophia/research/narrative.py`，修改 `sophia/research/register.py`，修改 `sophia/agent.py` |
| **依赖** | 任务 A-1 |
| **验收标准** | ① 输入一段半结构化访谈转录文本，能识别出至少 2 个叙事要素。② 转折点标注包含时间定位和事件描述。③ 角色建构分析能区分「自我」和「他者」的描述。④ 中文访谈文本分词正确。|

#### A-6：NVivo 风格编码系统

| 项 | 内容 |
|---|------|
| **目标** | 支持多级编码树、备忘录、编码一致性计算，适合团队质性研究 |
| **具体工作** | (1) 在 `sophia/research/qualitative.py` 中新增 `CodingTree` 类和 `CodingProject` 类。(2) `CodingTree` 支持：① 创建/删除/重命名编码节点 ② 多级层级（父子关系）③ 编码节点属性（颜色、描述、创建时间、修改时间）④ 导出为树形 JSON。(3) `CodingProject` 支持：① 关联编码树 + 文本数据 ② 为文本片段分配编码（标记起止位置 + 编码 ID + 编码者 ID）③ 备忘录 (memo) 管理（每个编码节点可附加备忘录）④ 编码一致性计算（Cohen's Kappa，两位编码者之间）⑤ 编码频率统计和交叉表 ⑥ 编码饱和度检测（新增编码数量随编码进程的边际递减曲线）。(4) 数据持久化到 workspace `/.sophia/coding_projects/` 目录，每个项目一个 JSON 文件。(5) 注册 5 个工具：`coding_project_create`、`coding_tree_edit`、`coding_assign`、`coding_memo`、`coding_reliability_report`。(6) 调整现有 `coding_reliability()` 方法，使其能读取 CodingProject 数据。|
| **涉及文件** | 修改 `sophia/research/qualitative.py`（追加类），修改 `sophia/research/register.py`（新增 `_register_coding_tools`），修改 `sophia/agent.py` |
| **依赖** | 任务 A-1 |
| **验收标准** | ① 能创建三级编码树（如「就业 > 灵活就业 > 平台经济」）。② 两位编码者对同一段文本编码后，Kappa > 0 时能正确计算。③ 备忘录能关联到具体编码节点。④ 编码饱和度曲线在编码量足够时呈现递减趋势。⑤ 数据持久化后重启系统不丢失。⑥ 5 个工具全部注册成功。|

#### A-7：改造现有 QualitativeEngine 的中文兼容性

| 项 | 内容 |
|---|------|
| **目标** | 让 `QualitativeEngine` 的现有 5 个方法（thematic、content、grounded_code、sentiment、coding_reliability）都能正确处理中文文本 |
| **具体工作** | (1) 修改 `content()` 方法：英文走空格分词，中文走 `ChineseTokenizer`。(2) 修改 `thematic()` 方法：LLM prompt 中增加中文示例和中文编码模板。(3) 修改 `grounded_code()` 方法：LLM prompt 增加中文编码范式（ Strauss & Corbin 中文版术语）。(4) 修改 `sentiment()` 方法：检测到中文文本时调用 `analyze_sentiment_cn()` 替代 VADER。(5) 在 `_EN_STOPWORDS` 旁边新增 `_CN_STOPWORDS` 常量。(6) 所有方法增加 `language` 参数（`'auto'`/`'en'`/`'zh'`），auto 模式通过字符比例自动判断。|
| **涉及文件** | 修改 `sophia/research/qualitative.py` |
| **依赖** | 任务 A-1，A-3 |
| **验收标准** | ① `content()` 对中文访谈文本能输出合理的词频表和共现矩阵。② `thematic()` 对中文文本能生成中文主题标签。③ `sentiment()` 对中文文本能给出正确的情感极性。④ `language='auto'` 对中英混合文本能正确识别主语言。⑤ 现有英文分析能力不受影响（回归测试全部通过）。|

#### A-8：中文 NLP 工具注册与集成测试

| 项 | 内容 |
|---|------|
| **目标** | 将所有新增中文 NLP 能力注册为可调用工具，并通过集成测试 |
| **具体工作** | (1) 新建 `sophia/tools/chinese_nlp.py`，实现 `register_chinese_nlp_tools(registry, engine)`。(2) 注册工具：`chinese_tokenize`、`chinese_keywords`、`chinese_sentiment`、`chinese_topics`。(3) 在 `sophia/agent.py` 的 `_register_tools()` 中调用注册函数。(4) 在 `sophia/swarm/roles.py` 中为 `data_analyst` 角色添加中文 NLP 工具权限。(5) 新建 `tests/test_chinese_nlp.py`，覆盖所有新增方法的单元测试。(6) 新建 `tests/test_discourse.py` 和 `tests/test_narrative.py`。(7) 新建 `tests/test_coding_project.py`，覆盖 CodingProject 的完整 CRUD 和一致性计算。(8) 所有测试要求：中文输入不报错、输出格式正确、无外部依赖时回退路径正常工作。|
| **涉及文件** | 新建 `sophia/tools/chinese_nlp.py`，修改 `sophia/agent.py`，修改 `sophia/swarm/roles.py`，新建 4 个测试文件 |
| **依赖** | 任务 A-1 至 A-7 |
| **验收标准** | ① `sophia tools list` 能看到所有新增工具。② `pytest tests/test_chinese_nlp.py tests/test_discourse.py tests/test_narrative.py tests/test_coding_project.py` 全部通过。③ swarm 的 data_analyst 角色能调用中文 NLP 工具。④ 无 jieba/snownlp 环境下所有测试仍然通过（回退路径）。|

---

### 模块 B：论文精读与笔记系统

**目标**：构建从「搜到论文」到「读懂论文」到「管理笔记」到「驱动写作」的完整闭环。

**现状**：`sophia/tools/research.py` 提供了 Semantic Scholar / arXiv / Crossref 的文献检索，但检索结果只有元数据。没有任何工具帮助用户深度阅读、提取信息、做笔记、或管理跨文献知识。

#### B-1：论文关键论点自动提取

| 项 | 内容 |
|---|------|
| **目标** | 从 PDF 或纯文本中自动提取论文的核心结构化信息 |
| **具体工作** | (1) 新建 `sophia/research/reader.py`，实现 `PaperReader` 类。(2) 实现 `extract_key_elements(text) -> Dict`，提取：① 研究问题 (research_questions) ② 核心论点 (core_arguments) ③ 研究方法 (methods) ④ 数据来源 (data_sources) ⑤ 主要发现 (main_findings) ⑥ 研究局限 (limitations) ⑦ 理论框架 (theoretical_framework)。(3) 使用 LLM 进行结构化提取，prompt 要求输出严格 JSON。(4) 回退路径：基于关键词匹配 + 正则的规则提取（识别「Research Question」「方法」「数据」等标志性段落）。(5) 支持 PDF 直接输入（通过 pymupdf 提取文本后调用）。(6) 对提取结果进行质量评分（完整性、一致性），标记低置信度字段。|
| **涉及文件** | 新建 `sophia/research/reader.py` |
| **依赖** | 无（pymupdf 已有） |
| **验收标准** | ① 输入一篇英文实证论文的 PDF，能正确提取出研究问题和主要发现。② 低质量文本（扫描版 PDF）能给出质量评分 < 50%。③ LLM 不可用时回退路径能提取至少 3 个要素。④ 输出为合法 JSON，结构符合规范。|

#### B-2：PDF 批注解析

| 项 | 内容 |
|---|------|
| **目标** | 读取用户已有的 PDF 高亮/批注，提取关键段落并归类 |
| **具体工作** | (1) 在 `PaperReader` 中实现 `extract_annotations(pdf_path) -> Dict`。(2) 使用 pymupdf 提取三类批注：① 高亮文本 (highlight) ② 下划线文本 (underline) ③ 文字批注 (text annotation / sticky note)。(3) 每条批注返回 `{page, rect, content, annotation_type, surrounding_context}`。(4) `surrounding_context` 提取批注前后各 100 字的上下文。(5) 自动按颜色分类（如果用户用不同颜色标记不同类别）。(6) 支持将批注导出为 Markdown 格式，按页面排序。|
| **涉及文件** | `sophia/research/reader.py`（追加方法） |
| **依赖** | pymupdf（已有） |
| **验收标准** | ① 能正确读取包含高亮和文字批注的 PDF。② 每条批注都有完整的页码和上下文。③ 无批注的 PDF 不报错，返回空列表。④ 导出为 Markdown 后格式正确可读。|

#### B-3：Zettelkasten 卡片笔记系统

| 项 | 内容 |
|---|------|
| **目标** | 实现学术级卡片笔记系统，支持原子化笔记和双向链接 |
| **具体工作** | (1) 新建 `sophia/research/notes.py`，实现 `ZettelkastenStore` 类。(2) 每张笔记卡片包含：`{id, title, content, tags: List[str], links: List[str], source_type, source_id, created_at, updated_at}`。(3) 笔记类型三种：① `concept`：概念卡片（定义 + 例子 + 关联概念）② `evidence`：证据卡片（来源 + 数据 + 方法 + 结论）③ `comment`：评论卡片（个人见解 + 论证 + 质疑）。(4) 双向链接：通过 `[[note_id]]` 语法在笔记内容中引用其他笔记，系统自动维护反向引用表。(5) 搜索接口：`search(query, tags, linked_to) -> List[Note]`，支持全文搜索 + 标签过滤 + 链接关系查询。(6) 图谱接口：`get_link_graph() -> Dict`，输出 `{nodes: List, edges: List}` 用于可视化。(7) 持久化到 workspace `/.sophia/notes/`，每张卡片一个 JSON 文件。(8) 从 `PaperReader.extract_key_elements()` 的结果一键生成 evidence 笔记。(9) 注册工具：`note_create`、`note_search`、`note_link`、`note_graph`、`note_from_paper`。|
| **涉及文件** | 新建 `sophia/research/notes.py`，新建 `sophia/tools/notes.py`，修改 `sophia/agent.py` |
| **依赖** | 任务 B-1 |
| **验收标准** | ① 创建 10 张不同类型的笔记卡片不报错。② `[[note-002]]` 链接能被正确解析，反向引用表自动更新。③ 全文搜索能找到包含特定关键词的笔记。④ `get_link_graph()` 输出的 nodes/edges 能被 networkx 正确加载。⑤ 数据持久化后重启不丢失。⑥ 5 个工具全部注册成功。|

#### B-4：文献关系图谱

| 项 | 内容 |
|---|------|
| **目标** | 自动生成文献间的引用/反对/继承/补充等关系的可视化图谱 |
| **具体工作** | (1) 新建 `sophia/research/literature_graph.py`，实现 `LiteratureGraph` 类。(2) 从 BibTeX 库和 citation_relations.json 中读取已有文献和关系数据。(3) 关系类型 6 种：`cites`（引用）、`contradicts`（反对/质疑）、`extends`（继承/扩展）、`supplements`（补充）、`method_similar`（方法相似）、`theory_similar`（理论同源）。(4) 实现 `build_graph(literature_ids) -> nx.DiGraph`。(5) 实现 `visualize(graph, layout='spring', output_path) -> str`，输出 PNG/SVG/HTML（pyvis 交互式）。(6) 实现 `detect_clusters(graph) -> List[List[str]]`，自动检测文献聚类（Louvain 社区检测或标签传播）。(7) 实现 `find_key_papers(graph) -> List[Dict]`，通过 PageRank/度中心性识别关键文献。(8) 注册工具：`literature_graph_build`、`literature_graph_visualize`、`literature_graph_clusters`。|
| **涉及文件** | 新建 `sophia/research/literature_graph.py`，新建 `sophia/tools/literature_graph.py`，修改 `sophia/agent.py` |
| **依赖** | networkx（pyproject.toml 的 advanced 可选依赖中已有），pyvis（新增可选依赖） |
| **验收标准** | ① 输入 20 篇文献的引用关系，能生成有向图并输出 PNG。② 社区检测能将方法相似的文献归为一类。③ PageRank 识别出的关键文献与人工判断一致（至少前 3 篇）。④ pyvis 交互式 HTML 能在浏览器中正常打开并操作。⑤ 无 pyvis 时退化为 matplotlib 静态图。|

#### B-5：跨文献对比矩阵

| 项 | 内容 |
|---|------|
| **目标** | 自动生成多篇文献在关键维度上的结构化对比表 |
| **具体工作** | (1) 在 `PaperReader` 中实现 `compare_papers(elements_list) -> Dict`。(2) 对比维度：研究问题、理论框架、研究方法、数据来源、样本量、主要发现、研究局限。(3) 输出为 Markdown 表格 + JSON 结构。(4) 自动检测共识点（多篇文献结论一致）和争议点（结论矛盾）。(5) 支持 LLM 辅助的语义级对比和纯规则的字面对比两种模式。|
| **涉及文件** | `sophia/research/reader.py`（追加方法） |
| **依赖** | 任务 B-1 |
| **验收标准** | ① 输入 3 篇论文的提取结果，能生成完整的对比表格。② 自动检测出至少 1 个共识点和 1 个争议点（如果存在）。③ Markdown 表格在终端和 Web UI 中都能正确渲染。|

#### B-6：精读系统工具注册与集成测试

| 项 | 内容 |
|---|------|
| **目标** | 注册所有精读工具，补充 Swarm 角色，编写测试 |
| **具体工作** | (1) 在 `sophia/agent.py` 的 `_register_tools()` 中调用所有新增注册函数。(2) 在 `sophia/swarm/roles.py` 中：① `literature_searcher` 角色增加 `literature_graph_build` 权限 ② 新增 `reader` 角色（论文精读专家），权限包含所有 reader/notes 工具。(3) 新建 `tests/test_reader.py`（PaperReader 单元测试）。(4) 新建 `tests/test_notes.py`（ZettelkastenStore 单元测试）。(5) 新建 `tests/test_literature_graph.py`（LiteratureGraph 单元测试）。(6) 新建 `tests/test_reading_workflow.py`（端到端：提取→笔记→图谱→对比 全流程测试）。|
| **涉及文件** | 修改 `sophia/agent.py`，修改 `sophia/swarm/roles.py`，新建 4 个测试文件 |
| **依赖** | 任务 B-1 至 B-5 |
| **验收标准** | ① 所有新工具在 `sophia tools list` 中可见。② 端到端测试覆盖「读取 PDF→提取要素→生成笔记→构建图谱→对比分析」全流程。③ `pytest` 新增测试全部通过。④ swarm 中 literature_searcher 和 reader 角色权限正确。|

---

### 模块 C：英文学术写作润色与投稿支持

**目标**：帮助中国文科生突破英文写作瓶颈，覆盖从草稿到投稿的全流程。

**现状**：系统完全没有英文润色能力。`sophia/research/apa.py` 只负责统计结果的 APA 格式化文字输出。`sophia/tools/citation.py` 支持 GB/T 7714 / APA / MLA / Chicago 四种引注格式，但没有格式转换功能。写作工具 (`sophia/tools/writing.py`) 只管文档结构管理，不管语言质量。

#### C-1：学术英语风格改写引擎

| 项 | 内容 |
|---|------|
| **目标** | 将中式英语/非正式英语改写为地道的学术英语 |
| **具体工作** | (1) 新建 `sophia/research/writing_en.py`，实现 `AcademicEnglishEngine` 类。(2) 实现 `polish(text, style='social_science') -> Dict`，style 支持：`social_science` / `humanities` / `education` / `public_policy`。(3) 改写维度：① 词汇升级（口语词→学术词，如「get」→「obtain」）② 句式调整（简单句→复杂句式、被动语态优化）③ Chinglish 检测与修正（内置 200 条常见中式英语模式规则）④ 冗余消除（删除无意义的填充词）⑤ 学术连接词优化（however/in contrast/moreover 等）。(4) 每处修改提供：`{original, revised, reason, category}`。(5) 支持 LLM 辅助改写和纯规则改写两种路径。(6) 输出修订对比 (diff) 和清洁版本。|
| **涉及文件** | 新建 `sophia/research/writing_en.py`，新建 `sophia/research/data/chinglish_patterns.json` |
| **依赖** | 无 |
| **验收标准** | ① 输入「The result of this study shows that the effect of education on income is very big」，输出中「very big」被改为「substantial」或「significant」。② 至少检测出 3 种 Chinglish 模式。③ 每处修改都有 reason 说明。④ 无 LLM 时纯规则路径能修正至少 50% 的常见 Chinglish。|

#### C-2：句式多样性与可读性分析

| 项 | 内容 |
|---|------|
| **目标** | 检测并修复重复句式、被动语态滥用、过长/过短句等问题 |
| **具体工作** | (1) 在 `AcademicEnglishEngine` 中实现 `analyze_readability(text) -> Dict`。(2) 分析指标：① 平均句长、句长标准差 ② 句式结构多样性（简单句/并列句/复合句/并列复合句比例）③ 被动语态比例 ④ 段落平均长度 ⑤ Flesch-Kincaid 可读性分数 ⑥ 词汇多样性 (TTR) ⑦ 学术词汇比例 (AWL coverage)。(3) 对比基准：内置社科/人文学科论文的平均值参考范围。(4) 输出改进建议列表：`{metric, current, target, suggestions: List[str]}`。(5) 实现 `diversify_sentences(text) -> Dict`，通过 LLM 或规则改写重复句式。|
| **涉及文件** | `sophia/research/writing_en.py`（追加方法） |
| **依赖** | 无 |
| **验收标准** | ① 能正确计算 Flesch-Kincaid 分数。② 被动语态比例超过 40% 时给出警告。③ 句式多样性指标低于参考范围时给出具体改进建议。④ 建议是可操作的（不是泛泛的「请改善句式」）。|

#### C-3：术语一致性管理

| 项 | 内容 |
|---|------|
| **目标** | 建立论文级术语表，全文检查术语使用是否一致 |
| **具体工作** | (1) 在 `AcademicEnglishEngine` 中实现 `build_glossary(text) -> List[Dict]`。(2) 自动提取文本中的关键术语（出现 3 次以上的名词短语）。(3) 对每个术语检查：① 全文拼写是否一致（如「e-commerce」vs「eCommerce」vs「electronic commerce」）② 中英文对照是否一致（如「social capital」/「社会资本」不应有时翻译为「社会资本」有时翻译为「社会资产」）③ 缩写首次出现是否定义（如首次使用 SEM 时应写「Structural Equation Modeling (SEM)」）。(4) 实现 `check_consistency(text, glossary) -> List[Dict]`，输出不一致列表。(5) 术语表可持久化到 workspace `/.sophia/glossary.json`。|
| **涉及文件** | `sophia/research/writing_en.py`（追加方法） |
| **依赖** | 无 |
| **验收标准** | ① 一篇 5000 词论文中，能检测出至少 2 处术语不一致。② 缩写未定义的情况能被检测到。③ 术语表 JSON 格式正确，可跨会话使用。|

#### C-4：投稿信与审稿回复模板生成

| 项 | 内容 |
|---|------|
| **目标** | 自动生成符合学术规范的投稿信和审稿人回复 |
| **具体工作** | (1) 在 `AcademicEnglishEngine` 中实现 `generate_cover_letter(paper_meta, journal) -> str`。(2) `paper_meta` 包含：title, authors, abstract, keywords, highlights。(3) `journal` 包含：name, scope, editor_name（可选）。(4) Cover letter 结构：① 称呼 ② 论文简介（2-3 句）③ 与期刊 scope 的契合度说明 ④ 核心贡献陈述 ⑤ 声明（原创性、无利益冲突）⑥ 结尾。(5) 实现 `generate_review_response(review_comments, author_revisions) -> str`。(6) 回复格式：逐条回复，每条包含：① 原审稿意见（引用）② 作者回复（感谢 + 回应 + 具体修改位置）③ 修改后的文本摘录。(7) 支持 LLM 生成 + 模板填充两种方式。|
| **涉及文件** | `sophia/research/writing_en.py`（追加方法），新建 `sophia/research/templates/cover_letter.txt`，新建 `sophia/research/templates/review_response.txt` |
| **依赖** | 无 |
| **验收标准** | ① 生成的 Cover letter 包含所有必要段落。② 格式符合 SSCI 期刊常见要求。③ 审稿回复模板能正确处理「大修」「小修」「拒稿重投」三种情况的措辞。④ LLM 不可用时模板填充仍能产出可用结果。|

#### C-5：引注格式智能转换

| 项 | 内容 |
|---|------|
| **目标** | 在 APA / Chicago / MLA / GB-T-7714 之间一键切换引注格式 |
| **具体工作** | (1) 在 `sophia/tools/citation.py` 中实现 `convert_citation_style(workspace, from_style, to_style) -> Dict`。(2) 解析现有 `.sophia/references.bib` 中的所有条目。(3) 使用内置格式化规则重新生成每种风格的引注文本。(4) 同时更新文档正文中内联引用的格式（如 `(张三, 2020)` → `(Zhang, 2020)`）。(5) 支持批量转换和单条转换。(6) 格式化规则覆盖：期刊论文、专著、论文集章节、学位论文、网络资源、政策文件等 8 种文献类型。|
| **涉及文件** | 修改 `sophia/tools/citation.py` |
| **依赖** | bibtexparser（已有） |
| **验收标准** | ① 同一条文献在 APA 和 GB-T-7714 格式下的输出差异正确。② 文档正文中的内联引用同步更新。③ 8 种文献类型至少 6 种有正确的格式化模板。④ 不支持的文献类型给出明确提示而非报错。|

#### C-6：英文写作工具注册与集成测试

| 项 | 内容 |
|---|------|
| **目标** | 注册所有英文写作工具，补充 Swarm 角色，编写测试 |
| **具体工作** | (1) 新建 `sophia/tools/writing_en.py`，实现 `register_writing_en_tools(registry, engine)`。(2) 注册工具：`en_polish`、`en_readability`、`en_glossary_build`、`en_consistency_check`、`en_cover_letter`、`en_review_response`、`citation_style_convert`。(3) 在 `sophia/agent.py` 的 `_register_tools()` 中调用。(4) 在 `sophia/swarm/roles.py` 中为 `writer` 角色增加英文写作工具权限。(5) 新建 `tests/test_writing_en.py`，覆盖所有方法。(6) 新建 `tests/test_citation_style_convert.py`。|
| **涉及文件** | 新建 `sophia/tools/writing_en.py`，修改 `sophia/agent.py`，修改 `sophia/swarm/roles.py`，新建 2 个测试文件 |
| **依赖** | 任务 C-1 至 C-5 |
| **验收标准** | ① 7 个工具全部注册成功并在 `sophia tools list` 中可见。② writer 角色能调用所有英文写作工具。③ 所有测试通过。④ 与现有 citation 工具无冲突。|

---

## Phase 2：学科深度适配（P1）

---

### 模块 D：学科专属模板与规范库

**目标**：为人文社科 6 大学科提供专属的写作模板、论证结构、脚注/引注规范和研究流程指引。

**现状**：`sophia/tools/writing.py` 中 `DEFAULT_OUTLINES` 只有 6 种通用大纲（paper / report / monograph / grant-nsfc / grant-nssfc / grant-moe）。`sophia/exporters/` 中有通用 LaTeX/DOCX/PDF 导出。`sophia/prompts/system.py` 的系统提示只有泛化的学术写作规则。没有学科级的差异处理。

#### D-1：历史学模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为历史学研究提供完整的写作和考证模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/history/` 目录。(2) 创建大纲模板 `outline_history_paper.json`：史料考证型论文结构（问题的提出→史料批判→考证分析→结论）和史学评论型论文结构。(3) 创建脚注模板 `footnote_rules.json`：历史学脚注体系（非作者-年份制），包含档案引用、古籍引用、方志引用、口述史料引用、未出版手稿引用等 12 种脚注格式的正则模板。(4) 创建史料批判清单 `source_criticism_checklist.json`：史料来源鉴定、作者背景审查、写作目的分析、时代语境还原、传抄/刊刻版本对比 5 个维度的检查项。(5) 创建 section_prompts.json：每个章节的写作指导 prompt（如「问题的提出」应包含学术脉络定位和核心史料说明）。(6) 修改 `sophia/tools/writing.py`，在 `DEFAULT_OUTLINES` 中增加 `paper-history` 类型。(7) 在 `sophia/exporters/latex_export.py` 中增加历史学脚注 LaTeX 模板。|
| **涉及文件** | 新建 `sophia/prompts/templates/history/` 下 4 个文件，修改 `sophia/tools/writing.py`，修改 `sophia/exporters/latex_export.py` |
| **依赖** | 无 |
| **验收标准** | ① 创建 `paper-history` 类型文档时自动使用历史学大纲结构。② 脚注模板覆盖古籍、档案、方志等至少 8 种史料类型。③ 史料批判清单能作为 review 工具的检查项使用。④ LaTeX 导出的脚注格式正确。|

#### D-2：文学/文化研究模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为文学和文化研究提供文本细读和理论应用模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/literature/` 目录。(2) 创建大纲模板 `outline_literary_analysis.json`：文本细读型（选段→文本分析→理论阐释→文化语境→结论）和理论应用型（理论框架→文本选择→分析→批评→结论）。(3) 创建理论框架库 `theory_frameworks.json`：精神分析、女性主义、后殖民主义、新历史主义、结构主义/后结构主义、接受美学、文化唯物主义、生态批评等 10 种理论框架的核心概念和应用步骤。(4) 创建文本细读指引 `close_reading_guide.json`：修辞手法识别清单、叙事视角分析步骤、象征/隐喻解读框架。(5) 修改 `sophia/tools/writing.py`，增加 `paper-literary` 类型。|
| **涉及文件** | 新建 `sophia/prompts/templates/literature/` 下 3 个文件，修改 `sophia/tools/writing.py` |
| **依赖** | 无 |
| **验收标准** | ① 理论框架库至少覆盖 10 种文学批评流派。② 每种理论框架包含：核心概念 3-5 个、应用步骤、示例论文引用、适用文本类型。③ 创建 `paper-literary` 时自动推荐理论框架。|

#### D-3：教育学研究模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为教育学研究提供行动研究和课例分析模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/education/` 目录。(2) 创建行动研究模板 `action_research.json`：计划→行动→观察→反思的循环结构，每个阶段的具体写作指引。(3) 创建课例分析模板 `lesson_study.json`：教学设计→课堂实施→教学反思→改进方案的格式。(4) 创建量表开发模板 `scale_development.json`：文献梳理→维度确定→题项生成→专家评审→预测试→信效度检验的完整流程。(5) 创建教育实验报告模板 `edu_experiment.json`：符合 APA 格式的教育实验报告结构。(6) 修改 `sophia/tools/writing.py`，增加 `paper-education`、`paper-action-research`、`paper-lesson-study` 类型。|
| **涉及文件** | 新建 `sophia/prompts/templates/education/` 下 4 个文件，修改 `sophia/tools/writing.py` |
| **依赖** | 无 |
| **验收标准** | ① 行动研究模板包含至少 2 轮循环的完整结构。② 课例分析模板与实际教学场景匹配。③ 量表开发模板覆盖从构念定义到信效度报告的全流程。|

#### D-4：社会学/人类学模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为社会学和人类学研究提供田野报告和民族志模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/sociology/` 目录。(2) 创建田野报告模板 `fieldwork_report.json`：研究背景→田野概况→进入田野→参与观察记录→访谈分析→理论对话→反思与结论。(3) 创建民族志模板 `ethnography.json`：场景描写→人物素描→事件叙述→文化解读→理论分析。(4) 创建伦理审查声明模板 `ethics_statement.json`：知情同意、匿名化处理、田野关系管理、敏感信息保护。(5) 创建参与观察记录模板 `observation_log.json`：时间/地点/人物/事件/研究者反思的结构化记录格式。(6) 修改 `sophia/tools/writing.py`，增加 `paper-fieldwork`、`paper-ethnography` 类型。|
| **涉及文件** | 新建 `sophia/prompts/templates/sociology/` 下 4 个文件，修改 `sophia/tools/writing.py` |
| **依赖** | 无 |
| **验收标准** | ① 田野报告模板的章节结构与人类学田野报告惯例一致。② 伦理声明覆盖国内常见伦理审查要求。③ 观察记录模板支持结构化填写。|

#### D-5：政治学/法学模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为政治学和法学研究提供案例分析和比较研究模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/politics_law/` 目录。(2) 创建案例分析模板 `case_analysis.json`：案例选择依据→案例描述→法律/政策分析→比较→结论。(3) 创建比较政治研究模板 `comparative_politics.json`：研究设计（Most Similar / Most Different Systems Design）→变量操作化→案例选择→数据收集→跨案例比较→结论。(4) 创建法律论证结构模板 `legal_argument.json`：IRAC 模式（Issue-Rule-Application-Conclusion）。(5) 修改 `sophia/tools/writing.py`，增加 `paper-case-analysis`、`paper-comparative` 类型。|
| **涉及文件** | 新建 `sophia/prompts/templates/politics_law/` 下 3 个文件，修改 `sophia/tools/writing.py` |
| **依赖** | 无 |
| **验收标准** | ① 案例分析模板支持单案例和多案例研究设计。② 比较政治模板正确区分 MSSD 和 MDSD 两种设计。③ 法律论证模板符合法学论文标准结构。|

#### D-6：心理学模板体系

| 项 | 内容 |
|---|------|
| **目标** | 为心理学研究提供实验报告和量表开发模板 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/psychology/` 目录。(2) 创建 APA 格式实验报告模板 `apa_experiment.json`：Introduction→Method→Results→Discussion。(3) 创建量表开发流程模板 `scale_dev.json`（复用教育学模板并增加心理学特有内容如探索性/验证性因子分析）。(4) 创建元分析报告模板 `meta_analysis_report.json`：PRISMA 流程图 + 效应量报告。(5) 修改 `sophia/tools/writing.py`，增加 `paper-psychology` 类型。|
| **涉及文件** | 新建 `sophia/prompts/templates/psychology/` 下 3 个文件，修改 `sophia/tools/writing.py` |
| **依赖** | 无 |
| **验收标准** | ① APA 实验报告模板符合 APA 7th edition 要求。② 元分析报告模板包含 PRISMA 各阶段。|

#### D-7：模板注册与智能推荐

| 项 | 内容 |
|---|------|
| **目标** | 将所有学科模板注册到系统，并实现基于研究问题的模板自动推荐 |
| **具体工作** | (1) 新建 `sophia/prompts/templates/registry.py`，实现 `TemplateRegistry` 类。(2) 加载所有学科目录下的模板文件。(3) 实现 `recommend_templates(research_question, discipline=None) -> List[Dict]`，基于关键词匹配推荐最合适的模板。(4) 实现 `get_template(template_id) -> Dict`。(5) 实现 `list_templates(discipline=None) -> List[Dict]`。(6) 在 `sophia/agent.py` 中初始化 TemplateRegistry。(7) 在系统 prompt (`sophia/prompts/system.py`) 中注入可用模板列表。(8) 新建 `tests/test_template_registry.py`。|
| **涉及文件** | 新建 `sophia/prompts/templates/registry.py`，修改 `sophia/agent.py`，修改 `sophia/prompts/system.py`，新建测试文件 |
| **依赖** | 任务 D-1 至 D-6 |
| **验收标准** | ① `list_templates()` 返回所有学科的模板列表。② 输入「我想做一个关于乡村教师职业认同的质性研究」推荐教育学或社会学模板。③ 模板推荐结果包含推荐理由。④ 所有学科模板格式统一且可被正确加载。|

---

### 模块 E：理论脉络与概念史工具

**目标**：帮助文科生理清「用什么理论」「这个概念怎么演变来的」这两个核心困惑。

**现状**：`sophia/research/advisor.py` 的 `MethodologyAdvisor` 是纯方法推荐引擎，不涉及理论推荐。`sophia/research/llm.py` 的 `LLMEngine` 可以做通用分析但不专门处理理论脉络。系统完全没有理论地图、概念史、学派对比的能力。

#### E-1：理论地图生成引擎

| 项 | 内容 |
|---|------|
| **目标** | 输入研究主题，自动梳理相关理论流派、代表人物、核心命题、相互关系 |
| **具体工作** | (1) 新建 `sophia/research/theory.py`，实现 `TheoryMapper` 类。(2) 内置理论知识库 `sophia/research/data/theory_kb.json`，初始覆盖社会学、政治学、教育学、传播学、心理学的 30+ 个核心理论流派，每个条目：`{theory_id, name_en, name_cn, discipline, founders, key_concepts, core_propositions, related_theories, competing_theories, methodological_implications}`。(3) 实现 `map_theories(topic, discipline=None) -> Dict`：① LLM 路径：让 LLM 分析主题并输出理论关联 ② 规则路径：基于知识库关键词匹配。(4) 输出结构：`{topic, theories: List[TheoryNode], relations: List[TheoryRelation], recommended: List[str]}`。(5) 实现 `export_theory_map(data, format='mermaid') -> str`，支持输出为 Mermaid 图、TikZ 代码、GraphViz DOT。(6) 注册工具 `theory_map`。|
| **涉及文件** | 新建 `sophia/research/theory.py`，新建 `sophia/research/data/theory_kb.json` |
| **依赖** | 无 |
| **验收标准** | ① 输入「数字不平等」能关联到数字鸿沟理论、技术接受模型、信息剥夺理论等至少 3 个理论。② 输出的关系图中能看到理论间的继承/对立关系。③ Mermaid 格式输出能在 Markdown 渲染器中正确显示。④ 知识库覆盖至少 5 个学科各 5 个理论。|

#### E-2：概念史追踪引擎

| 项 | 内容 |
|---|------|
| **目标** | 追踪一个学术概念在不同时期、不同学科中的含义演变 |
| **具体工作** | (1) 在 `TheoryMapper` 中实现 `trace_concept(concept, language='zh') -> Dict`。(2) 输出结构：`{concept, evolution_stages: [{period, discipline, definition, key_authors, seminal_works, shift_description}], current_debates: List[str], cross_disciplinary_usage: Dict[str, str]}`。(3) 完全依赖 LLM 生成（概念史需要深度推理），回退路径给出「需要 LLM 支持」提示。(4) 支持「内卷」「社会资本」「文化资本」「数字劳动」「治理」等高频社科概念的预置结果缓存（首次 LLM 生成后存储到 ResultStore，后续直接读取）。(5) 注册工具 `concept_trace`。|
| **涉及文件** | `sophia/research/theory.py`（追加方法） |
| **依赖** | 任务 E-1 |
| **验收标准** | ① 输入「社会资本」能输出至少 3 个演变阶段（Bourdieu→Coleman→Putnam 的经典脉络）。② 每个阶段有明确的定义差异说明。③ 跨学科使用差异被标注（如政治学 vs 社会学中的不同用法）。④ 缓存命中时不再调用 LLM。|

#### E-3：学派对比表生成器

| 项 | 内容 |
|---|------|
| **目标** | 自动生成不同理论流派的假设、方法、局限性的结构化对比 |
| **具体工作** | (1) 在 `TheoryMapper` 中实现 `compare_schools(theory_ids) -> Dict`。(2) 对比维度：本体论假设、认识论立场、方法论偏好、核心概念、代表学者、经典文献、主要批评、适用场景、局限性。(3) 输出为结构化表格 + Markdown 格式。(4) 支持 LLM 辅助和知识库直接提取两种模式。|
| **涉及文件** | `sophia/research/theory.py`（追加方法） |
| **依赖** | 任务 E-1 |
| **验收标准** | ① 输入「结构功能主义 vs 冲突理论」，输出 9 个维度的完整对比。② Markdown 表格在 Web UI 和 CLI 中都能正确显示。③ 每个维度都有实质性内容而非占位符。|

#### E-4：理论工具注册与测试

| 项 | 内容 |
|---|------|
| **目标** | 注册所有理论工具，编写测试 |
| **具体工作** | (1) 新建 `sophia/tools/theory.py`，实现 `register_theory_tools(registry, mapper)`，注册 `theory_map`、`concept_trace`、`compare_schools`。(2) 在 `sophia/agent.py` 中初始化 TheoryMapper 并注册。(3) 在 `sophia/swarm/roles.py` 中为 `methodologist` 角色增加理论工具权限。(4) 新建 `tests/test_theory.py`。(5) 新建 `tests/test_theory_kb.py`（验证知识库 JSON 格式和覆盖度）。|
| **涉及文件** | 新建 `sophia/tools/theory.py`，修改 `sophia/agent.py`，修改 `sophia/swarm/roles.py`，新建 2 个测试文件 |
| **依赖** | 任务 E-1 至 E-3 |
| **验收标准** | ① 3 个工具全部注册成功。② methodologist 角色能调用理论工具。③ 测试覆盖 LLM 路径和回退路径。④ 知识库 JSON 格式合法，所有字段完整。|

---

### 模块 F：论证逻辑检查器增强

**目标**：帮助文科生检测论文中最常见的逻辑问题——逻辑跳跃、循环论证、以偏概全、证据不足。

**现状**：`sophia/review/logic.py` 的 `LogicChecker` 有三个检查方法（`_check_methodology_match`、`_check_evidence_support`、`_check_conclusion_chain`），但全部基于关键词匹配，无法做深层的论证结构分析。`sophia/review/engine.py` 的 `ReviewEngine` 统一调度 6 个维度，逻辑维度权重 0.20。

#### F-1：论证链抽取与可视化

| 项 | 内容 |
|---|------|
| **目标** | 将论文中的「前提→推理→结论」链条抽取出来，用有向图展示 |
| **具体工作** | (1) 在 `sophia/review/logic.py` 中新增 `ArgumentChainExtractor` 类。(2) 实现 `extract_chains(doc) -> Dict`，输入文档 dict，输出论证链。(3) 使用 LLM 进行结构化抽取：① 识别每个论点的前提 (premises) ② 识别推理过程 (reasoning) ③ 识别结论 (conclusion) ④ 识别支撑证据 (evidence)。(4) 构建有向图：节点类型分为 premise / reasoning / conclusion / evidence 四种，边类型分为 supports / contradicts / assumes / implies 四种。(5) 实现 `visualize_chains(chains, output_path) -> str`，输出为 Mermaid 图或 matplotlib 有向图。(6) 在图上标注薄弱环节（证据不足的推理步骤用红色标注）。|
| **涉及文件** | 修改 `sophia/review/logic.py` |
| **依赖** | 无 |
| **验收标准** | ① 输入一篇有 3 个核心论点的论文，能抽取至少 3 条论证链。② 薄弱环节（无证据支撑的推理）被正确标注。③ Mermaid 输出能正确渲染。④ 无 LLM 时给出降级提示而非报错。|

#### F-2：逻辑谬误检测器

| 项 | 内容 |
|---|------|
| **目标** | 识别论文中的常见逻辑谬误 |
| **具体工作** | (1) 在 `LogicChecker` 中新增 `_detect_fallacies(doc) -> List[Dict]`。(2) 检测的谬误类型 12 种：① 滑坡谬误 (Slippery Slope) ② 稻草人谬误 (Straw Man) ③ 虚假因果 (Post Hoc) ④ 诉诸权威 (Appeal to Authority) ⑤ 以偏概全 (Hasty Generalization) ⑥ 循环论证 (Circular Reasoning) ⑦ 诉诸情感 (Appeal to Emotion) ⑧ 虚假两难 (False Dilemma) ⑨ 诉诸传统/常识 (Appeal to Tradition) ⑩ 相关当因果 (Correlation ≠ Causation) ⑪ 幸存者偏差 (Survivorship Bias) ⑫ 采樱桃谬误 (Cherry Picking)。(3) 每种谬误的实现包含：中文和英文的触发模式（正则 + 关键词）、LLM 辅助检测 prompt、误报控制规则。(4) 检测结果包含：`{fallacy_type, location: str, evidence: str, explanation: str, suggestion: str}`。|
| **涉及文件** | 修改 `sophia/review/logic.py` |
| **依赖** | 无 |
| **验收标准** | ① 能检测出包含明确虚假因果的段落。② 对 12 种谬误至少 8 种有正确的检测规则。③ 误报率 < 30%（在 10 篇正常论文上测试，平均每篇误报 < 2 处）。④ 每处检测都有 explanation 和 suggestion。|

#### F-3：证据充分性评估

| 项 | 内容 |
|---|------|
| **目标** | 检查每个核心论点是否有足够证据支撑，标注「裸论点」 |
| **具体工作** | (1) 在 `LogicChecker` 中新增 `_check_evidence_sufficiency(doc) -> List[Dict]`。(2) 分析逻辑：① 提取所有核心论点（通过「因此」「说明」「证明了」等连接词和段首/段尾位置识别）② 检查每个论点是否紧跟着数据/引文/案例/统计结果等证据 ③ 将缺乏证据的论点标记为「裸论点」④ 对有证据的论点评估证据类型多样性（数据+引文+案例 优于 单一证据类型）。(3) 输出：`{argument, location, evidence_count, evidence_types: List[str], sufficiency_score: float, is_naked: bool, suggestions: List[str]}`。|
| **涉及文件** | 修改 `sophia/review/logic.py` |
| **依赖** | 无 |
| **验收标准** | ① 能正确识别「裸论点」（有主张但完全无证据的段落）。② 证据类型区分数据/引文/案例/逻辑推理 4 种。③ sufficiency_score 的评分逻辑合理（0-100）。④ 对论文的改进建议具体可操作。|

#### F-4：论证结构评分

| 项 | 内容 |
|---|------|
| **目标** | 给出论证完整性的量化评估和改进建议 |
| **具体工作** | (1) 在 `LogicChecker` 中新增 `_score_argument_structure(doc) -> Dict`。(2) 评分维度（每项 0-20 分，总分 100）：① 论点明确性（核心论点是否清晰可辨）② 证据充分性（证据数量和类型覆盖）③ 推理连贯性（论点之间的逻辑关系）④ 反驳考虑（是否讨论了反例或替代解释）⑤ 结论谨慎性（结论是否过度推广）。(3) 总分映射到等级：A (90-100) / B (70-89) / C (50-69) / D (0-49)。(4) 输出包含每个维度的具体扣分原因和改进建议。(5) 将此评分整合到 `ReviewEngine.review()` 的逻辑维度评分中，替代现有的简单扣分机制。|
| **涉及文件** | 修改 `sophia/review/logic.py`，修改 `sophia/review/engine.py` |
| **依赖** | 任务 F-1 至 F-3 |
| **验收标准** | ① 5 个维度的评分独立且合理。② 高质量论文（逻辑严密、证据充分）得分 > 80。③ 低质量论文（大量裸论点、逻辑跳跃）得分 < 50。④ 改进建议具体到段落级别。⑤ ReviewEngine 的逻辑维度评分使用新评分机制。|

#### F-5：逻辑检查器测试

| 项 | 内容 |
|---|------|
| **目标** | 为增强后的逻辑检查器编写完整测试 |
| **具体工作** | (1) 新建 `tests/test_argument_chain.py`（论证链抽取测试）。(2) 新建 `tests/test_fallacy_detection.py`（谬误检测测试）。(3) 新建 `tests/test_evidence_sufficiency.py`（证据充分性测试）。(4) 新建 `tests/test_argument_scoring.py`（评分机制测试）。(5) 每个测试文件包含：正面测试（正确检测到问题）、负面测试（正常文本不误报）、边界测试（空文本、超短文本、纯代码文本）。|
| **涉及文件** | 新建 4 个测试文件 |
| **依赖** | 任务 F-1 至 F-4 |
| **验收标准** | ① 所有测试通过。② 正面测试覆盖 12 种谬误检测。③ 负面测试确保 10 段正常学术文本的误报 < 3 处。④ 空文本/超短文本不报错。|

---

## Phase 3：体验增强与差异化（P2）

---

### 模块 G：访谈与问卷数据采集管线

**目标**：补全从「数据采集」到「数据分析」的完整链路，让系统不再只有分析端。

**现状**：`sophia/tools/data_collection.py` 有宏观经济数据获取（akshare）和新闻爬取功能，但完全没有质性/量化研究的社会科学数据采集工具。

#### G-1：访谈提纲自动生成

| 项 | 内容 |
|---|------|
| **目标** | 根据研究问题和理论框架自动生成半结构化访谈提纲 |
| **具体工作** | (1) 新建 `sophia/research/collection.py`，实现 `DataCollectionEngine` 类。(2) 实现 `generate_interview_guide(research_question, theory_framework=None, target_population=None, n_questions=15) -> Dict`。(3) 访谈提纲结构：① 开场白和知情同意说明 ② 热身问题（2-3 题）③ 核心问题（按主题分组，每组 3-5 题）④ 追问/探测问题（每个核心问题配 2-3 个追问）⑤ 结束问题（1-2 题）⑥ 结束语。(4) 核心问题类型覆盖：经历型、意见型、知识型、感受型、行为型 5 种。(5) 支持 LLM 生成和模板填充两种方式。(6) 输出同时提供 Markdown 文本版和结构化 JSON 版。|
| **涉及文件** | 新建 `sophia/research/collection.py` |
| **依赖** | 无 |
| **验收标准** | ① 输入研究问题「高校青年教师的工作压力与应对策略」，能生成包含 3-4 个主题、每主题 3-5 题的访谈提纲。② 每个核心问题都配有追问。③ 问题类型不单一（不是全部「你怎么看」）。④ 包含知情同意说明。|

#### G-2：问卷题目设计

| 项 | 内容 |
|---|------|
| **目标** | 根据研究假设自动生成问卷题目 |
| **具体工作** | (1) 在 `DataCollectionEngine` 中实现 `generate_questionnaire(hypotheses, constructs=None, demographics=True) -> Dict`。(2) 问卷结构：① 问卷说明 ② 人口统计学变量（年龄、性别、学历、收入等，可选）③ 量表题目（按构念/维度分组）④ 开放题（可选）。(3) 题目类型：① Likert 5/7 点量表 ② 语义差异量表 ③ 多选题 ④ 排序题 ⑤ 开放题。(4) 每道量表题目包含：题干、测量维度、反向计分标记、参考来源。(5) 内置常用量表模板库：工作满意度、自我效能感、组织承诺、主观幸福感等 10 个经典量表的维度和示例题项（标注来源，提醒用户需要获取版权许可）。(6) 注册工具 `collection_questionnaire`。|
| **涉及文件** | `sophia/research/collection.py`（追加方法），新建 `sophia/research/data/scale_templates.json` |
| **依赖** | 无 |
| **验收标准** | ① 输入假设「社交媒体使用频率与孤独感正相关」，能生成包含社交媒体使用量表和孤独感量表的问卷。② Likert 题目格式正确（1-5 或 1-7 点）。③ 反向计分题目被正确标记。④ 10 个经典量表模板格式统一且包含来源说明。|

#### G-3：音频转录辅助

| 项 | 内容 |
|---|------|
| **目标** | 将访谈录音转为文本并按说话人分段 |
| **具体工作** | (1) 在 `DataCollectionEngine` 中实现 `transcribe_audio(audio_path, language='zh') -> Dict`。(2) 优先使用 OpenAI Whisper API（通过已配置的 provider），回退到本地 whisper 模型（可选依赖），最终回退到「需要音频转录支持」提示。(3) 输出：`{segments: [{start, end, speaker, text}], full_text: str, duration: float, language_detected: str}`。(4) 说话人分段：简单模式（按静音间隔分段），高级模式（LLM 根据内容判断访问者/受访者）。(5) 支持输出为 SRT/VTT 字幕格式和纯文本格式。(6) 注册工具 `collection_transcribe`。|
| **涉及文件** | `sophia/research/collection.py`（追加方法） |
| **依赖** | openai-whisper（可选） |
| **验收标准** | ① 有 Whisper 时能正确转录中文音频（输出文字与实际内容匹配率 > 80%）。② 按说话人分段至少能区分出 2 个说话人。③ 无 Whisper 时给出清晰的安装提示。④ SRT 格式输出能在视频播放器中正确加载。|

#### G-4：编码预标注

| 项 | 内容 |
|---|------|
| **目标** | 对访谈文本自动生成初始编码建议 |
| **具体工作** | (1) 在 `DataCollectionEngine` 中实现 `precode_transcript(text, coding_approach='thematic') -> Dict`。(2) 支持三种编码路径：① `thematic`：主题分析式编码（语义层面的主题标签）② `grounded`：扎根理论式编码（开放编码→轴心编码）③ `structural`：结构化编码（按预设主题框架编码）。(3) 输出：`{codes: [{code, frequency, representative_quotes: List[str], suggested_category}], categories: List[str], coding_memos: List[str]}`。(4) 明确告知用户这是初始建议，需要人工审核和修正。(5) 注册工具 `collection_precode`。|
| **涉及文件** | `sophia/research/collection.py`（追加方法） |
| **依赖** | 任务 A-1（中文分词） |
| **验收标准** | ① 对 10 段中文访谈文本生成至少 15 个编码。② 编码标签合理（不是随机词）。③ 每个编码有至少 1 条代表性原文引用。④ coding_memos 包含对编码策略的说明。|

#### G-5：采集工具注册与测试

| 项 | 内容 |
|---|------|
| **目标** | 注册所有采集工具，编写测试 |
| **具体工作** | (1) 新建 `sophia/tools/collection.py`，实现 `register_collection_tools(registry, engine)`。(2) 注册工具：`collection_interview_guide`、`collection_questionnaire`、`collection_transcribe`、`collection_precode`。(3) 在 `sophia/agent.py` 中初始化 DataCollectionEngine 并注册。(4) 在 `sophia/swarm/roles.py` 中新增 `field_worker` 角色（田野数据采集专家），权限包含所有采集工具。(5) 新建 `tests/test_collection.py`。|
| **涉及文件** | 新建 `sophia/tools/collection.py`，修改 `sophia/agent.py`，修改 `sophia/swarm/roles.py`，新建 `tests/test_collection.py` |
| **依赖** | 任务 G-1 至 G-4 |
| **验收标准** | ① 4 个工具全部注册成功。② field_worker 角色在 swarm 中可用。③ 测试覆盖所有采集工具的正面/负面/边界情况。|

---

### 模块 H：研究设计与方法论顾问增强

**目标**：增强现有方法论顾问，提供更深入的研究设计支持。

**现状**：`sophia/research/advisor.py` 的 `MethodologyAdvisor` 有 20 条方法规则，基于规则匹配推荐方法。`sophia/research/design.py` 的 `ResearchDesignEngine` 提供 DOE 和功效分析。但缺少研究问题诊断、混合方法设计、抽样策略建议等高阶功能。

#### H-1：研究问题诊断器

| 项 | 内容 |
|---|------|
| **目标** | 输入研究问题，自动判断类型并推荐适配方法 |
| **具体工作** | (1) 在 `MethodologyAdvisor` 中新增 `diagnose_question(question) -> Dict`。(2) 判断维度：① 问题类型：描述性 / 解释性 / 探索性 / 评价性 / 设计性 ② 研究范式：实证主义 / 解释主义 / 批判理论 / 实用主义 ③ 时间维度：横截面 / 纵向 / 回溯性 ④ 分析层次：个体 / 群体 / 组织 / 社会。(3) 基于诊断结果推荐 2-3 种适配方法，每种方法给出适用性评分和理由。(4) 支持 LLM 辅助诊断和关键词规则诊断两种路径。|
| **涉及文件** | 修改 `sophia/research/advisor.py` |
| **依赖** | 无 |
| **验收标准** | ① 输入「数字经济发展对城市创新效率的影响研究」正确判断为「解释性」。② 推荐方法包含因果推断类方法。③ 诊断结果包含 4 个维度的判断和理由。|

#### H-2：混合方法设计生成器

| 项 | 内容 |
|---|------|
| **目标** | 自动生成定性+定量三角验证的混合方法方案 |
| **具体工作** | (1) 在 `MethodologyAdvisor` 中新增 `design_mixed_method(qual_question, quant_question, priority='equal') -> Dict`。(2) 支持 4 种混合设计类型（Creswell 分类）：① 聚合式设计 (Convergent Parallel) ② 解释性顺序设计 (Explanatory Sequential) ③ 探索性顺序设计 (Exploratory Sequential) ④ 嵌入式设计 (Embedded)。(3) 输出：`{design_type, rationale, qual_phase: {methods, data, analysis}, quant_phase: {methods, data, analysis}, integration_points: List[Dict], timeline: List[str], validation_strategy: str}`。(4) integration_points 明确说明定性和定量数据在哪些节点整合、如何交叉验证。|
| **涉及文件** | 修改 `sophia/research/advisor.py` |
| **依赖** | 无 |
| **验收标准** | ① 能根据问题特征自动推荐最合适的混合设计类型。② 每种设计类型都有正确的阶段顺序。③ integration_points 至少有 2 个具体整合节点。④ 输出的方案可直接用于开题报告的方法章节。|

#### H-3：抽样策略建议器

| 项 | 内容 |
|---|------|
| **目标** | 根据研究设计推荐合适的抽样策略 |
| **具体工作** | (1) 在 `MethodologyAdvisor` 中新增 `recommend_sampling(research_design, population=None, constraints=None) -> Dict`。(2) 质性抽样策略：目的性抽样、理论抽样、滚雪球抽样、最大变异抽样、典型案例抽样、关键案例抽样。(3) 量化抽样策略：简单随机抽样、分层抽样、整群抽样、多阶段抽样、便利抽样。(4) 每种策略输出：`{strategy, description, when_to_use, sample_size_guidance, pros, cons, estimated_cost}`。(5) 基于 research_design 参数自动推荐，同时提供 2-3 个备选方案。|
| **涉及文件** | 修改 `sophia/research/advisor.py` |
| **依赖** | 无 |
| **验收标准** | ① 质性研究推荐目的性抽样或理论抽样，而非随机抽样。② 每种策略有明确的适用场景说明。③ 样本量建议是具体数字或范围，而非「足够多」。|

#### H-4：顾问增强测试

| 项 | 内容 |
|---|------|
| **目标** | 为新增顾问功能编写测试 |
| **具体工作** | (1) 新建 `tests/test_advisor_enhanced.py`。(2) 测试覆盖：研究问题诊断、混合方法设计、抽样策略建议。(3) 每个功能至少 5 个测试用例覆盖不同研究场景。|
| **涉及文件** | 新建 `tests/test_advisor_enhanced.py` |
| **依赖** | 任务 H-1 至 H-3 |
| **验收标准** | ① 所有测试通过。② 现有 advisor 测试不受影响。|

---

### 模块 I：研究伦理与 IRB 支持

**目标**：帮助研究者生成伦理审查材料和进行伦理自检。

**现状**：`sophia/review/ethics.py` 的 `EthicsChecker` 只做论文内容的伦理审查（如是否涉及弱势群体、是否知情同意）。没有伦理审查申请书的生成能力。

#### I-1：伦理审查材料生成器

| 项 | 内容 |
|---|------|
| **目标** | 自动生成伦理审查申请书、知情同意书、数据管理计划 |
| **具体工作** | (1) 新建 `sophia/research/ethics.py`，实现 `EthicsDocumentGenerator` 类。(2) 实现 `generate_consent_form(study_info) -> str`：知情同意书，包含研究目的、参与方式、风险说明、隐私保护、退出权利、联系方式。(3) 实现 `generate_ethics_application(study_info) -> str`：伦理审查申请书，包含研究背景、方法、参与者招募、风险评估、数据管理、利益冲突声明。(4) 实现 `generate_data_management_plan(study_info) -> str`：数据管理计划，包含数据收集、存储、匿名化、共享、销毁。(5) `study_info` 结构：`{title, researcher, institution, participants, methods, data_types, risks, benefits, duration}`。(6) 支持中文和英文两种输出。(7) 模板基于国内高校常见伦理审查表格格式。|
| **涉及文件** | 新建 `sophia/research/ethics.py`（与 review/ethics.py 不同，这是 research 模块下的生成器），新建 `sophia/research/templates/consent_form.txt`，新建 `sophia/research/templates/ethics_application.txt` |
| **依赖** | 无 |
| **验收标准** | ① 生成的知情同意书包含所有必要段落。② 伦理申请书格式符合高校伦理审查委员会要求。③ 数据管理计划覆盖数据全生命周期。④ 中英文输出格式都正确。|

#### I-2：研究伦理自检清单

| 项 | 内容 |
|---|------|
| **目标** | 提供交互式的伦理自检功能 |
| **具体工作** | (1) 在 `EthicsDocumentGenerator` 中实现 `checklist(study_info) -> List[Dict]`。(2) 检查项覆盖 8 个维度：① 参与者保护（弱势群体、未成年人、知情同意）② 数据隐私（匿名化、数据加密、第三方共享）③ 研究诚信（数据造假、选择性报告、剽窃）④ 利益冲突（资助方影响、研究者偏见）⑤ 文化敏感（跨文化研究的尊重与误解）⑥ 风险评估（心理风险、社会风险、法律风险）⑦ 退出机制（参与者随时退出的权利和程序）⑧ 结果传播（对参与者的反馈、负面结果报告）。(3) 每个检查项：`{dimension, item, status: 'pass'/'warning'/'fail'/'unknown', details, suggestion}`。(4) 基于研究信息自动判断部分检查项，其余标记为 unknown 要求用户确认。|
| **涉及文件** | `sophia/research/ethics.py`（追加方法） |
| **依赖** | 无 |
| **验收标准** | ① 8 个维度至少各有 2 个检查项。② 涉及未成年人研究时自动触发额外的保护检查。③ 每个检查项都有明确的 pass/warning/fail 判断标准。|

#### I-3：伦理工具注册与测试

| 项 | 内容 |
|---|------|
| **目标** | 注册伦理工具，编写测试 |
| **具体工作** | (1) 新建 `sophia/tools/ethics.py`，注册工具：`ethics_consent_form`、`ethics_application`、`ethics_data_plan`、`ethics_checklist`。(2) 在 `sophia/agent.py` 中注册。(3) 新建 `tests/test_ethics_generator.py`。|
| **涉及文件** | 新建 `sophia/tools/ethics.py`，修改 `sophia/agent.py`，新建 `tests/test_ethics_generator.py` |
| **依赖** | 任务 I-1，I-2 |
| **验收标准** | ① 4 个工具注册成功。② 生成的文档格式完整。③ 伦理自检清单覆盖所有维度。|

---

### 模块 J：期刊匹配与投稿指南

**目标**：输入论文摘要，自动推荐合适期刊并提供投稿指导。

#### J-1：期刊数据库

| 项 | 内容 |
|---|------|
| **目标** | 建立中文社科期刊信息库 |
| **具体工作** | (1) 新建 `sophia/research/journal_db.py`，实现 `JournalDatabase` 类。(2) 新建 `sophia/research/data/journals.json`，初始收录中文社科核心期刊 100 本（CSSCI 来源期刊为主）。(3) 每本期刊：`{journal_id, name, name_en, discipline, level: 'CSSCI'/'北大核心'/'AMI', scope: str, keywords: List[str], impact_factor: float, review_cycle_months: int, acceptance_rate: str, submission_url: str, format_requirements: str, word_limit: int}`。(4) 实现 `search_journals(query, discipline=None, level=None) -> List[Dict]`。(5) 实现 `match_journals(abstract, keywords) -> List[Tuple[Dict, float]]`，基于关键词+scope 语义匹配。|
| **涉及文件** | 新建 `sophia/research/journal_db.py`，新建 `sophia/research/data/journals.json` |
| **依赖** | 无 |
| **验收标准** | ① 数据库包含至少 100 本 CSSCI 期刊信息。② 搜索「社会学」能返回社会学相关期刊。③ 匹配算法能将教育学论文推荐到教育学期刊而非政治学期刊。|

#### J-2：投稿指导生成器

| 项 | 内容 |
|---|------|
| **目标** | 为目标期刊生成投稿前自检清单和格式指南 |
| **具体工作** | (1) 在 `JournalDatabase` 中实现 `get_submission_guide(journal_id) -> Dict`。(2) 输出：`{journal_info, format_checklist: List[str], common_rejection_reasons: List[str], writing_tips: List[str], similar_papers: List[str]}`。(3) 格式检查清单根据期刊的 format_requirements 自动生成。(4) 常见拒稿原因从期刊公开信息中整理。(5) 注册工具 `journal_match`、`journal_guide`。|
| **涉及文件** | `sophia/research/journal_db.py`（追加方法），新建 `sophia/tools/journal.py`，修改 `sophia/agent.py` |
| **依赖** | 任务 J-1 |
| **验收标准** | ① 匹配结果返回 5-10 本期刊，按匹配度排序。② 投稿指导包含具体可操作的检查项。③ 格式要求不是泛泛的「请参考期刊模板」。|

#### J-3：期刊工具测试

| 项 | 内容 |
|---|------|
| **目标** | 编写期刊工具的完整测试 |
| **具体工作** | (1) 新建 `tests/test_journal_db.py`。(2) 测试：数据库加载、搜索、匹配、投稿指导生成。|
| **涉及文件** | 新建 `tests/test_journal_db.py` |
| **依赖** | 任务 J-1，J-2 |
| **验收标准** | ① 数据库加载测试通过。② 搜索和匹配功能正常。③ 工具注册成功。|

---

### 模块 K：学术汇报 PPT 生成

**目标**：从论文/研究报告自动生成学术会议汇报或学位论文答辩 PPT。

#### K-1：PPT 结构规划器

| 项 | 内容 |
|---|------|
| **目标** | 从论文内容自动规划 PPT 的页面结构和内容 |
| **具体工作** | (1) 新建 `sophia/exporters/pptx_export.py`，实现 `PresentationGenerator` 类。(2) 实现 `plan_slides(doc, mode='conference') -> List[SlidePlan]`，mode 支持 `conference`（会议汇报 15-20 页）和 `defense`（答辩 25-35 页）。(3) SlidePlan 结构：`{slide_number, title, bullet_points: List[str], notes: str, layout_type: 'title'/'content'/'two_column'/'image'/'table', suggested_duration_minutes: float}`。(4) 页面分配逻辑：会议汇报→背景 2 页 + 文献 2 页 + 方法 3 页 + 结果 4 页 + 讨论 2 页 + 结论 1 页 + Q&A 1 页。答辩→增加理论基础 3 页 + 研究设计细节 3 页 + 创新点 2 页 + 未来展望 2 页。(5) 自动从论文 sections 中提取关键数据和图表，标注为「建议插入图表」。|
| **涉及文件** | 新建 `sophia/exporters/pptx_export.py` |
| **依赖** | 无 |
| **验收标准** | ① 会议模式生成 15-20 页的结构规划。② 答辩模式生成 25-35 页的结构规划。③ 每页有明确的内容来源（对应论文哪个章节）。④ 标注了建议插入图表的位置。|

#### K-2：PPT 渲染器

| 项 | 内容 |
|---|------|
| **目标** | 将 SlidePlan 渲染为实际的 PPTX 文件 |
| **具体工作** | (1) 在 `PresentationGenerator` 中实现 `render(slides, output_path, theme='academic') -> str`。(2) 使用 python-pptx 生成 PPTX。（注意：这里不是批量脚本，是逐页精心的模板化渲染。）(3) 内置 3 套学术主题模板：① `academic_blue`：蓝色系，适合正式学术场合 ② `minimal_white`：白色极简，适合会议汇报 ③ `classic_dark`：深色系，适合答辩。(4) 每页的渲染逻辑：① 标题页→居中大标题+作者+单位+日期 ② 内容页→标题+要点列表+备注 ③ 双栏页→左右分栏内容 ④ 表格页→数据表格+标题 ⑤ 图片页→图片占位+标题+说明。(5) 字体：中文使用「微软雅黑」，英文使用「Calibri」，字号遵循学术惯例。(6) 注册工具 `export_pptx`。|
| **涉及文件** | `sophia/exporters/pptx_export.py`（追加方法），新建 `sophia/tools/pptx.py`，修改 `sophia/agent.py` |
| **依赖** | python-pptx（需新增到 pyproject.toml 的 export 可选依赖） |
| **验收标准** | ① 生成的 PPTX 能被 PowerPoint 和 WPS 正常打开。② 3 套主题的视觉风格有明显差异。③ 每页内容不溢出、不重叠。④ 中文字体显示正确。|

#### K-3：PPT 工具测试

| 项 | 内容 |
|---|------|
| **目标** | 编写 PPT 生成功能的测试 |
| **具体工作** | (1) 新建 `tests/test_pptx_export.py`。(2) 测试：结构规划、渲染、三种主题、中英文混合内容。|
| **涉及文件** | 新建 `tests/test_pptx_export.py` |
| **依赖** | 任务 K-1，K-2 |
| **验收标准** | ① 生成的 PPTX 文件格式合法。② 三种主题的渲染结果不同。③ 中英文混合内容排版正确。|

---

### 模块 L：多语言学术翻译

**目标**：支持中英学术互译、术语一致性维护、小语种文献摘要翻译。

#### L-1：学术翻译引擎

| 项 | 内容 |
|---|------|
| **目标** | 提供保持学术术语一致性的翻译能力 |
| **具体工作** | (1) 新建 `sophia/research/translation.py`，实现 `AcademicTranslator` 类。(2) 实现 `translate(text, source_lang, target_lang, domain=None, glossary=None) -> Dict`。(3) domain 支持：`social_science` / `education` / `history` / `literature` / `politics`。(4) 翻译流程：① 术语预扫描（提取学术术语）② 术语表查找（优先使用用户提供的 glossary）③ LLM 翻译（带术语约束的 prompt）④ 后处理（检查术语一致性、格式保留）。(5) 输出：`{translated_text, terminology_used: List[Dict], warnings: List[str]}`。(6) 支持 markdown 格式保留（标题、列表、表格在翻译后结构不变）。(7) 完全依赖 LLM，无 LLM 时给出提示。|
| **涉及文件** | 新建 `sophia/research/translation.py` |
| **依赖** | 无 |
| **验收标准** | ① 中译英后学术术语前后一致（如「社会资本」始终译为「social capital」）。② Markdown 格式不丢失（标题层级、表格结构保持）。③ 翻译结果通顺，不含直译痕迹。④ 术语使用有 warning 时明确标注。|

#### L-2：术语表管理

| 项 | 内容 |
|---|------|
| **目标** | 管理跨语言的学术术语对照表 |
| **具体工作** | (1) 在 `AcademicTranslator` 中实现术语表管理功能。(2) 实现 `add_term(term, translations: Dict[str,str], domain) -> None`。(3) 实现 `lookup_term(term, source_lang, target_lang) -> str`。(4) 实现 `import_glossary(file_path, format='csv') -> None`。(5) 实现 `export_glossary(format='csv') -> str`。(6) 持久化到 workspace `/.sophia/glossary_translation.json`。(7) 内置初始术语表覆盖 5 个学科各 100 个核心术语的中英对照（共 500 条）。|
| **涉及文件** | `sophia/research/translation.py`（追加方法），新建 `sophia/research/data/glossary_initial.json` |
| **依赖** | 无 |
| **验收标准** | ① 初始术语表覆盖 5 个学科各 100 条。② 术语查找速度 < 10ms。③ CSV 导入/导出格式正确。④ 自定义术语优先于内置术语。|

#### L-3：小语种文献摘要翻译

| 项 | 内容 |
|---|------|
| **目标** | 支持日语、德语、法语文献的摘要翻译 |
| **具体工作** | (1) 在 `AcademicTranslator` 中实现 `translate_abstract(text, source_lang) -> Dict`。(2) source_lang 支持：`ja`（日语）/ `de`（德语）/ `fr`（法语）/ `ru`（俄语）/ `ko`（韩语）。(3) 输出包含：中文摘要和英文摘要（双语文本）。(4) 完全依赖 LLM 翻译。(5) 注册工具 `translate_academic`、`translate_abstract`。|
| **涉及文件** | `sophia/research/translation.py`（追加方法） |
| **依赖** | 任务 L-1 |
| **验收标准** | ① 日语文献摘要能正确翻译为中文（不影响专业术语）。② 输出同时包含中文和英文版本。③ 小语种检测正确（不会把德语误判为英语）。|

#### L-4：翻译工具注册与测试

| 项 | 内容 |
|---|------|
| **目标** | 注册翻译工具，编写测试 |
| **具体工作** | (1) 新建 `sophia/tools/translation.py`，注册工具。(2) 在 `sophia/agent.py` 中注册。(3) 新建 `tests/test_translation.py`。|
| **涉及文件** | 新建 `sophia/tools/translation.py`，修改 `sophia/agent.py`，新建 `tests/test_translation.py` |
| **依赖** | 任务 L-1 至 L-3 |
| **验收标准** | ① 工具注册成功。② 测试覆盖翻译、术语管理、摘要翻译。③ 无 LLM 时给出正确提示而非报错。|

---

## 全局任务

### G-1：pyproject.toml 依赖更新

| 项 | 内容 |
|---|------|
| **目标** | 将所有新增的可选依赖正确添加到 pyproject.toml |
| **具体工作** | (1) `[analysis]` 新增：jieba, pkuseg, snownlp, gensim, bertopic。(2) `[export]` 新增：python-pptx。(3) `[advanced]` 新增：pyvis。(4) 新增 `[nlp-cn]` 可选依赖组：jieba, pkuseg, snownlp, gensim, bertopic。(5) 更新 `[all]` 包含所有新依赖组。(6) 所有新增依赖均为可选，核心安装不受影响。|
| **涉及文件** | 修改 `pyproject.toml` |
| **验收标准** | ① `pip install -e ".[all]"` 不报错。② `pip install -e .`（核心安装）不包含新依赖。③ `pip install -e ".[nlp-cn]"` 安装中文 NLP 相关包。|

### G-2：系统 Prompt 更新

| 项 | 内容 |
|---|------|
| **目标** | 更新系统提示以反映新增能力 |
| **具体工作** | (1) 修改 `sophia/prompts/system.py` 的 `SYSTEM_PROMPT`。(2) 在核心能力描述中增加：中文质性分析、论文精读、英文润色、理论脉络、学科模板。(3) 在学术写作标准中增加：学科专属规范引用说明。(4) 增加中文写作风格规则（如果当前只有英文写作规则）。|
| **涉及文件** | 修改 `sophia/prompts/system.py` |
| **验收标准** | ① 系统提示包含所有新功能的描述。② 新增规则不与现有规则矛盾。③ 提示总长度不超过 4000 字符（避免过度占用 context）。|

### G-3：Web UI 适配

| 项 | 内容 |
|---|------|
| **目标** | 确保 Web UI 能正确展示所有新增功能 |
| **具体工作** | (1) 检查 `sophia/web/templates/index.html` 的 Markdown 渲染是否支持 Mermaid（用于理论地图和论证链可视化）。(2) 检查 `sophia/web/static/app.js` 的 tool card 渲染是否支持新的工具类型。(3) 检查笔记系统的数据是否需要新的 API endpoint（如 `/api/notes`）。(4) 确保 PPTX 和翻译结果的下载功能正常。|
| **涉及文件** | 修改 `sophia/web/templates/index.html`，修改 `sophia/web/static/app.js`，修改 `sophia/web/__init__.py` |
| **验收标准** | ① Mermaid 图在 Web UI 中正确渲染。② 新工具的调用和结果显示正常。③ 笔记 API 可访问。④ PPTX 和翻译结果可下载。|

### G-4：回归测试验证

| 项 | 内容 |
|---|------|
| **目标** | 确保所有新增功能不影响现有功能的正确性 |
| **具体工作** | (1) 运行完整测试套件 `pytest tests/ -v`，确保所有现有测试通过。(2) 运行语法编译 `python -m compileall sophia tests`。(3) 运行 `sophia doctor` 检查。(4) 检查工具列表 `sophia tools list` 确认所有新旧工具都可见。|
| **涉及文件** | 无新文件 |
| **验收标准** | ① 全部现有测试通过。② 无语法错误。③ `sophia doctor` 无新警告。④ 工具列表包含所有新增工具。|

---

## 依赖关系总览

```
Phase 1:
  A-1 → A-2 → A-3 → A-7 → A-8
  A-1 → A-4, A-5, A-6 → A-8
  B-1 → B-2, B-3, B-5 → B-6
  B-1 → B-4
  C-1 → C-2, C-3, C-4 → C-6
  C-5（独立）

Phase 2:
  D-1~D-6（相互独立）→ D-7
  E-1 → E-2, E-3 → E-4
  F-1~F-3（相互独立）→ F-4 → F-5

Phase 3:
  G-1~G-4（相互独立）→ G-5
  H-1~H-3（相互独立）→ H-4
  I-1, I-2（相互独立）→ I-3
  J-1 → J-2 → J-3
  K-1 → K-2 → K-3
  L-1 → L-3 → L-4
  L-2（独立）

全局:
  G-1~G-4 依赖所有模块完成
```

---

## 工具量统计

| 模块 | 新增工具数 | 新增文件数 | 修改文件数 | 新增测试文件 |
|---|---|---|---|---|
| A. 中文 NLP | 9 | 3 | 4 | 4 |
| B. 精读笔记 | 10 | 4 | 3 | 4 |
| C. 英文写作 | 7 | 3 | 3 | 2 |
| D. 学科模板 | 3 | 21 | 3 | 1 |
| E. 理论脉络 | 3 | 2 | 3 | 2 |
| F. 逻辑检查 | 0（增强现有） | 0 | 2 | 4 |
| G. 数据采集 | 4 | 2 | 3 | 1 |
| H. 方法顾问 | 0（增强现有） | 0 | 1 | 1 |
| I. 伦理支持 | 4 | 3 | 1 | 1 |
| J. 期刊匹配 | 2 | 2 | 1 | 1 |
| K. PPT 生成 | 1 | 2 | 1 | 1 |
| L. 多语言翻译 | 2 | 2 | 1 | 1 |
| 全局 | 0 | 0 | 4 | 0 |
| **合计** | **45** | **44** | **30** | **23** |
