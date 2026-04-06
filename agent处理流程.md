# Agent 处理流程

## 目标

输入两个 JSON 文件路径：

- 第一个参数：原始输入 JSON 路径
- 第二个参数：最终输出 JSON 路径

目标是将原始 transcript 结构化为适合英语学习的结果，并尽可能映射到 `semantic.coarse_unit`，为每个语言片段补充稳定的 coarse 语义信息。

整体分为四个阶段：

1. 阶段 0：数据清洗
2. 阶段 1：LLM 分片与解释生成
3. 阶段 2：代码侧校验与 token 编号
4. 阶段 3：数据库 coarse_unit 映射与结果回填

---

## 一、输入与输出

### 1. 原始输入结构

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "start": 13054,
      "end": 16441,
      "tokens": [
        {
          "index": 0,
          "text": "Okay,",
          "start": 13054,
          "end": 13391
        },
        {
          "index": 1,
          "text": "so",
          "start": 14723,
          "end": 14964
        },
        {
          "index": 2,
          "text": "what",
          "start": 14964,
          "end": 15109
        }
      ]
    }
  ]
}
```

### 2. 最终输出结构

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "explanation": "好吧，所以你到底是想跟我谈什么呢？",
      "tokens": [
        {
          "index": 4,
          "text": "wanted to",
          "explanation": "想要，当时想要，表示过去的意图或愿望。",
          "semanticElement": {
            "coarse_id": 166670,
            "baseForm": "want",
            "dictionary": "想要；希望；需要。",
            "reason": "第三阶段判断该 token 在当前语境中对应 want 的动词义，且 coarse_id 已通过数据库候选校验。"
          }
        }
      ]
    }
  ]
}
```

说明：

- `coarse_id` 允许为 `null`
- `dictionary` 表示最终保留的释义文本
- `semanticElement.reason` 表示第三阶段对该 token 的匹配或失败原因，必须为非空字符串
- 如果数据库成功匹配并判定语义一致，则 `dictionary` 使用数据库返回的 `chinese_def`
- 如果数据库没有可靠匹配，则保留第一阶段生成的词典释义

---

## 二、阶段 0：数据清洗

### 目标

压缩输入，减少上下文消耗，但保留句子顺序与原始 token 顺序。

### 清洗后传给第一阶段 LLM 的结构

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "tokens": [
        "Okay,",
        "so",
        "what",
        "is",
        "it",
        "that",
        "you",
        "wanted",
        "to",
        "talk",
        "to",
        "me",
        "about?"
      ]
    }
  ]
}
```

### 清洗规则

- 保留 `sentence.index`
- 保留 `sentence.text`
- `tokens` 仅保留原始 token 的 `text`
- 不把 token-level `index/start/end` 传给 LLM
- 原始 token 顺序必须在代码侧保留，供后处理校验使用

说明：

- 不信任 LLM 输出的 index 是对的
- 但代码仍然要保留原始 token 序列，后续用于验证模型输出是否只是在“按顺序合并原 token”

---

## 三、阶段 1：LLM 分片与解释生成

### 执行单位

阶段 1 不是逐句单独调用，而是按原始句子顺序，每 3 个句子组成一个批次，顺序执行。

规则：

- 按输入中的 `sentences` 顺序分批
- 每批最多包含 3 个句子
- 不并行打乱顺序执行
- 最后不足 3 句时，按实际句子数作为最后一个批次
- 每个批次的输出句子顺序必须与输入顺序完全一致

### 批次执行方式

为了复用上下文，整体流程按批次串行推进。

每个批次都要执行完整流程：

1. 阶段 1：LLM 分片与解释生成
2. 第二阶段：代码侧校验与 token 编号
3. 第三阶段：数据库 coarse_unit 映射
4. 将当前批次结果合并到累计结果
5. 立即保存一次当前累计结果到输出文件

然后再开始下一个批次。

说明：

- 不是“所有批次都做完阶段 1，再统一做阶段 3”
- 而是“每个批次单独走完整链路并保存一次”
- 这样可以复用当前上下文，也可以保留中间进度
- 若流程中断，可通过已保存文件中的最后一个 `sentence.index` 判断已完成到哪里
- 续跑时应从最后一个已成功保存的 `sentence.index` 之后继续
- 第三阶段必须直接使用第二阶段产出的 JSON 作为新输入开启新对话
- 第三阶段不继续 append 第一阶段的对话上下文

### 输出文件写入规则

每次批次保存时，输出文件必须采用“原子替换”方式写入，不能直接覆盖目标文件。

要求：

- 先将当前累计结果写入临时文件
- 临时文件写入完成并确认成功后，再通过 `rename` / 原子替换方式覆盖目标输出文件
- 不允许直接打开目标输出文件后原地覆盖写入

这样可以避免：

- 处理中断时留下半截 JSON
- 目标文件处于部分写入状态

### 临时文件命名规则

为了避免多进程并行执行时临时文件互相冲突，临时文件名必须基于：

- 原始输入 JSON 文件名
- 随机字符串

要求：

- 临时文件必须创建在目标输出文件所在目录下的 `temp/` 子目录中
- 如果 `temp/` 不存在，则先创建
- 如果 `temp/` 已存在，则直接复用
- 随机字符串必须足够避免并发冲突
- 不同进程即使处理同一个输出目标，也不能复用同一个临时文件名
- 文件名基于原始输入 JSON 文件名再加随机字符串生成
- 每次批次保存完成后，必须清理本次产生的临时文件
- `temp/` 目录本身可以保留，不要求清理

### 搜索审计记录

除了用于原子写入的临时文件，还应为第三阶段的每个 token 保存搜索审计记录。

要求：

- 搜索审计记录文件保存在目标输出文件所在目录下的 `temp/` 子目录中
- 搜索审计记录文件名固定为：`<原始输入 JSON 文件名>.search_audit.jsonl`
- 搜索审计记录文件在流程结束后不清理，保留用于排查和审计

建议记录内容：

- `sentence.index`
- `token.index`
- 第 1 / 2 / 3 回的 `mode`
- 每一回实际使用的 `queries`
- 每一回返回的候选数量
- agent 最终动作：`match` / `search` / `no_match`
- 最终 `coarse_id`
- 最终 `reason`

### 模型

`OPENAI_API_KEY` 从项目根目录 `.env` 读取。

第一阶段与第三阶段都采用：

- `langchain`
- `langchain_openai`
- OpenAI 模型
- structured output

原则：

- 输出结构由代码中的 schema 约束
- prompt 只负责描述任务规则
- 不在 prompt 中重复手写 JSON schema

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4-mini",
    reasoning_effort="medium",
)
```

### 任务目标

对每个批次中的每个句子完成三件事：

1. 生成整句中文解释
2. 将原始 token 按顺序合并成更适合英语学习的语言片段
3. 为每个片段生成上下文解释和基础词典信息

### 第一阶段结构化输出 Schema

第一阶段输出结构由代码中的 schema 定义，不在 prompt 中手写 JSON schema。

推荐用 `Pydantic` 定义：

```python
from pydantic import BaseModel

class SemanticElement(BaseModel):
    baseForm: str
    dictionary: str

class TokenItem(BaseModel):
    text: str
    explanation: str
    semanticElement: SemanticElement

class SentenceItem(BaseModel):
    index: int
    text: str
    explanation: str
    tokens: list[TokenItem]

class StageOneOutput(BaseModel):
    sentences: list[SentenceItem]
```

调用方式：

```python
structured_llm = llm.with_structured_output(StageOneOutput)
```

### 第一阶段硬约束

- 只能组合输入中已有的原始 tokens
- 不能改写 token 文本内容
- 不能遗漏任何原始 token
- 不能重复使用 token
- 不能打乱 token 顺序
- 不能跨句合并
- 输出中的每个 `tokens[].text` 必须能被代码按顺序映射回原始 token 序列

### 合并原则

以英语学习为目标进行分片。

应该优先合并：

- 固定短语
- 常见搭配
- 需要整体理解的语法片段
- 对学习者更有价值的多词表达

不应该合并：

- 只是普通相邻单词、但不构成固定表达的片段
- 仅仅因为常同时出现而机械拼接的结构

### 正面例子

- `by the way` -> 合并为一个 token
- `deal with` -> 合并为一个 token
- `at the same time` -> 合并为一个 token
- `looking forward to` -> 合并为一个 token
- `be addicted to` -> 合并为一个 token
- `get rid of` -> 合并为一个 token

### 反面例子

- `I am happy` -> 不合并成一个 token
- `the book` -> 不合并成一个 token
- `very good` -> 不合并成一个 token

### baseForm 规则

- `running` -> `run`
- `studies` -> `study`
- `wanted to` -> `want`
- `is addicted to` / `be addicted to` -> `addicted`
- `talk to` -> `talk`

### 第一阶段 Prompt

```text
请将英文文本进行结构化分析，并严格按输入 token 顺序完成语言分片与解释生成。

你必须遵守以下规则：

1. 对输入批次中的每个句子都输出一个整句中文 explanation。
2. 以输入 tokens 为唯一基础单位，只能将相邻 tokens 合并，不能改写文本。
3. 输出必须完整覆盖所有输入 tokens。
4. 输出 token 之间不得重叠、不得跳词、不得打乱顺序。
5. 合并目标以英语学习为导向：
   - 固定短语、常见搭配、需要整体理解的结构优先合并
   - 普通自由组合不要强行合并
6. 对每个输出 token：
   - 提供符合上下文的 explanation
   - 生成 semanticElement.baseForm
   - 生成 semanticElement.dictionary
7. 输入会包含按顺序排列的最多 3 个句子，你必须保持输出句子顺序与输入完全一致。
8. 输出结构由系统提供的 structured output schema 约束，不要自行改变字段结构。
9. 不要输出 schema 说明、markdown、解释性前后缀，只返回结构化结果。

正面例子：
- by the way
- deal with
- at the same time
- looking forward to
- be addicted to
- get rid of

反面例子：
- I am happy
- the book
- very good

输出结构不需要你手写，按系统提供的 structured output schema 返回即可。
```

### 第一阶段示例

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "explanation": "好吧，所以你到底是想跟我谈什么呢？",
      "tokens": [
        {
          "text": "Okay,",
          "explanation": "好吧，表示接受对方继续说，或引出接下来的话。",
          "semanticElement": {
            "baseForm": "okay",
            "dictionary": "好；可以；没问题；表示同意、接受或开始回应。"
          }
        },
        {
          "text": "so",
          "explanation": "那么，所以，这里用于承接前文并引出问题。",
          "semanticElement": {
            "baseForm": "so",
            "dictionary": "所以；那么；因此；用于连接上下文或推进话题。"
          }
        },
        {
          "text": "what is it that",
          "explanation": "到底是……，这是一个强调式提问结构，用来突出真正想问的内容。",
          "semanticElement": {
            "baseForm": "what",
            "dictionary": "什么；用于提问事物或内容。"
          }
        },
        {
          "text": "you",
          "explanation": "你。",
          "semanticElement": {
            "baseForm": "you",
            "dictionary": "你；你们。"
          }
        },
        {
          "text": "wanted to",
          "explanation": "想要，当时想要，表示过去的意图或愿望。",
          "semanticElement": {
            "baseForm": "want",
            "dictionary": "想要；希望；需要。"
          }
        },
        {
          "text": "talk to",
          "explanation": "和……谈、跟……说话；表示与某人交谈。",
          "semanticElement": {
            "baseForm": "talk",
            "dictionary": "说话；谈话；交谈。"
          }
        },
        {
          "text": "me",
          "explanation": "我，作宾语，表示谈话对象是“我”。",
          "semanticElement": {
            "baseForm": "me",
            "dictionary": "我；I 的宾格形式。"
          }
        },
        {
          "text": "about?",
          "explanation": "关于……，用于询问谈话的主题或内容。",
          "semanticElement": {
            "baseForm": "about",
            "dictionary": "关于；涉及；围绕。"
          }
        }
      ]
    }
  ]
}
```

---

## 四、第二阶段：代码侧校验与 token 编号

这一步不由 LLM 完成，而由程序完成。

### 必做校验

- `sentence.index` 是否保留
- 输出句子数量是否与输入一致
- 输出句子顺序是否与输入一致
- 每个输出 token 是否能按顺序映射回原始 token 列表
- 是否存在漏 token、重 token、跳 token、跨句合并

### token 编号

校验通过后，由代码统一给每个句子的所有 token 增加 `index`。

规则：

- `tokens[].index` 由代码生成
- 每个句子内从 `0` 开始
- 按该句中 token 的最终顺序连续递增
- 第二阶段生成的 `tokens[].index` 作为第三阶段定位当前 token 的依据

### 第二阶段产物

第二阶段的输出是一个新的 JSON：

- 结构沿用第一阶段结果
- 但每个 token 都已经带有 `index`
- 该 JSON 会直接作为第三阶段的新输入

### 校验失败处理

- 若某句无法顺序对齐，则该批次视为失败
- 对失败批次重试一次第一阶段
- 若重试后仍失败，则整个流程直接失败
- 当前失败批次不进入阶段 3
- 当前失败批次不写入输出文件
- 之前已经成功保存的批次结果保留，作为中间结果

---

## 五、第三阶段：数据库 coarse_unit 映射

第三阶段必须开启新对话处理，不能继续 append 第一阶段的上下文。

### 第三阶段输入

第三阶段直接使用第二阶段输出的带 `tokens.index` 的 JSON 作为新输入。

要求：

- 新对话只接收第二阶段产物
- 不带入第一阶段的历史对话内容
- 当前 token 由 `sentence.index + token.index` 唯一确定
- `A` 直接使用“当前批次完整的第二阶段 JSON”，不做额外裁剪优化

### 第三阶段上下文策略

假设第二阶段输出整体 JSON 为 `A`。

第三阶段不是把所有 token 的搜索过程连续 append 在同一个长对话里，而是采用“共享根上下文 + token 分支”的方式。

规则：

- `A` 是第三阶段所有 token 共用的根上下文
- 处理某个 token 时，只在 `A` 的基础上追加该 token 自己的搜索与判断对话
- 同一个 token 的最多三回搜索，使用 context append 的形式持续追加
- 一旦切换到下一个 token，必须回到 `A`
- 上一个 token 的搜索上下文不能带到下一个 token

示意：

- 处理 token `a`：`A + a1 + a2 + a3`
- 处理 token `b`：`A + b1 + b2 + b3`

说明：

- `a1/a2/a3` 表示 token `a` 在三回搜索中的对话增量
- `b1/b2/b3` 表示 token `b` 在三回搜索中的对话增量
- 当从 `a` 切换到 `b` 时，`a1/a2/a3` 必须丢弃，只保留 `A`
- 这样可以节省上下文，同时避免前一个 token 的搜索结果污染后一个 token 的判断

### 第三阶段缓存前缀设计

为了更稳定地利用 context cache，建议把第三阶段请求拆成：

- `H`：固定前缀
- `A`：第二阶段输出 JSON
- `a1/a2/a3`：token `a` 在三回搜索中的增量上下文
- `b1/b2/b3`：token `b` 在三回搜索中的增量上下文

推荐理解方式：

- `H + A + a1 + a2 + a3`
- `H + A + b1 + b2 + b3`

其中：

- `H` 应尽量固定不变
- `A` 是当前批次共享的第二阶段产物
- 对同一批次中的所有 token，`H + A` 都应该保持完全一致
- 不同 token 之间只有最后追加的增量部分不同

#### H 应包含什么

`H` 推荐包含：

- system prompt
- 第三阶段规则
- 输出协议
- 工具定义
- structured output schema

`A` 作为共享业务上下文，单独放在 `H` 后面。

#### 工程实现建议

这里的 `H + A`、`H + B` 是抽象概念，不要求一定手工把用户 prompt 拼成一个大字符串。

更推荐的实现方式是：

- 使用固定的 `messages` 前缀表示 `H`
- 在其后追加相同的第二阶段 JSON `A`
- 再在末尾追加当前 token 的增量消息

也就是说，工程上应优先保证“请求前缀完全一致”，而不是执着于“字符串相加”这个形式。

#### messages 形式示意

可将第三阶段请求理解为：

```ts
const H = [
  { role: "system", content: "第三阶段 system prompt" },
  { role: "user", content: "第三阶段规则与输出协议" },
];

const A = {
  role: "user",
  content: "第二阶段输出 JSON",
};

const tokenAReq = [
  ...H,
  A,
  { role: "user", content: "token a 第 1 / 2 / 3 回搜索增量" },
];

const tokenBReq = [
  ...H,
  A,
  { role: "user", content: "token b 第 1 / 2 / 3 回搜索增量" },
];
```

这里真正需要稳定复用的是前缀 `H + A`。

#### 为什么不建议机械地拼成一个大字符串

因为缓存关注的是“完全相同的请求前缀”，不是“语义差不多”。

如果采用 `messages` 形式，更稳妥的做法是：

- 固定 system / user / assistant 历史消息顺序
- 固定工具定义顺序
- 固定 schema 顺序
- 只在最后追加当前 token 的增量消息

这样更容易保证 `H + A` 真正一致。

#### 缓存命中注意事项

- 缓存依赖精确前缀匹配
- 只要前缀中有任何变化，命中率就会下降
- 可缓存的不只是纯文本，还包括 messages、tools、images、structured output schema 等请求前缀部分
- 只有当前缀足够长时，缓存才会实际生效；如果前缀太短，`cached_tokens` 可能为 `0`

### 第三阶段推荐实现方式

第三阶段推荐使用：

- `LangChain + OpenAI`
- structured output
- 显式控制器
- 显式查询执行器

不推荐使用：

- 通用 `AgentExecutor`
- ReAct 风格 agent
- 让 LLM 自行调用工具
- 依赖隐式 `memory` 维护长对话状态

原因：

- 第三阶段轮次固定，最多 3 回
- 查询工具固定，只有 `query_coarse_units.py`
- `coarse_id` 必须由脚本校验
- 每个 token 的上下文必须严格隔离
- 这些约束更适合“代码控流程，LLM 做结构化决策”

推荐组件拆分：

- `StageThreeController`
- `StageThreeDecisionLLM`
- `CoarseQueryRunner`
- `AuditLogger`

职责：

- `StageThreeController`
  - 遍历当前批次 token
  - 维护 `H + A`
  - 为每个 token 创建独立分支
  - 控制三回搜索
  - 校验 `coarse_id`
  - 回填最终结果
- `StageThreeDecisionLLM`
  - 只返回 `match` / `search` / `no_match`
- `CoarseQueryRunner`
  - 调用 `python3 supabase/query_coarse_units.py`
  - 处理 query 去重与结果解析
- `AuditLogger`
  - 将每个 token 的搜索过程追加写入 `temp/<原始输入 JSON 文件名>.search_audit.jsonl`

实现原则：

- 查询由脚本执行，不由 LLM 执行
- 每次 LLM 调用都显式构造 `messages`
- 不依赖隐藏会话状态
- 切换 token 时必须回到完全相同的 `H + A`
- 同一个 token 的第 2 / 3 回只是在前一次分支基础上追加增量消息

### 查询接口

使用脚本：

- [supabase/query_coarse_units.py](/Users/evan/Downloads/simple-audio-processing/supabase/query_coarse_units.py)

支持两种模式：

- `exact`
- `contain`

### 输入

对第二阶段产出的每个 token 做 coarse_unit 查询。

每个 token 至少具备：

- `index`
- `text`
- `semanticElement.baseForm`
- `explanation`
- `semanticElement.dictionary`

### 搜索策略

第三阶段允许 agent 在后续轮次自由发挥搜索路径，但总流程仍然固定，最多只搜索 3 回。

每次只处理一个 token，并且一次对话最多只确认这一个 token 的匹配结果。

这里的“1 回搜索”定义为：调用一次 `query_coarse_units.py`。

由于脚本一次最多支持 4 个查询参数，所以：

- 第 1 回虽然会同时查询 `token.text` 和 `baseForm`，但只算 1 回搜索
- 第 2 回最多传入 4 个 `exact` 查询词，算 1 回搜索
- 第 3 回最多传入 4 个 `contain` 查询词，算 1 回搜索

#### 第 1 回：默认自动搜索

固定执行：

1. `exact(token.text)`
2. `exact(baseForm)`

处理规则：

- 这两项 exact 查询属于同一次默认自动搜索
- 如果其中任意一步已经找到可可靠匹配的 coarse_unit，则立即停止该 token 的搜索，进入结果回填
- 如果两次都没有找到可匹配结果，则进入第 2 回

#### 第 2 回：agent 自由发挥，exact 模式

如果第 1 回未匹配成功，则允许 agent 自由决定搜索词，但仍使用 `exact` 模式。

限制：

- 最多执行 4 个相似查询词
- 这些查询词可基于 `token.text`、`baseForm`、时态还原、短语核心词、近似表达等自由调整
- 只要任意一个查询词命中并确认匹配成功，就立即停止该 token 的搜索，进入结果回填
- 若最多 4 个 exact 相似查询词仍未匹配成功，则进入第 3 回

#### 第 3 回：agent 自由发挥，contain 模式

如果前两回都未匹配成功，则允许 agent 自由决定搜索词，并使用 `contain` 模式。

限制：

- 最多执行 4 个相似查询词
- 只要任意一个查询词命中并确认匹配成功，就立即停止该 token 的搜索，进入结果回填
- 若这一回仍未匹配成功，则该 token 视为映射失败

#### 搜索终止条件

- 三回搜索中任意时点一旦匹配成功，立即停止搜索该 token
- 匹配成功后直接进入下一步，不再继续后续搜索
- 三回全部结束仍未匹配成功，则 `coarse_id = null`

### 候选筛选原则

agent 只能在返回候选中做“语义是否一致”的判断，不允许凭空生成 coarse_id。

优先考虑：

- `label` 与 token 文本或 baseForm 的匹配度
- `kind`
- `pos`
- `chinese_def`
- `chinese_criteria`
- `chinese_label`

### 映射成功后的处理

如果数据库里存在语义匹配成功的 coarse_unit：

- LLM 只返回它确认的 `coarse_id` 与匹配 `reason`
- 脚本必须先校验该 `coarse_id` 是否真实存在于当前轮数据库候选中
- 若校验成功：
  - `semanticElement.coarse_id = 命中的 id`
  - 用数据库返回的 `label` 替换 `semanticElement.baseForm`
  - 用数据库返回的 `chinese_def` 替换 `semanticElement.dictionary`
  - 将匹配原因写入 `semanticElement.reason`
  - 可以基于数据库释义微调当前 token 的 `explanation`

### 映射失败后的处理

如果没有可靠匹配：

- `semanticElement.coarse_id = null`
- 保留第一阶段生成的 `explanation`
- 保留第一阶段生成的 `dictionary`
- `semanticElement.reason` 必须写入明确的失败原因

---

## 六、第三阶段 Prompt 约束

```text
你的任务是：根据当前单词或短语在具体句子里的意思，从给定 coarse_unit 候选中选择最匹配的一个具体义项。

请把 coarse_unit 理解为“可学习语义单元”：
- 它可以对应单词
- 也可以对应短语
- 它表示的是某个单词或短语在具体语境中的一个具体意思

重点不是匹配表面字符串，而是匹配“这个词或短语在这里到底是什么意思”。

同一个单词如果意思差别很大，通常对应不同 coarse_unit。
例如：
- bank
  - I deposited money in the bank. -> 金融机构
  - We sat on the river bank. -> 河岸
- light
  - Turn on the light. -> 灯
  - This bag is light. -> 轻的
- run
  - I run every morning. -> 跑步
  - She runs the company. -> 经营、管理

短语也一样要按具体意思判断：
- take off
  - The plane took off. -> 起飞
  - He took off his jacket. -> 脱下
- work out
  - I work out at the gym. -> 锻炼
  - We need to work out the problem. -> 解决、想出办法

你需要做的事只有两类：
- 如果当前候选里已经有可靠匹配，就选出最合适的那个 coarse_unit
- 如果当前候选还不足以确认匹配，就给出下一步更合适的搜索词

判断时必须优先看语义是否一致，而不是只看字符串是否相似。
```

```text
请按下面的逻辑完成判断。

你该做什么：
- 你只处理当前这一个 token，当前处理对象由 sentence.index + token.index 唯一确定
- 你要判断：当前 token 在这个具体语境里的意思，是否已经能可靠映射到某个 coarse_unit

什么情形下返回 `match`：
- 当前候选里已经有语义可靠、最贴合当前语境的 coarse_unit
- 你可以明确判断“这个词或短语在这里就是这个意思”
- 返回 `match` 时，只返回 coarse_id 和 reason
- 如果你认为有必要优化 token 的 explanation，也可以同时返回完整优化后的 explanation

什么情形下返回 `search`：
- 当前候选还不足以确认匹配
- 但你认为继续搜索仍然有意义，可能找到更合适的义项

`search` 的用法：
- `search` 的作用是告诉系统：下一回应该查哪些词，以及用什么方式查
- `search.mode = exact`
  - 表示下一回做大小写不敏感的精确匹配
  - 适合查你已经比较确定的词或短语
- `search.mode = contain`
  - 表示下一回做包含匹配
  - 适合在精确匹配找不到时扩大范围查相近表达
- `search.queries`
  - 应该是你建议系统继续查询的词或短语

什么情形下返回 `no_match`：
- 已经到最后一回，仍然没有语义可靠的候选
- 当前候选虽然表面相似，但你无法确认意思一致
- 继续搜索也不太可能得到更可靠结果
- 返回 `no_match` 的意思是：当前 token 不应该绑定到任何 coarse_unit
```

---

## 七、第三阶段 agent 交流设计

第三阶段不应设计成“LLM 自己随意搜索、自己决定流程”的模式，而应设计成：

- 代码控制搜索流程
- agent 只负责判断当前候选是否匹配，以及在允许时给出下一回搜索词

也就是说，第三阶段采用“控制器 + 代理”的交互方式。

### LangChain 实现原则

第三阶段在工程上应实现为“确定性控制器 + 结构化决策模型”，而不是通用 agent 框架。

要求：

- 使用 `ChatOpenAI`
- 使用 `with_structured_output(...)`
- 由脚本控制搜索轮次
- 由脚本调用 `query_coarse_units.py`
- LLM 只负责返回结构化动作

不采用：

- `AgentExecutor`
- ReAct
- tools agent
- 自动工具调用链
- 隐式 memory

原因：

- 第三阶段不需要开放式多工具规划
- 第三阶段只需要在固定三回流程里做判断
- 确定性流程更容易审计、续跑和复现

### 角色分工

#### 代码控制器负责

- 维护共享根上下文 `H + A`
- 确定当前处理的 `sentence.index + token.index`
- 执行 `query_coarse_units.py`
- 记录当前是第几回搜索
- 限制搜索模式与查询词数量
- 校验 agent 输出是否符合协议
- 在 agent 确认匹配后回填结果

#### 第三阶段 agent 负责

- 阅读当前 token 与整句上下文
- 判断当前候选里是否存在可靠匹配
- 如果当前候选不足以匹配，则给出下一回搜索建议
- 严格按 JSON 协议返回结果

### 核心原则

- 一次只处理一个 token
- 一次对话最多只确认一个 token 的结果
- agent 不能直接访问数据库
- agent 不能编造 coarse_id
- agent 只能在控制器提供的候选中做选择
- agent 不能修改搜索轮次规则

### 当前 token 输入载荷

控制器在第三阶段中，应始终明确当前处理对象。

推荐载荷：

```json
{
  "sentence_index": 12,
  "token_index": 4,
  "sentence_text": "I want to go, but I don't want to wait.",
  "token": {
    "index": 4,
    "text": "wanted to",
    "explanation": "想要，当时想要，表示过去的意图或愿望。",
    "semanticElement": {
      "baseForm": "want",
      "dictionary": "想要；希望；需要。"
    }
  }
}
```

### agent 允许的返回动作

第三阶段 agent 每次只能返回以下三种动作之一，且通过 structured output schema 约束：

1. `match`
2. `search`
3. `no_match`

不允许返回自然语言散文，不允许返回混合格式。

推荐使用代码中的结构化输出 schema，例如：

```python
from typing import Literal, Union
from pydantic import BaseModel, Field

class MatchAction(BaseModel):
    action: Literal["match"]
    coarse_id: int
    reason: str
    explanation: str | None = None

class SearchAction(BaseModel):
    action: Literal["search"]
    mode: Literal["exact", "contain"]
    queries: list[str] = Field(min_length=1, max_length=4)
    reason: str

class NoMatchAction(BaseModel):
    action: Literal["no_match"]
    reason: str

StageThreeOutput = Union[MatchAction, SearchAction, NoMatchAction]
```

调用方式：

```python
structured_llm = llm.with_structured_output(StageThreeOutput)
```

推荐初始化方式：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4-mini",
    reasoning_effort="medium",
)

decision_llm = llm.with_structured_output(StageThreeOutput)
```

说明：

- 这里的 LLM 是“决策器”，不是“执行器”
- 它不直接查询数据库
- 它只根据控制器给出的当前 token 与候选，返回结构化决策

### 1. `match`：当前候选中已能确认匹配

当 agent 认为当前候选中存在可靠匹配时，必须返回：

```json
{
  "action": "match",
  "coarse_id": 166670,
  "reason": "当前 token 在此语境中对应 want 的动词义。",
  "explanation": "想要做某事，表示说话人当时的意图或愿望。"
}
```

字段要求：

- `coarse_id` 必须来自当前候选
- `reason` 用于调试和审计，且必须为非空字符串
- `explanation` 可选；如果返回，就应当是完整优化后的 explanation，而不是建议片段

### 2. `search`：当前候选不足，需要进入下一回搜索

如果当前候选无法确认匹配，但仍值得继续搜索，则返回：

```json
{
  "action": "search",
  "mode": "exact",
  "queries": ["want", "want to", "wanted", "would like"],
  "reason": "当前 token 与 want 系列表达接近，建议继续 exact 搜索。"
}
```

字段要求：

- `mode` 只能是 `exact` 或 `contain`
- `queries` 必须是字符串数组
- `queries` 在发送给脚本前由控制器去重
- `queries` 去重后最多保留 4 个
- `reason` 解释为何建议这些查询词

控制器必须校验：

- 第 2 回时 `mode` 必须是 `exact`
- 第 3 回时 `mode` 必须是 `contain`
- 若不符合当前轮次规则，则视为非法输出

### 3. `no_match`：当前 token 最终判定无法可靠映射

当已经到第 3 回，且仍无法可靠匹配时，agent 返回：

```json
{
  "action": "no_match",
  "reason": "现有候选与当前 token 的语义都不够一致。"
}
```

控制器收到 `no_match` 后：

- 设置 `coarse_id = null`
- 保留第一阶段的 `baseForm`
- 保留第一阶段的 `dictionary`
- 保留或轻微修正第一阶段的 `explanation`
- 将失败原因写入 `semanticElement.reason`，且不能为空

### 第 1 回调用设计

第 1 回由控制器自动执行搜索，不需要 agent 先提 query。

控制器先执行：

- `exact(token.text, baseForm)`

然后第一次调用 agent 时，输入应包含三部分：

1. 共享上下文 `H + A`
2. 当前 token 载荷
3. 第 1 回搜索结果
4. 当前这一轮的动态说明：
   - 这一轮输入里会看到什么
   - 这一轮允许返回什么

推荐结构：

```json
{
  "search_round": 1,
  "search_mode": "exact",
  "queries": ["wanted to", "want"],
  "candidates": {
    "results": []
  }
}
```

第 1 回后，agent 只允许返回：

- `match`
- 或 `search`
- 不允许返回 `no_match`

### 第 2 回调用设计

如果第 1 回没有匹配成功，则 agent 可以返回 `search`，要求控制器执行第 2 回 exact 搜索。

控制器执行完后，在当前 token 分支上下文中追加第 2 回结果，再第二次调用 agent。

新增内容应包含：

```json
{
  "search_round": 2,
  "search_mode": "exact",
  "queries": ["want", "want to", "wanted", "would like"],
  "candidates": {
    "results": []
  }
}
```

第 2 回后，agent 只允许返回：

- `match`
- 或 `search`
- 且此时 `search.mode` 只能是 `contain`
- 不允许返回 `no_match`

### 第 3 回调用设计

如果第 2 回仍未匹配成功，则 agent 可以返回 `search`，要求控制器执行第 3 回 contain 搜索。

控制器执行完后，在当前 token 分支上下文中追加第 3 回结果，再第三次调用 agent。

新增内容应包含：

```json
{
  "search_round": 3,
  "search_mode": "contain",
  "queries": ["want", "want to", "intention", "desire"],
  "candidates": {
    "results": []
  }
}
```

第 3 回后，agent 只允许返回：

- `match`
- 或 `no_match`
- 不允许返回 `search`

### 每轮动态说明

除固定的系统 prompt 和通用规则外，控制器在每一次 LLM 调用时，都必须额外注入“当前这一轮说明”，明确告诉模型：

- 这一轮输入里会看到什么
- 这一轮允许返回什么动作

#### 第 1 回动态说明

- 你会看到当前 token
- 你会看到当前句子
- 你会看到系统已经自动执行的第 1 回搜索结果：`exact(token.text, baseForm)`
- 这一轮允许返回：
  - `match`
  - `search`
- 这一轮不允许返回：
  - `no_match`

#### 第 2 回动态说明

- 你会看到当前 token
- 你会看到当前句子
- 你会看到第 1 回搜索历史
- 你会看到第 2 回搜索结果
- 这一轮允许返回：
  - `match`
  - `search`
- 如果返回 `search`，则 `mode` 只能是 `contain`
- 这一轮不允许返回：
  - `no_match`

#### 第 3 回动态说明

- 你会看到当前 token
- 你会看到当前句子
- 你会看到前两回搜索历史
- 你会看到第 3 回搜索结果
- 这一轮允许返回：
  - `match`
  - `no_match`
- 这一轮不允许返回：
  - `search`

### 当前 token 的完整分支示意

假设当前 token 是 `a`，第二阶段产物为 `A`。

则第三阶段上下文演化为：

- 第 1 回：`H + A + a1`
- 第 2 回：`H + A + a1 + a2`
- 第 3 回：`H + A + a1 + a2 + a3`

其中：

- `a1` 是第 1 回自动 exact 搜索结果
- `a2` 是第 2 回自由 exact 搜索结果
- `a3` 是第 3 回自由 contain 搜索结果

切换到下一个 token `b` 时：

- 丢弃 `a1 + a2 + a3`
- 回到 `H + A`
- 再开始 `H + A + b1 + b2 + b3`

### messages 构造规则

为了保证上下文分支清晰且可缓存，第三阶段每次调用都必须显式构造 `messages`，而不是依赖可变对话对象或隐式记忆。

推荐形式：

```python
messages = [
    *H,
    A_message,
    current_token_message,
    *history_messages,
    current_round_message,
]
```

其中：

- `H` 是固定前缀
- `A_message` 是当前批次完整的第二阶段 JSON
- `current_token_message` 标识当前处理的 `sentence.index + token.index`
- `history_messages` 是当前 token 在此前轮次的搜索历史
- `current_round_message` 是本轮搜索结果

规则：

- 处理同一个 token 时，`history_messages` 可以持续追加
- 切换到下一个 token 时，`history_messages` 必须清空
- 不允许把上一个 token 的历史带到下一个 token
- 不使用隐式 memory 保存这些状态

### 非法输出处理

如果 agent 出现以下情况，控制器应判定为非法输出：

- 输出不是 JSON
- `action` 不在允许集合中
- `match` 时 `coarse_id` 不在当前候选中
- 第 2 回返回了 `contain`
- 第 3 回返回了 `exact`
- 查询词数量超过 4
- 试图同时处理多个 token

处理建议：

- 对当前轮次重试一次 agent 调用
- 若重试后仍非法，则该 token 直接按 `no_match` 处理

### coarse_id 校验规则

当 agent 返回 `action = "match"` 时，控制器必须执行以下检查：

1. 检查 `coarse_id` 是否存在于当前轮数据库搜索结果中
2. 若存在，则视为合法匹配
3. 若不存在，则对当前轮次的 agent 调用重试一次
4. 若重试后返回的 `coarse_id` 仍不在当前数据库结果中，则该 token 直接进入失败流程

失败流程与 `no_match` 一致：

- `semanticElement.coarse_id = null`
- 保留第一阶段的 `baseForm`
- 保留第一阶段的 `dictionary`
- `semanticElement.reason` 写入明确的校验失败原因，且不能为空

### 推荐控制器判定逻辑

1. 控制器执行当前轮次搜索
2. 控制器把当前轮次结果附加到当前 token 分支上下文
3. agent 返回 `match` / `search` / `no_match`
4. 控制器校验返回格式是否合法
5. 若为 `match`，先校验 `coarse_id` 是否真实存在于当前数据库候选中
6. 校验通过后，再由脚本回填 `baseForm` / `dictionary` / `reason` 并结束该 token
7. 若 `coarse_id` 校验失败，则重试一次当前轮次 agent 调用；仍失败则按 `no_match` 处理
8. 若为 `search`，进入下一回
9. 若为 `no_match`，结束该 token 并写入 `coarse_id = null`

### 推荐控制器实现骨架

```python
def process_token(token_runtime, shared_context_A):
    rounds = []

    r1_queries = dedupe([token_runtime.text, token_runtime.base_form])
    r1_candidates = run_query("exact", r1_queries)
    rounds.append({"round_no": 1, "mode": "exact", "queries": r1_queries, "candidates": r1_candidates})
    decision = ask_llm(shared_context_A, token_runtime, rounds)

    if decision.action == "match":
        return finalize_match(decision, r1_candidates, token_runtime, rounds)

    if decision.action == "search":
        r2_candidates = run_query("exact", decision.queries)
        rounds.append({"round_no": 2, "mode": "exact", "queries": decision.queries, "candidates": r2_candidates})
        decision = ask_llm(shared_context_A, token_runtime, rounds)

        if decision.action == "match":
            return finalize_match(decision, r2_candidates, token_runtime, rounds)

        if decision.action == "search":
            r3_candidates = run_query("contain", decision.queries)
            rounds.append({"round_no": 3, "mode": "contain", "queries": decision.queries, "candidates": r3_candidates})
            decision = ask_llm(shared_context_A, token_runtime, rounds)

            if decision.action == "match":
                return finalize_match(decision, r3_candidates, token_runtime, rounds)

    return finalize_no_match(token_runtime, rounds, decision)
```

要求：

- `ask_llm(...)` 内部必须显式构造 `messages`
- `run_query(...)` 必须通过 `query_coarse_units.py` 执行
- `finalize_match(...)` 必须先校验 `coarse_id`
- `finalize_no_match(...)` 必须写入非空 `reason`

---

## 八、最终输出规则

每次批次成功完成后，都要把“当前累计结果”写入输出文件。输出文件中的结构如下：

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "句子原文",
      "explanation": "整句中文解释",
      "tokens": [
        {
          "index": 0,
          "text": "token 文本",
          "explanation": "最终上下文解释",
          "semanticElement": {
            "coarse_id": 166670,
            "baseForm": "want",
            "dictionary": "想要；希望；需要。",
            "reason": "第三阶段判断当前 token 与 want 的动词义匹配，且 coarse_id 已通过数据库候选校验。"
          }
        }
      ]
    }
  ]
}
```

### 约束

- `coarse_id` 推荐使用 JSON `null`，不要使用空字符串
- 输出中不保留数据库候选列表
- 输出中不保留搜索过程日志
- 最终结果只保留必要学习信息
- 输出文件既可能是最终完整结果，也可能是可继续处理的中间结果

---

## 九、推荐执行流程

### 代码流程

1. 读取输入 JSON
2. 按顺序切成每批最多 3 句
3. 初始化累计结果
4. 处理当前批次的清洗输入
5. 调用第一阶段 LLM
6. 对第一阶段输出做第二阶段代码侧顺序校验
7. 若校验失败，则对当前批次重试一次
8. 若重试仍失败，则流程终止
9. 校验通过后，由第二阶段代码为当前批次内每个 token 生成句内 `index`
10. 将第二阶段产出的带 `tokens.index` JSON 作为新输入，开启第三阶段新对话
11. 将第二阶段产物作为第三阶段共享根上下文 `A`
12. 对当前批次内每个 token，从 `A` 开始创建该 token 的分支上下文
13. 对当前 token 执行第 1 回默认自动 exact 搜索
    这一步是一次脚本调用，同时传入 `token.text` 和 `baseForm`
14. 若未匹配成功，则在当前 token 分支上下文中进入第 2 回，由 agent 在 exact 模式下自由尝试最多 4 个查询词
15. 若仍未匹配成功，则在当前 token 分支上下文中进入第 3 回，由 agent 在 contain 模式下自由尝试最多 4 个查询词
16. 将候选交给第三阶段 agent 做语义匹配，并在匹配成功时立即停止该 token 的搜索
17. 若 agent 返回 `match`，先校验 `coarse_id` 是否真实存在于当前数据库候选中；若非法则重试一次，仍非法则按失败流程处理
18. 校验成功后，由脚本回填当前批次的 `coarse_id` / `baseForm` / `dictionary` / `reason` / `explanation`
19. 将当前 token 的搜索审计记录写入目标输出目录下的 `temp/` 子目录
20. 当前 token 完成后，丢弃该 token 的分支上下文，回到共享根上下文 `A`
21. 继续处理当前批次的下一个 token
22. 当前批次全部 token 完成后，将结果合并到累计结果
23. 保存当前累计 JSON 到输出文件
24. 继续下一个批次，直到全部完成

### 推荐查询命令

单个查询：

```bash
python3 supabase/query_coarse_units.py exact apple
```

多个查询：

```bash
python3 supabase/query_coarse_units.py contain apple pear "be addicted to"
```

---

## 十、示例：第三阶段后的结果

```json
{
  "sentences": [
    {
      "index": 0,
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "explanation": "好吧，所以你到底是想跟我谈什么呢？",
      "tokens": [
        {
          "index": 0,
          "text": "Okay,",
          "explanation": "好吧，表示接受对方继续说，或引出接下来的话。",
          "semanticElement": {
            "coarse_id": 92341,
            "baseForm": "okay",
            "dictionary": "好；可以；没问题；表示同意、接受或开始回应。",
            "reason": "第三阶段判断该 token 与 okay 的相关 coarse_unit 语义一致，且 coarse_id 已通过数据库候选校验。"
          }
        },
        {
          "index": 4,
          "text": "wanted to",
          "explanation": "想要，当时想要，表示过去的意图或愿望。",
          "semanticElement": {
            "coarse_id": 166670,
            "baseForm": "want",
            "dictionary": "想要；希望；需要。",
            "reason": "第三阶段判断该 token 在当前语境中对应 want 的动词义，且 coarse_id 已通过数据库候选校验。"
          }
        },
        {
          "index": 7,
          "text": "about?",
          "explanation": "关于……，用于询问谈话的主题或内容。",
          "semanticElement": {
            "coarse_id": null,
            "baseForm": "about",
            "dictionary": "关于；涉及；围绕。",
            "reason": "三回搜索后仍未找到可靠匹配，按失败流程保留第一阶段结果。"
          }
        }
      ]
    }
  ]
}
```

---

## 十一、关键设计原则

- 第一阶段负责“学习导向的语言分片”
- 第二阶段负责“代码校验与 token 编号”
- 第三阶段负责“数据库标准化映射”
- 数据库映射是增强，不是强制
- 无法可靠匹配时，宁可返回 `null`，不要误绑 coarse_id
- 所有可验证约束都优先放在代码里，而不是只依赖 prompt
