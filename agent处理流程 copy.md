输入两个 json 文件路径,第一个是文件输入路径,第二个是输出保存路径

1.首先数据清理:
原始数据结构为:

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
        },
        {
          "index": 3,
          "text": "is",
          "start": 15109,
          "end": 15205
        },
        {
          "index": 4,
          "text": "it",
          "start": 15205,
          "end": 15350
        },
        {
          "index": 5,
          "text": "that",
          "start": 15350,
          "end": 15510
        },
        {
          "index": 6,
          "text": "you",
          "start": 15510,
          "end": 15558
        },
        {
          "index": 7,
          "text": "wanted",
          "start": 15671,
          "end": 15863
        },
        {
          "index": 8,
          "text": "to",
          "start": 15863,
          "end": 15928
        },
        {
          "index": 9,
          "text": "talk",
          "start": 15928,
          "end": 16088
        },
        {
          "index": 10,
          "text": "to",
          "start": 16088,
          "end": 16169
        },
        {
          "index": 11,
          "text": "me",
          "start": 16169,
          "end": 16265
        },
        {
          "index": 12,
          "text": "about?",
          "start": 16265,
          "end": 16441
        }
      ]
    }
  ]
}
```

清洗为以下结构, 由于 llm 输出的 index 实际并不可信,所以只输入和输出句子 index. 并尽可能压缩来减少 context 消耗.

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

设计一个 agent,使用这套模型, OPENAI_API_KEY从.env读取

```py
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-5.4",
    reasoning_effort="medium",
)
```

发送给 agent 做第一步,分词: 只可以组合 tokens,

prompt:

请将英文文本进行结构化分析, 组合tokens和解释 。请严格遵循以下指示：

1.  按顺序为每个句子提供一个整体的中文翻译或解释（explanation）。
2.  按顺序将以tokens为最基础单元, 进一步合并为有意义的语言元素分片。可以不合并而保持单个单词，对于简单常用的单词，也可合并短语固定搭配。
3.  **以英语学习为目的组合**：对于难度稍高的词组（如"be addicted to"），应该合并为一个token，在explanation中同时解释短语含义和核心单词，semanticElement.baseForm应为核心词的原形（如"addicted"）。

正面例子（应该这样合并）：

- "by the way" -> 保持为一个token（固定短语）
- "deal with" -> 保持为一个token（固定搭配）
- "at the same time" -> 保持为一个token（固定短语）
- "looking forward to" -> 保持为一个token（固定搭配）
- "be addicted to" -> 保持为一个token，explanation: "对...上瘾；addicted表示沉迷的、上瘾的"，baseForm: "addicted"
- "get rid of" -> 保持为一个token，explanation: "摆脱、除去；rid表示使摆脱"，baseForm: "rid"

反面例子（不该这样合并）：

- "I am happy" -> 非固定搭配或短语, 不要合并"I am"作为一个token，应该分别为"I"、"am"、"happy"三个tokens
- "the book" -> 非固定搭配或短语, 不要合并为一个token，应该分别为"the"、"book"两个token
- "very good" -> 不要合并为一个token，应该分别为"very"、"good"两个token

4.  为每个分片（token）提供符合上下文语境的中文解释（explanation）。
5.  为每个token生成一个semanticElement对象，包含：- baseForm: 单词的原形（如running -> run，studies -> study，is addicted to -> addicted）- dictionary: 无上下文语境的词典释义
6.  严格按照我提供的 JSON schema 格式，输出一个包含句子列表的 JSON 对象。

```json
{
  "sentences": [
    {
      "text": "句子的完整英文原文",
      "explanation": "整句中文解释，可为空",
      "tokens": [
        {
          "text": "token 文本",
          "explanation": "该 token 根据上下文语境给出的中文解释",
          "semanticElement": {
            "baseForm": "英文原形",
            "dictionary": "无上下文语境的中文词典释义"
          }
        }
      ]
    }
  ]
}
```

结果:

```json
{
  "sentences": [
    {
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
          "explanation": "想要，当时想要，表示过去的意图或愿望。want to 是常见搭配表示想要做",
          "semanticElement": {
            "baseForm": "want",
            "dictionary": "想要；希望；需要。"
          }
        },
        {
          "text": "talk to",
          "explanation": "和……谈、跟……说话；talk to 是常见搭配，表示与某人交谈。",
          "semanticElement": {
            "baseForm": "talk",
            "dictionary": "说话；谈话；交谈。talk to 表示与某人交谈。"
          }
        },
        {
          "text": "me",
          "explanation": "我，作宾语，表示谈话的对象是“我”。",
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

第二步,数据库映射, 使用 agent, 脚本supabase/query_coarse_units.py提供两种查询接口,
对每个合并完的tokens进行精确数据库映射, 让 agent 自行决定如何搜索, 搜到匹配到结果或者自行决定停止,
如果搜到则根据查询到的项目进行意思匹配, 如果成功匹配则让 token的coarse_id为匹配到的 id, 并根据该条目给出的信息优化explanation(包含上下文的解释),和使用chinese_def替换dictionary(从数据库直接拿到的真正字典释义)
如果搜索到但是没有项目能匹配, 或者没搜到结果,则让token的coarse_id为空,其他的也不变.

最终结果类似:

```json
{
  "sentences": [
    {
      "text": "Okay, so what is it that you wanted to talk to me about?",
      "explanation": "好吧，所以你到底是想跟我谈什么呢？",
      "tokens": [
        {
          "text": "Okay,",
          "explanation": "好吧，表示接受对方继续说，或引出接下来的话。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "okay",
            "dictionary": "好；可以；没问题；表示同意、接受或开始回应。"
          }
        },
        {
          "text": "so",
          "explanation": "那么，所以，这里用于承接前文并引出问题。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "so",
            "dictionary": "所以；那么；因此；用于连接上下文或推进话题。"
          }
        },
        {
          "text": "what is it that",
          "explanation": "到底是……，这是一个强调式提问结构，用来突出真正想问的内容。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "what",
            "dictionary": "什么；用于提问事物或内容。"
          }
        },
        {
          "text": "you",
          "explanation": "你。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "you",
            "dictionary": "你；你们。"
          }
        },
        {
          "text": "wanted to",
          "explanation": "想要，当时想要，表示过去的意图或愿望。want to 是常见搭配表示想要做",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "want",
            "dictionary": "想要；希望；需要。"
          }
        },
        {
          "text": "talk to",
          "explanation": "和……谈、跟……说话；talk to 是常见搭配，表示与某人交谈。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "talk",
            "dictionary": "说话；谈话；交谈。talk to 表示与某人交谈。"
          }
        },
        {
          "text": "me",
          "explanation": "我，作宾语，表示谈话的对象是“我”。",
          "semanticElement": {
            "coarse_id": "123213",
            "baseForm": "me",
            "dictionary": "我；I 的宾格形式。"
          }
        },
        {
          "text": "about?",
          "explanation": "关于……，用于询问谈话的主题或内容。",
          "semanticElement": {
            "coarse_id": "",
            "baseForm": "about",
            "dictionary": "关于；涉及；围绕。"
          }
        }
      ]
    }
  ]
}
```

最后保存最终结果
