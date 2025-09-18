# Simple Audio Processing Pipeline

一个完整的音频处理流水线，用于将音频文件转录为文本并进行智能分析。该项目使用 Whisper 进行语音识别，然后通过 Gemini API 对转录内容进行深度分析和词汇解释。

## 功能特性

- 🎵 **多模型语音识别**: 支持多种 Whisper 模型 (WhisperX, OpenAI Whisper, Whisper Diarization 等)
- 🧹 **智能数据清理**: 自动清理和结构化转录数据
- 🤖 **AI 文本分析**: 使用 Gemini API 进行句子分析和词汇解释
- 🔄 **并行处理**: 支持批量并行处理提高效率
- 🔐 **安全配置**: API 密钥通过环境变量管理
- 📊 **完整流水线**: 从音频文件到结构化分析结果的一站式处理

## 项目结构

```
.
├── main.py                    # 主流水线脚本
├── 1whisper.py               # Whisper 语音识别模块
├── 2data-cleansing.py        # 数据清理模块
├── 3llm.py                   # LLM 分析模块
├── gemini.py                 # Gemini API 封装
├── models_config.py          # 模型配置文件
├── .env                      # 环境变量配置 (需要创建)
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖列表
├── .gitignore                # Git 忽略文件配置
└── README.md                 # 项目文档
```

### 输出目录结构

```
├── 原始媒体/                  # 音频文件存放目录
├── 1transcript-raw/          # Whisper 原始转录结果
├── 2cleaned-data/            # 清理后的结构化数据
└── 3llm/                     # LLM 分析最终结果
```

## 安装和设置

### 1. 克隆项目

```bash
git clone https://github.com/acquisit-ai/simple-audio-processing.git
cd simple-audio-processing
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install replicate>=0.15.0 google-genai>=0.6.0 python-dotenv>=1.0.0 pydantic>=2.0.0
```

#### 依赖说明

- **replicate**: Replicate API 客户端，用于调用 Whisper 模型
- **google-genai**: Google Gemini API 客户端，用于文本分析
- **python-dotenv**: 环境变量管理，用于安全存储 API 密钥
- **pydantic**: 数据验证和序列化，用于结构化 API 响应

### 4. 配置 API 密钥

复制环境变量模板并填入你的 API 密钥：

编辑 `.env` 文件：

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Replicate API Configuration
REPLICATE_API_TOKEN=your_replicate_token_here
```

#### 获取 API 密钥

- **Gemini API**: 访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取
- **Replicate API**: 访问 [Replicate](https://replicate.com/account/api-tokens) 获取

### 5. 准备音频文件

将要处理的音频文件放入 `原始媒体/` 目录中。支持的格式：

- MP3, MP4, WAV, AVI, MOV 等

## 使用方法

### 快速开始

1. 将音频文件放入 `原始媒体/` 目录
2. 运行主流水线：

```bash
python main.py
```

### 分步执行

你也可以单独运行各个模块：

#### 1. 语音识别

```bash
python 1whisper.py
```

#### 2. 数据清理

```bash
python 2data-cleansing.py
```

#### 3. LLM 分析

```bash
python 3llm.py
```

### 自定义配置

#### 选择 Whisper 模型

编辑 `1whisper.py` 中的 `USE_MODEL` 参数：

```python
USE_MODEL = 1  # 1=WhisperX, 2=OpenAI Whisper, 3=Whisper Diarization, etc.
```

#### 调整批处理大小

编辑 `3llm.py` 中的配置：

```python
DEFAULT_BATCH_SIZE = 10      # 每批次处理的句子数量
DEFAULT_MAX_WORKERS = 5      # 最大并行线程数
```

## 处理流程

```mermaid
graph LR
    A[音频文件] --> B[Whisper 转录]
    B --> C[数据清理]
    C --> D[句子提取]
    D --> E[Gemini 分析]
    E --> F[结构化结果]
```

### 详细步骤

1. **语音识别**: 使用 Whisper 模型将音频转换为带时间戳的文本
2. **数据清理**: 清理转录结果，去除无效数据，结构化输出
3. **句子提取**: 从清理后的数据中提取句子内容
4. **智能分析**: 使用 Gemini API 对英文句子进行：
   - 整句翻译/解释
   - 词汇分解和解释
   - 语法结构分析

## 输出格式

最终输出的 JSON 文件包含详细的分析结果：

```json
{
  "language": "en",
  "total_sentences": 3,
  "sentences": [
    {
      "index": 0,
      "text": "Number one most racist country in Europe...",
      "explanation": "欧洲最种族主义的国家排名第一...",
      "tokens": [
        {
          "index": 0,
          "text": "Number one",
          "explanation": "排名第一"
        }
      ]
    }
  ]
}
```

## 常见问题

### Q: 支持哪些音频格式？

A: 支持 MP3, MP4, WAV, AVI, MOV 等常见格式。

### Q: 如何更改处理的音频文件？

A: 修改 `main.py` 中的 `audio_file` 变量路径。

### Q: API 调用失败怎么办？

A: 检查：

1. API 密钥是否正确设置
2. 网络连接是否正常
3. API 服务是否正常

### Q: 如何提高处理速度？

A: 可以调整 `3llm.py` 中的并行参数：

- 增加 `DEFAULT_MAX_WORKERS`
- 调整 `DEFAULT_BATCH_SIZE`

## 贡献

欢迎提交 Pull Request 或创建 Issue 来改进项目。

## 许可证

MIT License

## 联系方式

如有问题请创建 GitHub Issue。
