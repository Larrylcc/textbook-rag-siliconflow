# 教材智能问答项目教程 (基于 SiliconFlow & ChromaDB)

## 1. 项目简介

本项目旨在通过 SiliconFlow 提供的免费文本嵌入模型 `BAAI/bge-large-zh-v1.5` 对预处理的软件工程教材（或其他文档）的向量数据进行相似性搜索。用户可以通过输入查询语句，项目会调用 SiliconFlow API 对查询进行嵌入，并在本地的 ChromaDB 向量数据库中执行相似性搜索，从而实现一个基础的、针对特定教材的检索增强生成 (RAG) 功能的后端支持。

本教程主要介绍如何使用**已有的**教材向量数据库进行查询。关于如何自行处理和嵌入新的 PDF 文档，请参见教程末尾的“扩展”部分。

## 2. 功能特性

* **相似性搜索**: 对用户输入的查询进行嵌入，并在预先构建好的本地 ChromaDB 向量数据库中查找最相关的文本块。
* **结果提取**:方便地获取检索到的文本内容及其元数据（如来源、页码），以便进一步使用。

（对于自行处理文档的用户，还包括 PDF 文档处理、文本切分、文本嵌入、向量存储和页码修正等功能，详见扩展部分。）

## 3. 技术栈

* **Python**: 主要编程语言。
* **Langchain**: 用于构建 LLM 应用的框架，此处用于与 ChromaDB 的集成和查询。
* **SiliconFlow API**: 提供文本嵌入服务。
* **ChromaDB**: 本地运行的向量数据库，用于存储和检索文本向量。
* **Requests**: 用于与 SiliconFlow API 进行 HTTP 通信。
* **Dotenv**: 用于管理环境变量（如 API 密钥）。

## 4. 环境准备

### 4.0 克隆仓库

将此 GitHub 仓库克隆到本地

```bash
git clone https://github.com/Larrylcc/textbook-rag-siliconflow.git
```

移动到本地仓库文件夹中

```bash
cd textbook-rag-siliconflow
```

### 4.1 Python 环境

确保您的系统中已安装 Python (推荐 3.8 或更高版本)。您可以从 [Python 官网](https://www.python.org/) 下载并安装。

建议使用虚拟环境来管理项目依赖，当然也可以使用 conda 管理 python 环境。以下是创建和激活 venv 的流程：

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

以下是创建和激活conda环境的流程：

```bash
conda create --yes --name rag python=3.12
conda activate rag
```

### 4.2 SiliconFlow API Key

1. 访问 [SiliconFlow 官网](https://www.siliconflow.cn/) 并注册账户。
2. 登录后，在您的账户信息或 API 管理页面获取 API Key。这个 Key 将用于访问嵌入模型（即使是查询也需要对查询本身进行嵌入）。
3. 此处嵌入模型的访问是完全免费的。

### 4.3 安装依赖包

项目所需的 Python 包已在 `requirements.txt` 文件中列出。在激活虚拟环境后，使用 pip 安装它们：

```bash
pip install -r requirements.txt
```

主要依赖包括：

* `python-dotenv`: 用于加载 `.env` 文件中的环境变量。
* `langchain-chroma`: Langchain 与 ChromaDB 的集成。
* `langchain-community`: Langchain 社区组件。
* `langchain-text-splitters`: Langchain 文本切分工具 (主要用于扩展部分)。
* `langchain-core`: Langchain 核心库。
* `requests`: 用于发送 HTTP 请求到 SiliconFlow API。
* `pypdf`: `PyPDFLoader` 需要的 PDF 处理库 (主要用于扩展部分)。

## 5. 项目配置

### 5.1 创建 `.env` 文件

在项目的根目录下创建一个名为 `.env` 的文件。此文件用于存储您的 SiliconFlow API Key。

文件内容如下：

```env
# .env
SILICON_API_KEY="YOUR_SILICONFLOW_API_KEY_HERE"
```

将 `YOUR_SILICONFLOW_API_KEY_HERE` 替换为您从 SiliconFlow 获取的真实 API Key。

### 5.2 获取并放置预处理的教材数据库

本项目的主要使用方式是基于一个已经处理和嵌入好的教材向量数据库。

项目中已包含名为 `local_pdf_chroma_db_sf` 的文件夹。这个文件夹包含了所有预处理好的教材向量数据。

## 6. 运行步骤：使用现有数据库进行查询

在完成环境准备和项目配置后，您可以运行查询脚本来对已嵌入的教材内容进行相似性搜索。

### 6.1 修改查询内容

打开 `query_local_db_siliconflow.py` 文件。您会看到一个 `query` 变量，用于定义您想搜索的问题。

找到类似以下的行：

```python
# query = "什么是软件设计模式？" # 将此更改为您的查询
# query = "解释一下什么是回归测试"
query = "解释面向对象设计"
```

将 `query` 变量的值修改为您想要提出的具体问题。例如，如果您想问关于敏捷开发的问题，可以修改为：

```python
query = "请解释什么是敏捷开发？"
```

保存文件。

### 6.2 执行查询脚本

在您的终端中，确保虚拟环境已激活，然后运行查询脚本：

```bash
python query_local_db_siliconflow.py
```

**脚本执行流程：**

1. 加载 `.env` 文件中的 `SILICON_API_KEY`。
2. 初始化 SiliconFlow 嵌入模型 (`BAAI/bge-large-zh-v1.5`)。
3. 加载位于 `./local_pdf_chroma_db_sf` 目录中名为 `sf_pdf_documents_collection` 的 ChromaDB 数据库。
4. 脚本会将您在 `query` 变量中定义的查询通过 SiliconFlow API 转换为向量。
5. 使用查询向量在 ChromaDB 中执行相似性搜索，默认检索 `k=5` 个最相关的文档块。
6. 打印出检索到的文档内容片段、来源文件和页码。

### 6.3 获取并使用检索结果

脚本执行相似性搜索后，相关的文档块会存储在 `retrieved_docs` 变量中。

在 `query_local_db_siliconflow.py` 中，这部分代码如下：

```python
# ... existing code ...
try:
    # k 是要检索的相似文档的数量
    retrieved_docs = vector_store.similarity_search(query, k=5)  # 检索5个最相关的文档

    if retrieved_docs:
        print(f"\n找到 {len(retrieved_docs)} 个相关文档:")
        # ... (代码继续打印文档) ...
# ... existing code ...
```

`retrieved_docs` 是一个包含 Langchain `Document` 对象的列表。每个 `Document` 对象通常包含：

* `page_content`: 检索到的文本块内容。
* `metadata`: 一个字典，包含该文本块的元数据，如 `source` (来源文件名) 和 `page` (在原始文档中的页码)。

您可以直接在脚本中进一步处理这个 `retrieved_docs` 列表。例如：

* 将其内容传递给一个大型语言模型 (LLM) 作为生成回答的上下文。
* 将检索到的信息汇总并保存到文件中。
* 在其他 Python 模块或函数中导入并使用这些结果。

脚本默认会打印这些文档的摘要信息。如果您想在程序中直接使用这些数据，`retrieved_docs` 变量就是您所需要的。

## 7. 文件结构说明 (主要使用场景)

```
.
├── query_local_db_siliconflow.py   # 脚本：从ChromaDB加载并执行相似性搜索
├── langchain_embed_siliconflow.py  # 脚本：(用于扩展功能) 加载PDF，嵌入并存入ChromaDB
├── requirements.txt                # Python 依赖包列表
├── .env                            # 环境变量文件 (需自行创建)
└── local_pdf_chroma_db_sf/         # 文件夹：存放预处理的 ChromaDB 教材数据库 (需自行获取并放置)
    └── ... (ChromaDB 文件)
```

(如果您要使用扩展功能自行处理PDF，还会有一个 `textbook/` 文件夹。)

## 8. 自定义与调整 (查询部分)

### 8.1 ChromaDB 路径与集合名称

* 如果您获取的预处理数据库 `local_pdf_chroma_db_sf` 文件夹被放置在不同路径，或者其内部的集合名称不是默认的 `sf_pdf_documents_collection`，您需要在 `query_local_db_siliconflow.py` 中更新以下变量的值：
  * `persist_directory`: ChromaDB 存储目录。
  * `collection_name`: ChromaDB 中的集合名称。
    确保这些值与您的数据库实际情况匹配。

### 8.2 嵌入模型

* `query_local_db_siliconflow.py` 脚本中初始化的 `SiliconFlowEmbeddings` 类必须使用与创建数据库时相同的嵌入模型。对于本项目提供的预处理数据库，默认使用的是 `BAAI/bge-large-zh-v1.5`。如果您使用了不同模型创建的数据库，请确保在此处也进行相应修改。

### 8.3 检索数量

* 在 `query_local_db_siliconflow.py` 中，`vector_store.similarity_search(query, k=5)` 这行代码中的 `k=5` 表示检索5个最相关的文档。您可以根据需要调整 `k` 的值。

## 9. 扩展：自行处理和嵌入新的 PDF 文档

如果您希望处理自己的 PDF 文档（例如，不同的教材、新的文档版本），或者从头开始创建/更新教材向量数据库，可以按照以下步骤操作。

### 9.1 准备 PDF 教材

1. 在项目根目录下创建一个名为 `textbook` 的文件夹 (如果脚本中 `pdf_files_directory` 指向其他位置，请创建对应文件夹)。
2. 将您希望处理的 PDF 文件放入 `textbook` 文件夹中。脚本 `langchain_embed_siliconflow.py` 会从此目录加载 PDF。

### 9.2 运行嵌入脚本

此步骤会加载 `textbook` 文件夹中的 PDF，进行切分和嵌入，然后将结果存储到本地 ChromaDB。

执行脚本：

```bash
python langchain_embed_siliconflow.py
```

**脚本执行流程：**

1. 加载 `.env` 文件中的 `SILICON_API_KEY`。
2. 初始化 SiliconFlow 嵌入模型 (`BAAI/bge-large-zh-v1.5`)。
3. 从 `./textbook` 目录加载 PDF 文档。
4. 将加载的文档内容切分成小块。
5. （可选）根据 `PAGE_OFFSET_N` 变量修正文档块的页码元数据。
6. 调用 SiliconFlow API 将文本块批量嵌入为向量。
7. 将嵌入向量和元数据存储到 `./local_pdf_chroma_db_sf` 目录下的 ChromaDB 数据库中，集合名称为 `sf_pdf_documents_collection`。也可自行修改，注意，创建和查询时使用的名称需相同。
8. 成功后，您会看到提示信息，表明数据库已创建并持久化。

**注意：**

* 首次运行此脚本时，如果 PDF 文件较多或较大，嵌入过程可能需要一些时间。
* 如果 `./local_pdf_chroma_db_sf` 目录已存在并且包含同名集合，再次运行可能会覆盖或追加数据，具体行为取决于 ChromaDB 的配置（本项目脚本默认会重新创建或使用现有集合）。

### 9.3 自定义嵌入过程

在 `langchain_embed_siliconflow.py` 脚本中，您可以调整以下参数：

* **PDF 文件目录**: 修改 `pdf_files_directory` 变量的值可以更改 PDF 文件的来源目录。

  ```python
  # langchain_embed_siliconflow.py
  pdf_files_directory = "./textbook" # 修改这里
  ```
* **ChromaDB 路径与集合名称 (创建时)**:

  * `persist_directory`: ChromaDB 存储目录。
  * `collection_name`: ChromaDB 中的集合名称。
    如果您修改这些，请确保在查询脚本中也使用相同的值。
* **页码偏移量 `PAGE_OFFSET_N`**:
  用于修正 PDF 的物理页码与 PDF 阅读器显示页码之间的偏移。

  ```python
  # langchain_embed_siliconflow.py
  PAGE_OFFSET_N = 17  #  <-- 根据您的PDF具体情况调整此值
  ```
* **嵌入模型和参数 (创建时)**:
  脚本中指定了嵌入模型 `model_name="BAAI/bge-large-zh-v1.5"` 和 `batch_size`。如果您想更换模型或调整批处理大小，可以在 `SiliconFlowEmbeddings` 初始化部分进行修改。

## 10. 注意事项

* **API Key 安全**: 不要将您的 `.env` 文件或包含 API Key 的任何文件提交到公共代码仓库（如 GitHub）。确保将其添加到 `.gitignore` 文件中。
* **SiliconFlow API 限制**: 免费 API Key 可能存在速率限制或使用量限制。如果遇到频繁的 API 错误，请检查 SiliconFlow 的文档或联系其支持。
* **网络连接**: 嵌入过程和查询过程都需要稳定的网络连接来访问 SiliconFlow API。
* **错误处理**: 脚本中包含了一些基本的错误处理和打印信息，有助于调试。如果遇到问题，请仔细阅读控制台输出。

希望本教程能帮助您顺利运行和使用此项目！
