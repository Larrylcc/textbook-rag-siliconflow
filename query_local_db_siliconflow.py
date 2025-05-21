import os
from dotenv import load_dotenv
from langchain_chroma import Chroma

# 从 langchain_embed_siliconflow.py 导入自定义的 SiliconFlowEmbeddings 类
# 确保这两个文件在同一目录，或者 langchain_embed_siliconflow.py 在 Python 的搜索路径中
from langchain_embed_siliconflow import SiliconFlowEmbeddings

# 1. 加载环境变量
load_dotenv()
silicon_api_key = os.getenv("SILICON_API_KEY")

if not silicon_api_key:
    raise ValueError("在环境变量中未找到 SILICON_API_KEY。请在您的 .env 文件中设置它。")

# 2. 定义 Chroma 数据库存储的目录和集合名称
# 这些必须与 langchain_embed_siliconflow.py 中用于创建数据库的名称相匹配
persist_directory = "./local_pdf_chroma_db_sf"
collection_name = "sf_pdf_documents_collection"

# 3. 初始化 SiliconFlow Embeddings 模型
# 使用与创建数据库时相同的模型和参数
embeddings = SiliconFlowEmbeddings(
    api_key=silicon_api_key,
    model_name="BAAI/bge-large-zh-v1.5",  # 确保与创建时使用的模型一致
    # api_base_url 和 batch_size 等其他参数使用默认值即可，因为查询时通常是单个文本
)

# 4. 加载现有的 Chroma 数据库
print(
    f"正在从以下位置加载 Chroma 数据库: {persist_directory}，集合名称: {collection_name} (使用 SiliconFlow 嵌入)"
)
try:
    if not os.path.exists(persist_directory):
        print(f"错误: 持久化目录 '{persist_directory}' 未找到。")
        print(
            "请确保您已经运行了 langchain_embed_siliconflow.py 来创建数据库，或者路径正确。"
        )
        exit()

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,  # 关键：使用相同的 embedding 函数实例
    )
    print("成功加载 Chroma 数据库。")
except Exception as e:
    print(f"加载 Chroma 数据库时出错: {e}")
    print("请确保 persist_directory 路径正确且数据库文件完好无损。")
    print(
        "同时验证 collection_name 是否与创建时使用的名称匹配，以及 embedding_function 是否正确初始化。"
    )
    exit()

# 5. 执行相似性搜索 (示例)
# query = "什么是软件设计模式？" # 将此更改为您的查询
# query = "解释一下什么是回归测试"
query = "解释面向对象设计"


print(f"\n正在为以下内容执行相似性搜索: '{query}'")

try:
    # k 是要检索的相似文档的数量
    retrieved_docs = vector_store.similarity_search(query, k=5)  # 检索5个最相关的文档

    if retrieved_docs:
        print(f"\n找到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- 文档 {i+1}  ---")

            print(f"内容: {doc.page_content[:800]}...")  # 打印前800个字符
            if doc.metadata and "source" in doc.metadata:
                print(f"来源: {doc.metadata['source']}")
            if doc.metadata and "page" in doc.metadata:
                print(f"页码: {doc.metadata['page']}")
    else:
        print("未找到与您的查询相关的文档。")

except Exception as e:
    print(f"相似性搜索期间出错: {e}")
