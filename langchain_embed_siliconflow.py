import os
import time  # 用于潜在的速率限制
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from typing import List
import requests
import json


# --- 自定义 SiliconFlow 嵌入类 ---
class SiliconFlowEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        api_base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 32,  # 根据 API 限制或性能进行调整
        request_timeout: int = 60,  # API 请求超时时间（秒）
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"{api_base_url}/embeddings"
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """嵌入单批文本。"""
        payload = {
            "model": self.model_name,
            "input": texts,  # API 文档建议 'input' 可以是字符串列表以进行批处理
            "encoding_format": "float",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()  # 针对 HTTP 错误引发异常
            response_data = response.json()

            if "data" in response_data and isinstance(response_data["data"], list):
                # 确保嵌入向量的顺序与输入文本的顺序相同
                # API 应该按顺序返回它们，但最好注意一下
                # 对于 BGE 模型，-large 版本的维度通常是 1024，-base 版本是 768
                # 如果可能，从第一个嵌入向量获取维度更安全，或者假设一个值
                # 对于 BAAI/bge-large-zh-v1.5，维度是 1024
                embeddings = [item["embedding"] for item in response_data["data"]]
                if len(embeddings) == len(texts):
                    return embeddings
                else:
                    print(
                        f"警告：收到的嵌入向量数量 ({len(embeddings)}) 与发送的文本数量 ({len(texts)}) 不匹配。"
                    )
                    # 后备方案：返回空嵌入向量或引发错误
                    return [[0.0] * 1024 for _ in texts]  # 占位符，调整维度
            else:
                print(f"错误：SiliconFlow API 返回了意外的响应格式：{response_data}")
                return [[0.0] * 1024 for _ in texts]  # 占位符

        except requests.exceptions.HTTPError as http_err:
            print(f"嵌入时发生 HTTP 错误：{http_err}")
            print(f"响应内容：{response.content.decode()}")
            return [[0.0] * 1024 for _ in texts]  # 占位符
        except requests.exceptions.Timeout:
            print(f"嵌入批处理时请求超时。")
            return [[0.0] * 1024 for _ in texts]  # 占位符
        except Exception as e:
            print(f"嵌入批处理时发生错误：{e}")
            return [[0.0] * 1024 for _ in texts]  # 占位符

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(
                f"正在嵌入批次 {i // self.batch_size + 1}/{(len(texts) -1) // self.batch_size + 1}，大小：{len(batch)}"
            )
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            # 可选：添加一个小延迟以避免达到速率限制（如果有）
            # time.sleep(0.1)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # 对于单个查询，API 期望 'input' 是一个字符串，而不是列表。
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            if (
                "data" in response_data
                and isinstance(response_data["data"], list)
                and len(response_data["data"]) > 0
            ):
                return response_data["data"][0]["embedding"]
            else:
                print(f"错误：查询的响应格式意外：{response_data}")
                return [0.0] * 1024  # 占位符，调整维度
        except requests.exceptions.HTTPError as http_err:
            print(f"嵌入查询时发生 HTTP 错误：{http_err}")
            print(f"响应内容：{response.content.decode()}")
            return [0.0] * 1024  # 占位符
        except requests.exceptions.Timeout:
            print(f"嵌入查询时请求超时。")
            return [0.0] * 1024  # 占位符
        except Exception as e:
            print(f"嵌入查询时发生错误：{e}")
            return [0.0] * 1024  # 占位符


# --- 主脚本逻辑 ---
# 这部分仅在直接执行脚本时运行
if __name__ == "__main__":
    load_dotenv()
    silicon_api_key = os.getenv("SILICON_API_KEY")
    silicon_base_url = "https://api.siliconflow.cn/v1"

    if not silicon_api_key:
        raise ValueError(
            "在环境变量中未找到 SILICON_API_KEY。请在您的 .env 文件中设置它。"
        )

    pdf_files_directory = "./textbook"
    persist_directory = "./local_pdf_chroma_db_sf"  # 已更改以避免冲突
    collection_name = "sf_pdf_documents_collection"  # 已更改以避免冲突

    # 使用自定义的 SiliconFlowEmbeddings 类
    embeddings = SiliconFlowEmbeddings(
        api_key=silicon_api_key,
        api_base_url=silicon_base_url,
        model_name="BAAI/bge-large-zh-v1.5",
        batch_size=16,  # SiliconFlow 文档提到每个请求的 token 限制为 2048，
        # 并且输入数组长度最大为 256。批处理文本更安全。
        # 16-32 个文本的 batch_size 应该是合理的。
    )

    print(f"正在目录中查找 .pdf 文件: {pdf_files_directory}")

    if not os.path.isdir(pdf_files_directory):
        print(f"错误：目录 {pdf_files_directory} 未找到")
        print("请创建此目录并将您的 .pdf 文件放入其中，或更正路径。")
    else:
        loader = DirectoryLoader(
            pdf_files_directory,
            glob="**/*.pdf",
            show_progress=True,
            loader_cls=PyPDFLoader,
        )

        print(f"正在从 '{pdf_files_directory}' 加载 PDF 文档...")
        documents = loader.load()

        if not documents:
            print(
                f"在 {pdf_files_directory} 中未找到 .pdf 文件。请确保文件存在且 glob 模式正确。"
            )
        else:
            print(f"已加载 {len(documents)} 个 PDF 文档。")

            # 过滤掉任何可能是 None 或没有 page_content 的文档
            valid_documents = [
                doc
                for doc in documents
                if doc and hasattr(doc, "page_content") and doc.page_content.strip()
            ]
            if len(valid_documents) != len(documents):
                print(
                    f"警告: 移除了 {len(documents) - len(valid_documents)} 个无效或空的文档。"
                )

            if not valid_documents:
                print("没有有效的文档内容可供处理。")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # 减小 chunk_size 以更好地适应批处理中每个项目的 token 限制
                    chunk_overlap=100,
                )
                all_splits = text_splitter.split_documents(valid_documents)
                # 进一步过滤：确保没有拆分仅为空白
                all_splits = [
                    split for split in all_splits if split.page_content.strip()
                ]

                if not all_splits:
                    print("分割后没有有效的文本块可供嵌入。")
                else:
                    print(f"已将 PDF 文档分割成 {len(all_splits)} 个有效文本块。")

                    # --- 开始页码修正 ---
                    # N 是 PDF 中打印为“第 1 页”之前的页数。
                    # 根据示例（物理页 259 显示为 PDF 页 275），N = 17。
                    # P_pdf_0_indexed = P_physical_1_indexed + N - 1
                    # 我们想要存储 P_physical_1_indexed。
                    # P_physical_1_indexed = P_pdf_0_indexed - N + 1
                    PAGE_OFFSET_N = 17  #  <-- 重要：此值根据您的PDF确定

                    print(f"正在根据偏移量 {PAGE_OFFSET_N} 修正页码...")
                    for split_doc in all_splits:
                        if split_doc.metadata and "page" in split_doc.metadata:
                            original_pdf_page_0_indexed = split_doc.metadata["page"]
                            # 计算实际的1索引物理页码
                            physical_page_1_indexed = (
                                original_pdf_page_0_indexed - PAGE_OFFSET_N + 1
                            )
                            split_doc.metadata["page"] = physical_page_1_indexed

                            # 可选：为调试目的打印一些转换示例
                            # if original_pdf_page_0_indexed < 5 or \
                            #    (original_pdf_page_0_indexed >= 270 and original_pdf_page_0_indexed <= 280) : # 检查特定范围
                            #    print(f"  文档来源: {split_doc.metadata.get('source', 'N/A')}, "
                            #          f"原始PDF页(0索引): {original_pdf_page_0_indexed} -> "
                            #          f"修正后物理页(1索引): {physical_page_1_indexed}")
                    print("页码修正完成。")
                    # --- 结束页码修正 ---

                    print(
                        f"正在初始化 Chroma 数据库于: {persist_directory}，集合名称: {collection_name} (使用自定义硅基流动嵌入)"
                    )

                    # 确保 all_splits 包含带有 page_content 的 Document 对象
                    # 自定义嵌入器期望 List[str]，Chroma.from_documents 期望 List[Document]
                    # 如果我们传递 List[Document]，自定义类的 embed_documents 方法将处理提取 page_content
                    # 然而，Chroma.from_documents 将调用 embeddings.embed_documents([doc.page_content for doc in documents])
                    # 因此我们的 SiliconFlowEmbeddings.embed_documents 需要正确处理字符串列表。

                    vector_store = Chroma.from_documents(
                        documents=all_splits,  # Document 对象列表
                        embedding=embeddings,  # 我们的自定义嵌入类
                        collection_name=collection_name,
                        persist_directory=persist_directory,
                    )

                    print(
                        f"已成功在 '{persist_directory}' 创建并持久化 Chroma 数据库 (集合为 '{collection_name}')，使用硅基流动嵌入模型处理 PDF。"
                    )
                    print("您现在可以打包此目录以进行共享。")
