# 完整RAG系统构建示例
# 确保已安装: pip install langchain langchain-openai langchain-chroma langchain-community python-dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings

# 1. 加载和预处理文档
print("步骤1: 加载和预处理文档...")
loader = TextLoader("./data/deepseek百度百科.txt")  # 你的文档文件
documents = loader.load()

# 2. 文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个块的大小
    chunk_overlap=200,  # 块之间的重叠
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.split_documents(documents)
print(f"文档被分割为 {len(texts)} 个块")

# 3. 创建向量存储和检索器
print("步骤2: 创建向量数据库...")
embeddings = get_ali_embeddings()

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="rag_demo",
    persist_directory="./data/chroma_db"  # 持久化存储路径
)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={"k": 4}  # 返回前4个最相关文档
)

# 4. 初始化LLM
print("步骤3: 初始化大语言模型...")
llm = get_ali_model_client()

# 5. 创建提示模板
prompt = ChatPromptTemplate.from_template("""
你是一个专业的问答助手，请根据提供的上下文信息回答问题。
如果上下文中没有足够信息，请明确说明你不知道答案，不要编造信息。

上下文信息：
{context}

用户问题：{input}

请基于上述上下文信息，提供一个准确、详细的答案：
""")

# 6. 创建文档处理链
print("步骤4: 创建文档处理链...")
document_chain = create_stuff_documents_chain(llm, prompt)

# 7. 创建完整的检索链
print("步骤5: 创建检索链...")
rag_chain = create_retrieval_chain(retriever, document_chain)

# 8. 测试问答
print("\n" + "=" * 50)
print("RAG系统已就绪，开始测试...")
print("=" * 50)

# 测试问题
test_questions = [
    "文档中提到的主要技术是什么？",
    "这些技术有什么应用场景？",
    "请总结文档的核心观点。"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n问题 {i}: {question}")
    print("-" * 30)

    # 执行RAG流程
    result = rag_chain.invoke({"input": question})

    print(f"答案: {result['answer']}")
    print(f"\n检索到的文档数量: {len(result['context'])}")

    # 显示检索到的文档片段
    for j, doc in enumerate(result['context'][:2], 1):  # 只显示前2个
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        print(f"文档 {j}: {preview}")

    print("=" * 50)

# 9. 清理资源
print("\n清理向量数据库...")
vectorstore.delete_collection()