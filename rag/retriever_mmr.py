# 环境准备：pip install langchain langchain-chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from models import get_lc_model_client, get_ali_embeddings, ALI_TONGYI_API_KEY_OS_VAR_NAME, ALI_TONGYI_EMBEDDING_MODEL, get_ali_model_client


# 1. 模拟一个包含多领域AI知识的小型文档库
documents = [
    Document(page_content="GPU，尤其是NVIDIA的系列产品，通过CUDA架构提供大规模并行计算能力，是训练大模型的基石。"),
    Document(page_content="TPU是谷歌专门为神经网络训练设计的张量处理单元，在特定模型上能效比极高。"),
    Document(page_content="分布式训练框架，如PyTorch DDP和DeepSpeed，解决了单卡显存不足问题，实现了超大规模模型训练。"),
    Document(page_content="长序列处理瓶颈：自注意力复杂度随序列长度平方增长。使用FlashAttention与多查询注意力等算法，在保持性能的前提下大幅降低显存和计算开销。"),
    Document(page_content="注意力机制是Transformer模型的核心，它允许模型在处理序列时动态关注不同部分。"),
    Document(page_content="混合精度训练（FP16/BF16）可显著减少显存占用并提升计算吞吐，是加速训练的关键工程手段。"),
    Document(page_content="模型剪枝和量化是模型压缩的主要技术，用于减少模型大小和推理延迟，便于部署。"),
    Document(page_content="收敛速度慢：模型越大，收敛所需步数越多。分布式优化器与自适应学习率调度可加速收敛，而课程学习（从易到难）能提升学习效率。"),
    Document(page_content="多模态对齐难题：不同模态信息难以融合。对比学习与交叉注意力机制是主流方法，在共享潜空间中对齐文本、图像等特征。"),
    Document(page_content="推理成本高昂：大模型部署成本极高。采用模型量化、稀疏化与MOE架构，在性能损失极小的情况下，大幅降低推理延迟和资源消耗。"),
    Document(page_content="伦理安全，以及如何应对AI快速进化引发的危机"),
    Document(page_content="有很多公司在AI领域都取得了不可忽视的成绩，包括国产厂商"),
]   # 问题query = "训练大型AI模型有哪些技术挑战和解决方案？"

# 2. 文本分割与向量化存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(documents)

# 使用本地Chroma向量数据库
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=get_ali_embeddings() # 或使用开源Embeddings模型
)

# 3. 定义检索器：重点对比普通检索 vs MMR检索
query = "训练大型AI模型有哪些技术挑战和解决方案？"

# 方法A：普通相似性检索
print("=== 普通相似性检索结果 ==")
retriever_standard = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
standard_docs = retriever_standard.invoke(query)
for i, doc in enumerate(standard_docs):
    print(f"[Doc {i+1}]: {doc.page_content}...")

print("==" * 60)

# 方法B：MMR多样性检索
print("\n=== MMR多样性检索结果0 (λ=0.4) ==")
retriever_mmr = vector_store.as_retriever(
    search_type="mmr", # 关键参数
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.4}
)
mmr_docs = retriever_mmr.invoke(query)
for i, doc in enumerate(mmr_docs):
    print(f"[Doc {i+1}]: {doc.page_content}...")


print("==" * 60)
print("\n=== MMR多样性检索结果1 (λ=0.1) ==")
retriever_mmr1 = vector_store.as_retriever(
    search_type="mmr", # 关键参数
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.1}
)
mmr_docs1 = retriever_mmr1.invoke(query)
for i, doc in enumerate(mmr_docs1):
    print(f"[Doc {i+1}]: {doc.page_content}...")