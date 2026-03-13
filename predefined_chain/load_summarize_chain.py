# 导入必要的库
# 确保已安装: pip install langchain langchain-openai
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import langchain_community
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_lc_model_client, get_ali_model_client


# 1. 加载文档（这里以文本文件为例）
loader = TextLoader("./data/deepseek百度百科.txt")  # 请替换为你的文档路径
documents = loader.load()

# 2. 初始化大语言模型
llm = get_ali_model_client()

# 3. 定义明确要求中文摘要的提示词模板
# 注意：{text} 是占位符，链会自动将文档内容填充到这里
prompt_template = """请用中文，简洁地总结以下文本的主要内容：

{text}

请确保摘要完全使用中文，并涵盖核心要点。
中文摘要："""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 3. 创建摘要链 - 最简单的调用方式
# -   **`stuff`**：文档总长度小于模型上下文窗口的80%时使用，最简单快速
#     **`map_reduce`**：处理超长文档（如书籍、长报告），支持并行处理
#     **`refine`**：需要最高质量摘要的场景，可生成最连贯的结果
#     **`map_rerank`**：为每个分块评分并选择最佳摘要，适用于多文档汇总
chain = load_summarize_chain(
    llm=llm,
    chain_type="stuff",
    prompt=prompt # 关键：覆盖默认prompt
    # chain_type_kwargs={"prompt": prompt}  # 覆盖默认提示词
)  # 使用stuff策略

# 4. 执行摘要
summary = chain.invoke(documents)
print("文档摘要：")
print(summary["output_text"])