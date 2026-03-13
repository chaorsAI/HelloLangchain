# 导入必要的库
# 确保已安装: pip install langchain langchain-openai
import os
import bs4
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import langchain_community
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from models import get_lc_model_client, get_ali_model_client


# 1. 加载文档（这里以文本文件为例）
loader = TextLoader("./data/deepseek百度百科.txt")  # 请替换为你的文档路径
documents = loader.load()

# 2. 初始化大语言模型
# 如果需要控制LLM调用的并发数，可以在get_ali_model_client函数中进行相应配置
llm = get_ali_model_client()

# 可选：如果需要进一步控制文档处理，可以调整文档分割参数
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_documents = text_splitter.split_documents(documents)

# 3. 定义明确要求中文摘要的提示词模板

# ====================== 第一部分：stuff模式 ======================
# 注意：{text} 是占位符，链会自动将文档内容填充到这里
stuff_template = """你的任务是为中文用户生成摘要。
你必须，且只能使用中文进行回复。：

{text}

请确保摘要完全使用中文，并涵盖核心要点。
中文摘要："""
stuff_prompt = PromptTemplate(template=stuff_template, input_variables=["text"])

# 3. 创建摘要链 - 最简单的调用方式
# -   **`stuff`**：文档总长度小于模型上下文窗口的80%时使用，最简单快速
#     **`map_reduce`**：处理超长文档（如书籍、长报告），支持并行处理
#     **`refine`**：需要最高质量摘要的场景，可生成最连贯的结果

# chain = load_summarize_chain(
#     llm=llm,
#     chain_type="stuff", # 使用stuff策略
#     prompt=stuff_prompt # 关键：覆盖默认prompt
# )
# print("----------" + "chain_type=stuff 示例" + "----------")


# ====================== 第二部分：refine模式 ======================
# 初始Prompt（处理第一个文档块）
initial_template = """你是一个专业的中文文档处理AI。你的所有输出必须且只能使用中文。

请仔细阅读以下文本内容，并生成一个初始的**中文**摘要。

文本内容：
{text}

请严格遵守以下要求：
1. 摘要必须完全使用中文撰写
2. 提取核心事实与观点
3. 保持客观，不添加额外评论
4. 控制长度在200字以内

重要提醒：**你正在为中文用户工作，必须使用中文输出**。

初始中文摘要："""
initial_prompt = PromptTemplate(template=initial_template)
# initial_prompt = PromptTemplate(template=initial_template, input_variables=['text'])

refine_template = """你是一个专业的中文文档处理AI。你的所有输出必须且只能使用中文。

现有摘要：
{existing_answer}

现在需要你基于以下新增内容，对上述摘要进行完善和优化：

新增内容：
{text}

请严格按照以下步骤操作：
1. 仔细评估新增内容是否包含重要信息
2. 如果新增内容不重要，保持原摘要不变
3. 如果新增内容重要，将其关键信息融合到现有摘要中
4. 用更精炼、流畅的**中文**重写整个摘要
5. 禁止切换为英文或其他语言
6. 保持摘要的连贯性和完整性

重要提醒：**你正在为中文用户工作，必须使用中文输出**。

优化后的完整中文摘要："""
refine_prompt = PromptTemplate(template=refine_template)
# refine_prompt = PromptTemplate(template=initial_template, input_variables=['existing_answer', 'text'])
# chain = load_summarize_chain(
#     llm=llm,
#     chain_type="refine", # 使用refine策略
#     question_prompt=initial_prompt,
#     refine_prompt=refine_prompt
# )
# print("----------" + "chain_type=refine 示例" + "----------")


# ====================== 第三部分：map_reduce模式 ======================
# 1. 定义Map阶段的Prompt
map_template = """
请用一段话简要总结以下文本的核心内容，保留关键事实、数据和观点：
文本：\n{text}\n
摘要：
"""
map_prompt = PromptTemplate(template=map_template)

# 2. 定义Reduce（Combine）阶段的Prompt
combine_template = """
你收到了以下几份关于同一主题的摘要片段，请将它们整合成一份完整的、连贯的、无重复的总摘要：

{text}

完整的总摘要：
"""
combine_prompt = PromptTemplate(template=combine_template)

chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce", # 指定使用map_reduce策略
    map_prompt=map_prompt,
    combine_prompt=combine_prompt
)
print("----------" + "chain_type=map_reduce 示例" + "----------")


# 4. 执行摘要
summary = chain.invoke(documents)
print("文档摘要：")
print(summary["output_text"])