# 基于Langchain-RAG的Web网页摘要检索

import os
import bs4
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

from models import get_lc_model_client, ALI_TONGYI_EMBEDDING_MODEL, ALI_TONGYI_API_KEY_OS_VAR_NAME, get_ali_embeddings, \
    get_tencent_embeddings, get_ali_model_client, get_baichuan_embeddings, get_ali_embeddings

#获得访问大模型客户端
client = get_ali_model_client()

# 获得一个嵌入模型的实例
llm_embeddings = get_ali_embeddings()

# #获得文档，文档内容来自网页https://www.news.cn/fortune/20250212/895ac6738b7b477db8d7f36c315aae22/c.html
# loader = WebBaseLoader(
#     web_path=["https://www.news.cn/fortune/20250212/895ac6738b7b477db8d7f36c315aae22/c.html"],
#     bs_kwargs=dict(
#         # 切割，css 中的class选择器完成
#         parse_only = bs4.SoupStrainer(class_=("main-left left","title"))
#     )
# )

loader = WebBaseLoader(
        web_path="https://www.gov.cn/yaowen/liebiao/202603/content_7061810.htm",
        # bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
        # 分割，将网页中目标内容进行分割
        bs_kwargs={"parse_only":bs4.SoupStrainer(id="UCAP-CONTENT")}
        )
docs = loader.load()
# print(len(docs))
# print(docs)

#文本的切割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)
# for s  in documents:
#     print(s,end="**\n")

# 实例化向量空间
vector_store = Chroma.from_documents(documents=documents,embedding=llm_embeddings)

#检索器
retriever = vector_store.as_retriever()

# 注意这里的prompt模板中包含 {context} 和 {input} 的模板
#需要使用{context}，这个变量，来表示上下文，这个变量，会自动从retriever中获取。
#而human中也限定了变量{input}，链的必须使用这个变量。
system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# ====================== 方式1：retrieval_chain + documents_chain ======================
#创建链，预定义链 create_stuff_documents_chain 文档链
documents_chain = create_stuff_documents_chain(client,prompt_template)
#  参数1:是检索器  参数2:是文档链
retrieval_chain = create_retrieval_chain(retriever, documents_chain)
# 用大模型生成答案
resp = retrieval_chain.invoke({"input":"会议摘要?"})

print("=========方式1：retrieval_chain + documents_chain==========")
print(resp["answer"])


# ====================== 方式2：load_summarize_chain ======================
# 定义明确要求中文摘要的提示词模板
# 注意：{text} 是占位符，链会自动将文档内容填充到这里
prompt_template = """请用中文，简洁地总结以下文本的主要内容：

{text}

请确保摘要完全使用中文，并涵盖核心要点。
中文摘要："""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

summarize_chain = load_summarize_chain(
    llm=client,
    chain_type="stuff", # 使用stuff策略
    prompt=prompt
    )

# 4. 执行摘要
summary = summarize_chain.invoke(docs)
print("=========方式2：load_summarize_chain ==========")
# print(summary)
print(summary["output_text"])