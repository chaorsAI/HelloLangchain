# 完整对话式RAG系统示例
# 确保已安装: pip install langchain langchain-openai langchain-chroma
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document

import asyncio
from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings

# 1. 准备示例知识库（实际应用中来自真实文档）
documents = [
    Document(
        page_content="LangChain是一个用于开发大语言模型应用的框架，由Harrison Chase在2022年创建。它提供模块化组件，简化了构建复杂AI应用的过程。"),
    Document(
        page_content="LCEL（LangChain Expression Language）是LangChain v0.1.0引入的声明式语言，用于组合链和代理。它使用管道操作符(|)连接组件。"),
    Document(
        page_content="RAG（检索增强生成）技术结合检索和生成，让大模型能够访问外部知识库。LangChain通过RetrievalQA链实现RAG。"),
    Document(
        page_content="对话式RAG允许用户进行多轮对话，系统能记住历史上下文。这通过ConversationalRetrievalChain或create_history_aware_retriever实现。"),
    Document(page_content="向量数据库如Chroma、Pinecone、Weaviate用于存储和检索文档嵌入。LangChain支持多种向量存储后端。"),
    Document(
        page_content="create_history_aware_retriever是对话式RAG的核心组件，它重写当前查询以包含历史上下文，使检索更准确。"),
]

# 2. 创建向量存储和基础检索器
print("创建向量数据库和基础检索器...")
embeddings = get_ali_embeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 初始化LLM
llm = get_ali_model_client()

# 4. 创建历史感知检索器
print("\n创建历史感知检索器...")
contextualize_q_system_prompt = """
你是一个查询重写助手。基于对话历史和最新的用户问题，创建一个独立的问题。
这个独立的问题应该包含理解用户意图所需的所有上下文。

如果对话历史是相关的，将其整合到新问题中。
如果对话历史不相关或为空，直接返回原始问题。

对话历史：
{chat_history}

用户问题：{input}

独立的问题：
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=base_retriever,  # 基础检索器
    prompt=contextualize_q_prompt
)

# 5. 创建对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 6. 创建问答链
print("\n创建问答链...")
qa_system_prompt = """
你是一个有帮助的AI助手，基于以下上下文和对话历史回答问题。

上下文：
{context}

对话历史：
{chat_history}

用户问题：{input}

基于上下文回答问题。如果上下文不包含相关信息，请明确说明你不知道，不要编造信息。
保持答案简洁、准确。
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 7. 创建完整的对话式RAG链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# 8. 定义对话处理器
def process_conversation():
    print("\n" + "=" * 60)
    print("对话式RAG系统已就绪")
    print("输入'退出'或'quit'结束对话")
    print("=" * 60)

    while True:
        # 获取用户输入
        user_input = input("\n你: ").strip()

        if user_input.lower() in ["退出", "quit", "exit"]:
            print("结束对话。")
            break

        if not user_input:
            continue

        try:
            # 准备输入（包含记忆）
            inputs = {
                "input": user_input,
                "chat_history": memory.chat_memory.messages
            }

            # 调用RAG链
            print("思考中...", end="", flush=True)
            result = rag_chain.invoke(inputs)
            print("\r" + " " * 20 + "\r", end="")  # 清除"思考中..."

            # 显示答案
            print(f"助手: {result['answer']}")

            # 显示检索到的文档（调试信息）
            print(f"\n[检索到 {len(result['context'])} 个相关文档]")
            for i, doc in enumerate(result['context'], 1):
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  文档{i}: {preview}")

            # 保存到记忆
            memory.save_context({"input": user_input}, {"answer": result['answer']})

        except Exception as e:
            print(f"\n错误: {str(e)}")


# 9. 异步版本（适用于Web应用）
async def async_conversation_example():
    """异步版本的对话示例，适用于Web应用"""
    print("异步对话示例...")

    # 模拟对话历史
    conversation_history = [
        {"role": "human", "content": "什么是LangChain？"},
        {"role": "ai", "content": "LangChain是一个用于开发大语言模型应用的框架。"}
    ]

    # 模拟后续问题
    follow_up_questions = [
        "它有哪些主要组件？",
        "RAG在LangChain中如何实现？",
        "什么是历史感知检索器？"
    ]

    for question in follow_up_questions:
        print(f"\n用户: {question}")

        # 准备输入
        inputs = {
            "input": question,
            "chat_history": conversation_history
        }

        # 异步调用
        result = await rag_chain.ainvoke(inputs)

        print(f"助手: {result['answer'][:150]}...")

        # 更新对话历史
        conversation_history.append({"role": "human", "content": question})
        conversation_history.append({"role": "ai", "content": result['answer']})

        # 显示重写后的查询（调试）
        if hasattr(history_aware_retriever, 'last_rewritten_query'):
            print(f"[重写查询: {history_aware_retriever.last_rewritten_query}]")

        await asyncio.sleep(0.5)  # 模拟延迟


# 10. 高级功能：查询重写可视化
def visualize_query_rewriting():
    """可视化展示查询重写过程"""
    print("\n" + "=" * 60)
    print("查询重写过程可视化")
    print("=" * 60)

    # 模拟多轮对话
    test_conversations = [
        {
            "history": [
                ("human", "LangChain是什么？"),
                ("ai", "LangChain是一个用于开发大语言模型应用的框架。")
            ],
            "current_question": "它支持哪些数据库？"
        },
        {
            "history": [
                ("human", "什么是RAG？"),
                ("ai", "RAG是检索增强生成，结合检索和生成技术。"),
                ("human", "它在LangChain中怎么用？"),
                ("ai", "通过RetrievalQA链实现RAG功能。")
            ],
            "current_question": "那对话式版本呢？"
        },
        {
            "history": [],  # 无历史
            "current_question": "什么是向量数据库？"
        }
    ]

    for i, conv in enumerate(test_conversations, 1):
        print(f"\n案例 {i}:")
        print(f"对话历史: {conv['history']}")
        print(f"当前问题: {conv['current_question']}")

        # 创建临时记忆
        temp_memory = ConversationBufferMemory(return_messages=True)
        for role, content in conv['history']:
            if role == "human":
                temp_memory.chat_memory.add_user_message(content)
            else:
                temp_memory.chat_memory.add_ai_message(content)

        # 模拟重写过程
        inputs = {
            "input": conv['current_question'],
            "chat_history": temp_memory.chat_memory.messages
        }

        # 手动重写查询（模拟）
        rewritten_query = llm.invoke(
            contextualize_q_prompt.format_messages(**inputs)
        ).content

        print(f"重写后查询: {rewritten_query}")
        print("-" * 40)


# 11. 性能优化版本
class OptimizedConversationalRAG:
    def __init__(self, llm, retriever, max_history_length=10):
        self.llm = llm
        self.base_retriever = retriever
        self.max_history_length = max_history_length
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_history_length=max_history_length
        )

        # 创建历史感知检索器（带压缩的提示）
        compressed_prompt = ChatPromptTemplate.from_messages([
            ("system", """基于最近{max_turns}轮对话历史重写查询。只保留必要上下文。

            历史：{chat_history}
            当前问题：{input}
            独立查询："""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=compressed_prompt
        )

        # 创建问答链
        qa_prompt = ChatPromptTemplate.from_template("""
        基于上下文回答。上下文不完整时，结合常识但需注明。

        上下文：{context}
        历史：{chat_history}
        问题：{input}

        答案：""")

        self.qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.qa_chain
        )

    def query(self, question, use_history=True):
        """执行查询，可选择是否使用历史"""
        inputs = {"input": question}

        if use_history:
            inputs["chat_history"] = self.memory.chat_memory.messages

        result = self.rag_chain.invoke(inputs)

        # 更新记忆
        if use_history:
            self.memory.save_context(
                {"input": question},
                {"output": result["answer"]}
            )

        return result

    def clear_history(self):
        """清空对话历史"""
        self.memory.clear()


# 运行示例
if __name__ == "__main__":
    print("=" * 60)
    print("create_history_aware_retriever 示例")
    print("=" * 60)

    # 运行查询重写可视化
    visualize_query_rewriting()

    # 运行异步示例
    print("\n" + "=" * 60)
    print("异步对话示例")
    print("=" * 60)
    asyncio.run(async_conversation_example())

    # 使用优化版本
    print("\n" + "=" * 60)
    print("优化版本示例")
    print("=" * 60)

    optimized_rag = OptimizedConversationalRAG(llm, base_retriever, max_history_length=5)

    test_questions = [
        "什么是LangChain？",
        "它有哪些主要功能？",
        "RAG是什么？",
        "如何在LangChain中实现RAG？"
    ]

    for q in test_questions:
        print(f"\n用户: {q}")
        result = optimized_rag.query(q)
        print(f"助手: {result['answer'][:100]}...")
        print(f"检索文档数: {len(result['context'])}")

    # 显示记忆状态
    print(f"\n当前记忆长度: {len(optimized_rag.memory.chat_memory.messages)} 条消息")

    # 清空记忆
    optimized_rag.clear_history()
    print("记忆已清空")