"""
RunnableBranch基础示例：理解条件路由
"""
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi

import time

print("=" * 60)
print("RunnableBranch基础示例：条件路由")
print("=" * 60)


# 1. 定义条件判断函数
def classify_query(query_data: dict) -> str:
    """根据查询内容分类"""
    query = query_data.get("query", "").lower()

    if "价格" in query or "多少钱" in query or "cost" in query:
        return "price_inquiry"
    elif "故障" in query or "问题" in query or "error" in query:
        return "technical_issue"
    elif "退货" in query or "退款" in query or "return" in query:
        return "refund_request"
    elif "客服" in query or "人工" in query or "support" in query:
        return "human_support"
    else:
        return "general_inquiry"


# 2. 定义不同分支的处理链
llm = ChatTongyi()

# 价格查询分支
price_chain = (
        ChatPromptTemplate.from_template("""
    你是一个专业的销售顾问。用户询问价格信息。

    用户查询：{query}

    请以专业、友好的方式回答价格相关问题。
    如果知道具体价格，请明确告知。
    如果不知道，请提供获取价格的途径。
    """)
        | llm
        | StrOutputParser()
)

# 技术问题分支
tech_chain = (
        ChatPromptTemplate.from_template("""
    你是一个技术专家。用户遇到技术问题。

    用户查询：{query}

    请提供详细的技术解决方案。
    分步骤说明，确保用户能理解。
    如果问题复杂，建议联系技术支持。
    """)
        | llm
        | StrOutputParser()
)

# 退款请求分支
refund_chain = (
        ChatPromptTemplate.from_template("""
    你是一个客服专员。用户希望退款或退货。

    用户查询：{query}

    请友好地解释退款政策。
    询问订单详细信息以便协助。
    提供明确的后续步骤。
    """)
        | llm
        | StrOutputParser()
)

# 通用查询分支
general_chain = (
        ChatPromptTemplate.from_template("""
    你是一个有用的助手。回答用户的通用查询。

    用户查询：{query}

    请提供有帮助的回答。
    """)
        | llm
        | StrOutputParser()
)

# 3. 创建RunnableBranch
# 格式：RunnableBranch( (条件1, 分支1), (条件2, 分支2), ..., 默认分支 )
query_router = RunnableBranch(
    # 条件是一个函数，接收输入数据，返回True/False
    (lambda x: classify_query(x) == "price_inquiry", price_chain),
    (lambda x: classify_query(x) == "technical_issue", tech_chain),
    (lambda x: classify_query(x) == "refund_request", refund_chain),
    # 默认分支（当所有条件都不满足时执行）
    general_chain
)

# 4. 构建完整链：接收查询 -> 路由 -> 处理
full_chain = RunnableLambda(
    lambda x: {"query": x}  # 将字符串包装为字典
) | query_router

# 5. 测试不同查询
test_queries = [
    "iPhone 15的价格是多少？",
    "我的手机无法开机，怎么办？",
    "我想退货，流程是什么？",
    "转人工客服",
    "你们公司的营业时间是什么？",
    "推荐一款适合拍照的手机"
]

print("测试不同查询的路由结果：\n")
for i, query in enumerate(test_queries, 1):
    print(f"{i}. 查询: {query}")
    print("-" * 40)

    start_time = time.time()
    try:
        response = full_chain.invoke(query)
        elapsed = time.time() - start_time

        # 显示分类结果
        category = classify_query({"query": query})
        print(f"分类: {category}")
        print(f"响应时间: {elapsed:.2f}秒")
        print(f"回答: {response}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "=" * 60 + "\n")