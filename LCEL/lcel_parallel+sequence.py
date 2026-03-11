from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableParallel, RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI

import asyncio
import time

# 1. 模拟两个“信息源”——通常它们是外部API、数据库查询或工具调用
def query_knowledge_base(product_name: str) -> str:
    """模拟查询产品知识库（可能访问数据库或内部文档）"""
    time.sleep(0.2)  # 模拟网络延迟
    knowledge = {
        "智能音箱X3": "这是我们的旗舰款智能音箱。核心特色：1. 搭载8核AI芯片，唤醒率99.9%；2. 支持全屋智能联动；3. 内置Hi-Fi级音响，获得金耳朵认证；4. 待机时间长达72小时。"
    }
    return knowledge.get(product_name, "未找到该产品的官方资料。")

def query_recent_feedback(product_name: str) -> str:
    """模拟查询近期用户评论（可能调用舆情分析API）"""
    time.sleep(0.3)  # 模拟另一个服务的延迟
    feedback_db = {
        "智能音箱X3": "最近30天用户评价精华：1. 音质受到普遍好评，低音表现突出；2. 与‘智慧家居’App偶尔出现连接不稳定（占比约5%的反馈）；3. 新出的‘儿童模式’很受家庭用户欢迎。"
    }
    return feedback_db.get(product_name, "暂无该产品的近期用户反馈。")


# 2. 核心：构建 RunnableParallel 来并行收集
# 它接收一个字典，定义多个并行的执行分支
parallel_info_gathering = RunnableParallel({
    "official_info": RunnableLambda(lambda x: query_knowledge_base(x["product_name"])),
    "user_feedback": RunnableLambda(lambda x: query_recent_feedback(x["product_name"])),
    # RunnablePassthrough() 用于将原始输入（如product_name）也传递下去
    "product_name": RunnablePassthrough()
})

# 4. 构建 RunnableSequence 来串联分析与决策
# 这是一个提示词模板，它将接收parallel_info_gathering输出的所有信息
analysis_prompt = ChatPromptTemplate.from_template("""
你是一名专业的客服助理。请根据以下关于产品“{product_name}”的信息，生成一份给用户的回复要点。

【官方产品信息】
{official_info}

【近期用户反馈摘要】
{user_feedback}

---
生成要求：
1. 首先概括产品的核心优势（来自官方信息）。
2. 然后，提及用户反馈中注意到的亮点。
3. 最后，如果用户反馈中提到了任何潜在问题或顾虑，请用委婉、专业的方式在回复中有所提及或安抚。
4. 回复语言口语化，亲切，面向最终消费者。
请直接输出回复内容：
""")

# 5. 将并行和串行组合成完整的链
# 使用管道操作符 `|` 连接，形成：输入 -> Parallel -> Sequence -> 输出
agent_chain = (
        {"product_name": RunnablePassthrough()}  # 将用户输入转化为字典格式，键为"product_name"
        | parallel_info_gathering
        | analysis_prompt
        | ChatTongyi()
        | StrOutputParser()
)

# 6. 执行测试
if __name__ == "__main__":
    user_query = "智能音箱X3"
    print(f"用户问题：{user_query}\n")
    print("=" * 50)
    start_time = time.time()

    # 调用链
    response = agent_chain.invoke(user_query)

    end_time = time.time()

    print(f"Agent回复（生成耗时：{end_time - start_time:.2f}秒）：\n")
    print(response)
    print("=" * 50)

    # 为了更直观地展示并行效果，我们可以异步地看看各个分支的耗时
    print("\n【架构师洞察】并行执行验证：")


    async def show_parallel():
        import asyncio
        # 模拟并行任务
        tasks = [
            asyncio.to_thread(query_knowledge_base, "智能音箱X3"),
            asyncio.to_thread(query_recent_feedback, "智能音箱X3")
        ]
        results = await asyncio.gather(*tasks)
        print(f"  知识库查询结果长度：{len(results[0])}字符")
        print(f"  用户反馈查询结果长度：{len(results[1])}字符")


    asyncio.run(show_parallel())
