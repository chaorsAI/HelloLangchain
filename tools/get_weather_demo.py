# 天气查询Demo实战 get_weather_demo

import requests
from datetime import datetime
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import  init_chat_model
from langchain_core.prompts import  PromptTemplate
# from langchain_core.tools import tool
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import BaseTool, StructuredTool
from langchain_classic.schema import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings

# ---------- 1. 核心天气查询工具 ----------
def get_weather_simple(city: str) -> dict:
    """
    【演示用途】一个不依赖API Key的简易天气查询函数。
    核心脆弱性：完全依赖于一个第三方、可随时变更或失效的公开端点。

    Returns:
        dict: 包含天气信息或错误的字典。
    """
    # 示例使用一个理论上存在的公共API（此URL仅为示例格式，可能无效）
    url = f"http://wttr.in/{city}?format=j1"

    try:
        # 明确设置用户代理，并设置短超时
        headers = {'User-Agent': 'Mozilla/5.0 (Demo Weather Client)'}
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # 深度解析依赖API返回结构，这里是一个假设的路径
        current = data.get('current_condition', [{}])[0]

        return {
            'city': city,
            'temp_C': current.get('temp_C'),
            'temp_feels':current.get('FeelsLikeC'),
            'weather': current.get('weatherDesc', [{}])[0].get('value'),
            'humidity':current.get('humidity'),
            'pressure':current.get('pressure'),
            'source': 'wttr.in',
            'last_update': datetime.now().strftime('%H:%M')
        }
    except Exception as e:
        # 任何异常都返回一个友好的错误信息
        return {
            'city': city,
            'error': True,
            'message': f'无法获取天气。服务可能不稳定或已变更。技术信息: {type(e).__name__}'
        }

# ---------- 2. 工具定义层 ----------
class WeatherQueryInput(BaseModel):
    """天气查询的输入参数模式"""
    city: str = Field(description="要查询天气的城市名称，例如北京、Shanghai")

class WeatherTool(BaseTool):
    """天气查询工具的LangChain标准化封装"""
    name: str = "get_weather"
    description: str = "查询指定城市的实时天气信息，包括温度、天气状况和湿度。"
    args_schema: Type[BaseModel] = WeatherQueryInput

    def _run(self, city:str) -> str:
        """工具的核心执行逻辑，返回字符串结果供Agent读取"""
        weather_result = get_weather_simple(city)
        if weather_result.get('error'):
            return f"无法获取{city}的天气：{weather_result['message']}"
        return (
            f"{weather_result['city']}的天气：{weather_result['weather']},"
            f"温度：{weather_result['temp_C']}°C,"
            f"体感温度：{weather_result['temp_feels']}°C,"
            f"湿度：{weather_result['humidity']}%,"
            f"大气压：{weather_result['pressure']}%,"
            f"(数据来源：{weather_result['source']}，更新时间：{weather_result['last_update']})"
        )
    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("此工具暂不支持异步调用。")

# ---------- 3. Agent组装与执行 ----------
def create_weather_agent():
    """
    创建并返回一个具备天气查询能力的Agent执行器
    """
    # 获取模型
    model = get_lc_model_client()
    # model = get_ali_model_client()
    # 创建工具列表
    tools = [WeatherTool()]

    # 使用ReAct提示模板，让Agent具备“思考-行动”的推理能力
    system_prompt = "你是人工智能助手。需要帮助用户解决各种问题。"

    # 创建短期记忆实例
    # 短期记忆构建：智能体中使用InMemorySaver()实现单会话的短期记忆
    memory = InMemorySaver()

    # 大模型客户端绑定工具
    # 创建Agent
    agent = create_agent(
        model=model,  # 聊天模型
        tools=tools,  # 工具列表
        system_prompt=system_prompt,
        checkpointer=memory  # 传入记忆组件
    )

    return agent

# 使用示例
if __name__ == '__main__':
    # 1. 创建Agent
    weather_agent = create_weather_agent()

    # 2. 调用示例
    print("----------" + "示例1" + "----------")
    result = weather_agent.invoke(
        input={"messages": [{"role": "user", "content": "北京的天气怎么样"}]},
        config={"configurable": {"thread_id": "user_1"}}  # 会话唯一标识，用于区分不同用户
    )
    print(result)

    # 示例2：包含比较的查询
    print("----------" + "示例2：比较查询" + "----------")
    result2 = weather_agent.invoke(
        input={"messages": [{"role": "user", "content": "北京和深圳哪里更热？"}]},
        config={"configurable": {"thread_id": "user_1"}}  # 会话唯一标识，用于区分不同用户
    )
    print("示例2回复:", result2)


    # 示例3：多轮对话
    print("-------------------" + "示例3：多轮对话" + "-------------------")
    # 第一步：用户发起新对话
    print("******" + "第一轮对话" + "******")
    result2_0 = weather_agent.invoke(
        input={"messages": [{"role": "user", "content": "我想知道几个城市的天气"}]},
        config={"configurable": {"thread_id": "user3_1"}}  # 会话唯一标识，用于区分不同用户
    )
    print("AI首次回复:", result2_0["messages"][-1].content)  # 输出AI的回复

    # 第二步：用户继续提问，使用相同的thread_id
    print("******" + "第二轮对话" + "******")
    result2_1 = weather_agent.invoke(
        input={"messages": [{"role": "user", "content": "先看下杭州的天气吧"}]},
        config={"configurable": {"thread_id": "user3_1"}}  # 会话唯一标识，用于区分不同用户
    )
    print("AI二轮回复:", result2_1["messages"][-1].content)

    print("******" + "第三轮对话" + "******")
    result2_2 = weather_agent.invoke(
        input={"messages": [{"role": "user", "content": "我最开始问的啥问题？"}]},
        config={"configurable": {"thread_id": "user3_1"}}  # 会话唯一标识，用于区分不同用户
    )
    print("AI三轮回复:", result2_2["messages"][-1].content)