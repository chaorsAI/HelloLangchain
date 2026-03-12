import requests
from datetime import datetime
from typing import Optional, Type
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
from LCEL.lcel_sequence import prompt

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
            'weather': current.get('weatherDesc', [{}])[0].get('value'),
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

    def _run(self, city: str) -> str:
        """工具的核心执行逻辑，返回字符串结果供Agent读取"""
        result = get_weather_simple(city)
        if result.get('error'):
            return f"无法获取{city}的天气：{result['message']}"
        return (
            f"{result['city']}的天气：{result['description']}，"
            f"温度{result['temperature']}°C，"
            f"湿度{result['humidity']}%。"
            f"（数据来源：{result['source']}，更新时间{result['last_update']}）"
        )

# ---------- 3. Agent组装与执行 ----------
def create_weather_agent(openai_api_key: str):
    """
    创建并返回一个具备天气查询能力的Agent执行器
    """
    # 获取模型
    model = get_ali_model_client()
    # 创建工具列表
    tools = [WeatherTool()]

    # 使用ReAct提示模板，让Agent具备“思考-行动”的推理能力
    prompt = PromptTemplate.from_template(
        """
        你是一个有用的助手，可以回答用户问题并使用工具。
        如果你需要使用工具，请按照以下格式回复：

        思考：[你的推理过程]
        行动：[工具名称]
        行动输入：[工具输入参数]

        工具可用：
        {tools}

        用户问题：{input}

        {agent_scratchpad}
        """
    )

    # 大模型客户端绑定工具
    # 创建Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 设为True可看到Agent的思考过程
        handle_parsing_errors=True
    )

    return agent_executor

# 使用示例
if __name__ == '__main__':
    # 1. 创建Agent
    weather_agent = create_weather_agent()

    # 2. 调用示例
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": "北京的天气怎么样"}]
    })
    print(result)
    print("--" * 20)

    # 示例2：包含比较的查询
    result2 = weather_agent.invoke({
        "messages": [{"role": "user", "content": "北京和深圳哪里更热？"}]
    })
    print("示例2回复:", result2["output"])

    # 示例3：多轮对话
    messages = [
        HumanMessage(content="我想知道几个城市的天气"),
        AIMessage(content="我可以帮您查询，请告诉我城市名称。"),
        HumanMessage(content="先看下杭州的天气吧")
    ]

    result3 = weather_agent.invoke({"messages": messages})
    print("示例3回复:", result3["output"])