import datetime
import os

from dashscope import api_key
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from langchain.chat_models import  init_chat_model
from langchain_core.prompts import  PromptTemplate
# from langchain_core.tools import tool
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.tools import BaseTool
from langchain_classic.agents import AgentExecutor, create_react_agent

from LCEL.lcel_sequence import prompt

from pydantic import BaseModel, Field
from typing import Type, Optional

from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings

# 使用Pydantic定义输入参数的模型。本例中无需输入，所以模型为空。
class DateToolInput(BaseModel):
    # 这个工具不需要任何输入参数。
    # 如果未来需要扩展，可以在这里添加字段，例如：
    # format: str = Field("YYYY-MM-DD", description="日期格式")
    query: Optional[str] = Field(
        default=None,
        description="查询日期的提示词，可为空"
    )

class DateTool(BaseTool):
    """
    一个获取当前具体日期的简单工具。
    它是继承BaseTool类的最简示例。
    """
    # 1. 定义工具名称。这将是Agent在思考时提到的名字。
    name: str = "get_date"
    # 2. 定义工具描述。清晰准确的描述直接决定了Agent能否在正确场景下想起并使用它。
    description: str = "当需要知道今天的准确日期（年月日）时，使用此工具。"

    # 3. 定义参数模式。即使此工具无需参数，也最好显式定义一个空的Schema，这是良好的实践。
    args_schema: Type[BaseModel] = DateToolInput

    # 4. 实现核心的 _run 方法。这里是所有业务逻辑存放的地方。
    def _run(self, query: Optional[str] = None) -> str:
        """执行工具，返回当前日期字符串。"""
        # 使用datetime模块获取当前日期，并格式化为易读的字符串。
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        return f"今天是：{current_date}"

    # 5. （可选）实现 _arun 方法以支持异步调用。简单工具可暂不实现。
    async def _arun(self, query: Optional[str] = None):
        raise NotImplementedError("此工具暂不支持异步调用。")

# ============ 如何使用这个工具？============
# 实例化工具
date_tool = DateTool()

# 查看工具的元信息，这些信息会被提供给大模型（Agent）
print(f"工具名称: {date_tool.name}")
print(f"工具描述: {date_tool.description}")

# 调用工具。因为工具定义中`args_schema`为空，所以invoke可以传入空字典或不传参。
# 方式一：invoke
# result = date_tool.invoke("今天是几月几号？")
# result = date_tool.invoke("")
# print(result) # 输出：今天是：2026-03-13

# 方式二：run
# result = date_tool.run("今天是几月几号？")
# result = date_tool.run("")
# print(result)

# 方式三：Agent
# 获取模型
model = get_lc_model_client()
# 创建工具列表
tools = [date_tool]

# 使用ReAct提示模板，让Agent具备“思考-行动”的推理能力
prompt = "你是人工智能助手。需要帮助用户解决各种问题。"

# 大模型客户端绑定工具
# 创建Agent
agent = create_agent(
    model=model,  # 聊天模型
    tools=tools, # 工具列表
    system_prompt=prompt
)

# 调用示例
result = agent.invoke(
    {"messages": [{"role": "user", "content": "今天是几月几号？"}]},
)

print(result)
print("--" * 20)

