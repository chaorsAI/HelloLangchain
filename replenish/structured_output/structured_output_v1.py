from models import *
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy


# 结构化输出 Structured output
#    在某些情况下，您可能希望代理以特定格式返回输出。LangChain通过response_format参数提供了结构化输出的策略。


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# ToolStrategy: 工具策略
#     使用人工调用工具生成结构化输出。这适用于任何支持工具调用的模型：
agent = create_agent(
    model=qwen,
    tools=[],
    response_format=ToolStrategy(ContactInfo)
)


result = agent.invoke({
    "messages": [
        {"role": "user",
         "content": "从：Jeff，jeff@123.com, 16263668888 提取联系信息"}
    ]
})

print(result)
# print(type(result["structured_response"]))  # <class '__main__.ContactInfo'>
print(result["structured_response"])  # name='Jeff' email='jeff@123.com' phone='16263668888'






