from operator import itemgetter

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, chain
from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser

# 创建一个Pydantic模型, 用于估算费用
class TripDetails(BaseModel):
    destination: str = Field(description="旅行目的地城市")
    duration: int = Field(description="旅行天数")
    estimated_cost_per_person: float = Field(description="估算人均成本", default=None)
    summary: str = Field(description="给用户的旅行建议摘要")

def calculate_total_cost(trip_details: TripDetails) -> TripDetails:
    """根据旅行天数和目的地，估算一个非常粗略的人均成本"""
    daily_cost = {"北京": 600, "东京": 1200}.get(trip_details.destination, 500)
    trip_details.estimated_cost_per_person = daily_cost * trip_details.duration
    return trip_details

@chain
def chain_calculate_total_cost(trip_info: dict) -> dict:
    return calculate_total_cost(trip_info)

model = ChatTongyi()
out = PydanticOutputParser(pydantic_object=TripDetails)
format_instructions = out.get_format_instructions()
prompt = ChatPromptTemplate.from_template("""
你是一名专业的旅游助手。请根据以下关于旅游信息“{tripInfo}”，生成一份给用户的回复要点\n{format_instructions}。

【旅游信息】
{tripInfo}

""")
prompt = prompt.partial(format_instructions=format_instructions)

# 方法1
chain = prompt | model | out | RunnableLambda(calculate_total_cost)
# 方法2
# chain = prompt | model | out | chain_calculate_total_cost
result = chain.invoke({"tripInfo": "北京-东京 3天"})
print(result.model_dump()) # 查看完整的结构化结果
print("-" * 50)
print(result.summary) # 访问文本回复
print("-" * 50)
print(f"估算成本：{result.estimated_cost_per_person}") # 访问计算后的成本