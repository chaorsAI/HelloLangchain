from typing import List
from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from models import get_lc_model_client

# 1. 定义你的结构化数据模型
class MovieReview(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分，0-10分")
    tags: List[str] = Field(description="电影标签")

    @validator('year')
    def year_must_be_valid(cls, v):
        if v > 2025:
            raise ValueError('年份不能超过2025年')
        return v

# 2. 创建解析器，并注入格式指令到提示词
parser = PydanticOutputParser(pydantic_object=MovieReview)
prompt = PromptTemplate(
    template="请根据以下影评文本，提取信息。\n{format_instructions}\n文本：{review}\n",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model = get_lc_model_client()
# 3. 构建并执行链
chain = prompt | model | parser
review_text = "《流浪地球2》于2023年上映，是一部宏大的科幻灾难片，我认为可以打9.3分，它包含了科幻、灾难和爱国情怀等元素。"
try:
    result = chain.invoke({"review": review_text})
    print(result)
    print("--" * 10)
    print(f"标题：{result.title}")
    print(f"年份：{result.year}")
    print(f"标签：{result.tags}")
except Exception as e:
    print(f"解析失败，模型输出格式可能不符合要求：{e}")