# 最简单的APIChain示例
from langchain_openai import ChatOpenAI
from langchain_classic.chains import APIChain
from langchain_classic.prompts import PromptTemplate
import requests
import json

from models import get_lc_model_client, get_ali_model_client


# 1. 初始化大语言模型（使用DeepSeek-V2）
llm = get_lc_model_client()

# 2. 准备一个简单的API文档（告诉模型如何调用天气API）
api_docs = """
这是一个天气查询API，可以查询城市的气温。

基础URL: https://api.open-meteo.com/v1/forecast
请求方法: GET
参数: 
  latitude: 纬度 (如: 39.9042 表示北京)
  longitude: 经度 (如: 116.4074 表示北京)
  current: 固定为 "temperature_2m" 表示查询当前温度

示例调用: https://api.open-meteo.com/v1/forecast?latitude=39.9042&longitude=116.4074&current=temperature_2m

响应格式: {"current": {"temperature_2m": 15.5}}
"""

# 3. 创建APIChain
chain = APIChain.from_llm_and_api_docs(
    llm=llm,
    api_docs=api_docs,
    limit_to_domains=["https://api.open-meteo.com"],  # 只允许调用这个域名
    verbose=True  # 显示详细过程
)

# 4. 使用链进行查询
question = "查询北京现在的温度"
result = chain.run(question)
print(f"问题: {question}")
print(f"回答: {result}")