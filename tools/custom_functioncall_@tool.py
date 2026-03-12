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

from LCEL.lcel_sequence import prompt

from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings


# 获取模型
model = get_ali_model_client()

# 注意：函数的描述必须写在函数体中的第一行
@tool
def get_date():
    # 文本描述相当于工具说明说，大模型正是依靠这段说明来选择对应工具
    """ 获取今天的具体日期 """
    # """ 获取今天的北京的天气 """
    return datetime.date.today().strftime("%Y-%m-%d")


import webbrowser
@tool
def open_browser(url, browser_name=None):
    # """ 获取浏览器，打开网站 """
    """ 获取浏览器，打开网站可以做很多事情，包括查询天气，汽车限号等 """
    if browser_name:
        # 获取特定浏览器的控制器
        browser = webbrowser.get(browser_name)
    else:
        # 使用默认浏览器
        browser = webbrowser
    # 打开浏览器并导航到指定的URL
    browser.open(url)

# 大模型客户端绑定工具
agent = create_agent(
    model,
    tools=[get_date, open_browser],
)

# prompt = ChatPromptTemplate.from_template(
#     SystemMessagePromptTemplate.from_template("user"),
#     HumanMessagePromptTemplate("{content}")
# )
# 执行agent
# agent.invoke()期望的输入是一个字典
# ❌❌❌错误写法
# result = agent.invoke(prompt.format(content = "帮我打开淘宝"))
# result = agent.invoke({"input":"请快速打开咸鱼"})
# ✅✅✅正确写法
# 获取今天的日期
# result = agent.invoke({"messages":[{"role":"user","content":"今天是几月几号？"}]})
# 1. 用户发文
# 2. 大模型自己发现搞不定
# 3. 尝试寻找合适工具。找到tool_calls = get_date
# 4. get_date获取到信息ToolMessage(2026-03-12)
# 5.返回给大模型AIMessage
# result = agent.invoke({"messages":[{"role":"user","content":"请快速打开咸鱼"}]})
# result = agent.invoke({"messages":[{"role":"user","content":"帮我打开淘宝"}]})
result = agent.invoke({"messages":[{"role":"user","content":"今天是几月几号？"}]})
# result = agent.invoke({"messages":[{"role":"user","content":"今天北京的天气怎么样？"}]})
print( result)


# 1.描述正确，打开网址正确
# 2.描述错误，问错误信息，依然打开网址，但无信息
# 3.通过打开网址获取天气

