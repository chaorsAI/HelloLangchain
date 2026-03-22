from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models import *


# 1.构建提示词
prompt = ChatPromptTemplate([
    ("system", "把用户输入的中文翻译成{language}"),
    ("user", "{text}"),
])
prompt = prompt.format(language="英文", text="Jeff喜欢打篮球")


# 2. 访问大模型
result = qwen.invoke(prompt)
print(result)
print('-' * 100)

# 使用输出解析器 StrOutputParser
str_parser = StrOutputParser()
str_result = str_parser.invoke(result)
print("StrOutputParser:", str_result)
print('-' * 100)


# 03_content_blocks_v1(版本V1.1)
print(result.content_blocks)
