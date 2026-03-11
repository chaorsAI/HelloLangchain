
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate

from models import get_lc_model_client


# pip install redis
# session_id 识别用户   0.3版本中 redis_url 访问路径
# history = RedisChatMessageHistory(session_id="my_session_id", redis_url="redis://localhost:6379")
# langchain 1.0  url
history = RedisChatMessageHistory(session_id="my_session_id1", url="redis://localhost:6379")

# history.clear()  # 清空历史消息

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是人工智能助手"),
        # 作用就是向提示词中插入一段上下文消息
        # ("placeholder", "{messages}"),
        MessagesPlaceholder(variable_name="messages"),  #QA QA
        # HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
client = get_lc_model_client()
parser = StrOutputParser()
chain =  chat_template | client | parser

while True:
    user_input = input("用户：")
    if user_input == "exit":
        break

    # 添加用户输入
    # print("user_input:" + user_input)
    history.add_user_message(user_input)
    # 访问LLM时，history.messages 获取所有的历史消息
    response = chain.invoke({'messages': history.messages})

    print("history:",history.messages)
    print(f"大模型回复》》》：{response}")

    # 将大模型的回复加入历史记录
    history.add_ai_message(response)

# # 第一次对话
# history.add_user_message("你是谁？")
# aimessage = client.invoke(history.messages)
# history.add_ai_message(aimessage)
# print(aimessage)
# print("==================")
# 第二次对话
# history.add_user_message("重复一次")
# print(history.messages)
# aimessage = client.invoke(history.messages)
# history.add_ai_message(aimessage)
# print(aimessage)




