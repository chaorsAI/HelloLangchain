'''学习如何将工具和大模型进行绑定和如何自定义工具
需要安装
pip install mysqlclient
pip install SQLALchemy'''
import os
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults

from models import get_lc_model_client, get_ali_model_client, ALI_TONGYI_MAX_MODEL

#获得访问大模型客户端，这里为什么要改变模型？
client = get_ali_model_client(model=ALI_TONGYI_MAX_MODEL)

#数据库配置
HOSTNAME ='127.0.0.1'
PORT ='3306'
DATABASE = 'world'
USERNAME = 'root'
PASSWORD ='1234'
MYSQL_URI ='mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)

#定义一个函数，用于获取数据库中的所有表名，并且包装为LangChain中的工具

def get_table_names():
    return db.get_table_names()
# 函数的描述绑定到工具中
get_table_names_tool = Tool(
    name="获取表名",
    func=get_table_names,
    description="获取数据库中的所有表名"
)
#把工具绑定到大模型中
client_with_tools = client.bind_tools([get_table_names_tool])

resp = client_with_tools.invoke([HumanMessage(content="请从国家表中查询出China的所有数据")])
print(resp)
print("**",resp.content)
print("**",resp.tool_calls)
'''
从上面的执行结果可以看到，大模型判断出为了回答问题，需要使用工具，但是工具的使用不是大模型负责的，而应该是我们的应用负责的，
所以在大模型使用工具回答问题的过程中，往往需要多次和大模型交互才能得到最终的结果。
针对这种情况，在LangChain应用程序中调用工具，链是非常好的选择
'''
