#LangChain访问MySQL数据库
from langchain_community.utilities import SQLDatabase

#数据库配置
HOSTNAME ='127.0.0.1'
PORT ='3306'
DATABASE = 'world'
USERNAME = 'root'
PASSWORD ='1234'
MYSQL_URI ='mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
db = SQLDatabase.from_uri(MYSQL_URI)
# 获取数据库中所有的表名称
print(db.get_usable_table_names())
# 执行sql的函数
print(db.run('select * from country limit 1'))

# 将自然语言转sql,执行获取对应的结果
# 执行工具，获取对应的数据库名称
# 输入需求，执行sql 返回对应的结果



