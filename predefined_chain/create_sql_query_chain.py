# 导入必要的库
import re

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_sql_query_chain
# 用于直接调用sql
from langchain_community.tools import QuerySQLDataBaseTool

from models import get_lc_model_client, get_ali_model_client, get_ali_embeddings


# 1. 连接到数据库 (这里创建一个内存SQLite数据库并添加样例数据)
from sqlalchemy import create_engine, text
import pandas as pd

# 创建内存数据库引擎
engine = create_engine("sqlite:///../test_sql.db")

# 创建样例数据
with engine.connect() as conn:
    # 使用此连接执行建表操作
    df = pd.DataFrame({
        'employee_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'department': ['Sales', 'Engineering', 'Sales'],
        'salary': [70000, 85000, 72000]
    })

    # 将DataFrame写入数据库
    df.to_sql('employees', conn, index=False, if_exists='replace')
    # 对于SQLite，最好显式提交
    conn.commit()

    # 使用同一个连接对象来创建LangChain的SQLDatabase对象
    # db = SQLDatabase(engine=engine, engine_connection=conn)
    db = SQLDatabase(engine=engine)

# 3. 初始化大语言模型 (此处以OpenAI为例，你需要设置自己的API_KEY)
# 实践中，完全可以用DeepSeek-V2等高性能开源模型替代
llm = get_lc_model_client()

# 4. 创建查询链！
chain = create_sql_query_chain(llm, db)

# 5. 使用链进行查询
response = chain.invoke({"question": "薪资最高的员工是谁？"})
# response格式为：
# ```sql
# SELECT "name", "salary"
# FROM employees
# ORDER BY "salary" DESC
# LIMIT 1
# ```
# 需要转换后才能直接执行

print("AI生成的SQL语句:", response)
print("==" * 20)
# 使用正则表达式，移除可选的```sql标记和尾部```
clean_sql = re.sub(r"```sql\n|```$", "", response, flags=re.IGNORECASE).strip()
print("清洗后的SQL语句:", clean_sql)
print("==" * 20)

# 输出示例: SELECT name FROM employees ORDER BY salary DESC LIMIT 1

# 6. (可选) 如果你想直接执行SQL并获取结果，可以这样做
execute_query_tool = QuerySQLDataBaseTool(db=db)
result = execute_query_tool.invoke(clean_sql)
print("查询结果:", result)
# 输出示例: [('Bob',)]