from operator import itemgetter

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, chain


# 1.提供自定义函数
# 输入字符串，返回字符串长度
def length_function(text):
    return len(text)

#将两个字符串长度的数量相乘
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


# 输入字典类型，返回字符串长度的乘积
# 方式一：RunnableLambda(length_function)

# 方式二：装饰器的方式
@chain
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

prompt = ChatPromptTemplate.from_template("{a} + {b} = ? 计算结果是多少？")
model = ChatTongyi()
out = StrOutputParser()

# 1.通过链来计算字符串的长度
# chain = length_function("hello")  # 普通函数调用
# chain = RunnableLambda(length_function)
# # 调用执行链
# print(chain.invoke("hello"))

# 2.通过大模型计算两个字符串的长度的和
# chain = (
#         {"a":itemgetter("k1")| RunnableLambda(length_function) ,
#          "b":itemgetter("k2")| RunnableLambda(length_function) }
#         |prompt | model | out)
#
# print(chain.invoke({"k1": "hello", "k2": "Langchain"}))

# 传统隐式转换：itemgetter("k1")被隐式转换为 RunnableLambda，依赖LangChain的自动类型适配机制。
# 字典{"text1":..., "text2":...}被隐式识别为并行处理结构
# 适用于LangChain 0.1.x及更早版本
chain = (
        {"a":itemgetter("k1") | RunnableLambda(length_function) ,
         "b":({"text1":itemgetter("k1"),"text2":itemgetter("k2")} | multiple_length_function)
        } | prompt | model | out)
print(chain.invoke({"k1": "hello", "k2": "Langchain"}))
# 以上代码可能报错(版本兼容)：Expected a callable type for `func`.Instead got an unsupported type: <class 'langchain_core.runnables.base.RunnableLambda'>
#  | 链式调用中要求每一个节点必须是一个Runnable！！！推荐显式声明风格
# 每个组件都是显式的Runnable实例；使用RunnableParallel明确声明并行结构
# RunnableLambda显式包装每个数据处理单元，完全符合LangChain新版本的类型系统
chain = (
    {
        "a":RunnableLambda(lambda d:d["k1"]) | RunnableLambda(length_function),
        "b":RunnableParallel({
            "text1":RunnableLambda(lambda d:d["k1"]),
            "text2":RunnableLambda(lambda d:d["k2"])
        }) | multiple_length_function
    } | prompt | model | out
)
# print(chain.invoke({"k1": "hello", "k2": "Langchain"}))


# 测试装饰器
# chain = (
#         {"a":itemgetter("k1")| RunnableLambda(length_function) ,  #5
#          # {"text1":"hello","text2":"world"}
#          #  b=25
#          "b":({"text1":itemgetter("k1"),"text2":itemgetter("k2")}|multiple_length_function   ) }
#         |prompt | model | out)
#
# print(chain.invoke({"k1": "hello", "k2": "world"}))

