
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from models import *


# 人机交互中间件 Human In The Loop(HITL) 允许您在代理工具调用中添加人工监督。
#  当模型提出可能需要审核的操作（例如，写入文件或执行 SQL）时，中间件可以暂停执行并等待决策。
#
#  它通过检查每个工具调用是否符合可配置的策略来实现这一点。如果需要干预，中间件会发出中断以暂停执行。
#   图状态使用 LangGraph 的持久层保存，因此执行可以安全地暂停并在稍后恢复。
#
#    决策类型	        描述	                                    示例用例
# -------------------------------------------------------------------------------
# ✅ approve	该行动按原样批准并执行，未作任何更改。	    请按原样发送电子邮件草稿。
# ✏️ edit	    工具调用经过修改后执行。	                发送电子邮件前请更改收件人
# ❌ reject	    工具调用被拒绝，并在对话中添加了解释。	    驳回邮件草稿并解释如何修改。



# 定义工具
@tool
def delete_file(path: str) -> str:
    """从文件系统删除一个文件."""
    print("Tool: delete_file")
    return f"已删除 {path}"

@tool
def read_file(path: str) -> str:
    """从文件系统读取一个文件."""
    print("Tool: read_file")
    return f"内容 {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件."""
    print("Tool: send_email")
    return f"发送邮件到 {to}"

# 必须支持多轮会话
checkpointer = MemorySaver()

# 创建agent
#  这里使用的是 create_deep_agent,
#  如果是create_agent(), interrupt_on需要写入：
#          middleware=[
#             HumanInTheLoopMiddleware(
#                 interrupt_on={}
#             )
#          ]
agent = create_deep_agent(
    model=qwen,
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # 默认: approve, edit, reject
        "read_file": False,   # 不需要人工审核
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # 要么同意，要么拒绝
    },
    checkpointer=checkpointer  # 必需!
)

# 处理中断
#    当触发中断时，代理会暂停执行并返回控制权。检查结果中是否存在中断并进行相应的处理。

import uuid
from langgraph.types import Command


# 创建 config 带 thread_id 保持会话状态
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 调用 09_Agent_v1
result = agent.invoke(
    input={"messages": [{"role": "user", "content": "删除文件temp.txt"}]},
    config=config
)

# 如果遇到需要中断，则进入if
if result.get("__interrupt__"):
    # 提取 中断 信息
    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]

    # 创建一个从工具名称到检查配置的查找映射
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    # 向用户显示挂起的操作
    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"Tool: {action['name']}")
        print(f"Arguments: {action['args']}")
        print(f"Allowed decisions: {review_config['allowed_decisions']}")

    # 获取用户决策（每个action_request一个，按顺序）
    decisions = [
        {"type": "approve"},  # 用户同意删除
        # {"type": "reject"},  # 用户拒绝删除
    ]

    # 通过决策恢复执行
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config  # 必须使用相同的配置！
    )

print()

# print(result["messages"])
print(result["messages"][-1].content)

