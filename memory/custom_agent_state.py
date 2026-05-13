from langchain.agents import create_agent, AgentState
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from llm_init import ollama_llm_qwen

"""
自定义状态：
1.定义一个类，CustomAgentState，传入AgentState，定义要存储的状态字段
2.构建agent的时候指定 state_schema = CustomAgentState
3.调用agent的时候，通过传入自定义状态数据/在agent运行中，通过......方式设置

在Agent运行中，可以通过中间件（@before_model,@after_model,tool）,方式读取，修改状态
"""


class CustomAgentState(AgentState):
    name: str
    hobbies: list
    other_info: dict


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """
    获取用户信息

    :param runtime:
    :return: str
    """

    print("runtime:", runtime)
    name = runtime.state["name"]
    hobbies = runtime.state["hobbies"]
    other_info = runtime.state["other_info"]

    return f"用户姓名:{name},兴趣:{hobbies},其他的信息:{other_info}"


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_user_info],
    # checkpionter是把过往的交互过程，保存在内存，或者DB，Redis
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState
)
config = {"configurable": {"thread_id": "session_1"}}

# agent调用模型，必须是messages的结构体作为入参
response = agent.invoke({
    "messages": [{"role": "user", "content": "我叫小飞侠，你是谁？"}],
    "name": "user001",
    "hobbies": ["football", "volleyball"],
    "other_info": {"city": "sh", "address": "pan gu road"}

}, config=config)
print(type(response))
# messages包含各个环节与大模型的交互对话过程
print(response)
# messages字段返回的最后一个是AIMessage，大模型真正返回的问题解答
print(response["messages"][-1].content)

print("================")

# 此处的问题是 获取用户的信息，大模型会调用Tool
response_two = agent.invoke({"messages": [{"role": "user", "content": "获取用户的信息"}]}, config=config)
print(response_two["messages"][-1].content)

print("================")

state = agent.get_state(config=config)
print(state)
