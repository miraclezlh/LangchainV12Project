
from langchain.agents import create_agent, AgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from llm_init import ollama_llm_qwen

"""
自定义状态：
1.定义一个类，这个类AgentState，定义要存储的状态字段
2.构建agent的时候指定 state_schema = CustomAgentState
3.调用agent的时候，通过传入自定义状态数据/在agent运行中，通过......方式设置

在Agent运行中，可以通过中间件（@before_model,@after_model,tool）,方式读取，修改状态
"""

class CustomAgentState(AgentState):
    name: str
    hobbies: list
    other_info: dict

@tool
def get_user_info(runtime: ToolRuntime)  -> str:
    """
    获取用户的信息

    Args:
        runtime: ToolRuntime
    """
    print("runtime:",runtime)
    name = runtime.state["name"]
    hobbies = runtime.state["hobbies"]
    other_info = runtime.state["other_info"]

    return f"用户姓名:{name},兴趣:{hobbies},其他的信息:{other_info}"

@tool
def update_user_info(name:str, hobbies:list, runtime: ToolRuntime)  -> Command:
    """
    更新用户的信息

    Args:
        runtime: ToolRuntime
    """

    if not name or not hobbies:
        return Command(
            update={
                "messages":[
                    ToolMessage(
                        content="缺少用户名或爱好",
                        tool_call_id = runtime.tool_call_id
                    )
                ]
            }
        )

    update={
        "name":name,
        "hobbies":hobbies,
        "messages":[
            ToolMessage(
                content=f"用户姓名:{name},兴趣:{hobbies}",
                tool_call_id = runtime.tool_call_id
            )
        ]
    }

    return Command(
        update=update
    )


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_user_info,update_user_info],
    # system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。",
    # checkpionter是把过往的交互过程，保存在内存，或者DB，Redis
    checkpointer= InMemorySaver(),
    state_schema= CustomAgentState
)

config = {"configurable": {"thread_id": "session_1"}}

# agent调用模型，必须是mesages的结构体作为入参
response = agent.invoke({
    "messages": [{"role": "user", "content": "我叫小飞侠，你是谁？"}],
    "name":"小飞侠",
    "hobbies":["football","volleyball"],
    "other_info":{"city":"sh","address":"pan gu road"}

}, config=config)
print(type(response))
print(response)
print(response["messages"][-1].content)

print("================")

response_two = agent.invoke({"messages": [{"role": "user", "content": "我叫林海，我的兴趣爱好还有乒乓球，请更新我的信息"}]}, config=config)
print(response_two["messages"][-1].content)

print("================")

state = agent.get_state(config=config)
print(state)

print("================")

response_three = agent.invoke({"messages": [{"role": "user", "content": "获取我的信息"}]}, config=config)
print(response_three["messages"][-1].content)

print("================")

state = agent.get_state(config=config)
print(state)



