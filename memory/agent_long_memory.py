from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

from llm_init import ollama_llm_qwen

"""
一次会话：包含多次对话，HumanMessage
checkpointer可以实现短期记忆



"""


store=InMemoryStore() # 创建长期记忆存储

store.put(
    ("users",), # 命名空间,简单理解成目录
    "user_123",# 键
    {"name":"张三","age":18,"city":"北京","hobby":"旅游"} # 值
)

store.put(
    ("users",), # 命名空间,简单理解成目录
    "user_456",# 键
    {"name":"李四","age":30,"city":"南京","hobby":"足球"} # 值
)

@tool
def get_user_info(runtime:ToolRuntime) -> str | None:
    """
    从长期记忆中获取用户信息
    :param runtime:
    :return:
    """

    # 获取store
    store = runtime.store

    # 获取user_id
    user_id = "user_123"

    # 从长期记忆中获取用户信息
    user_data = store.get(("users",), user_id)

    if user_data:
        print("user_data:",user_data.value)
        value = user_data.value
        return f"姓名:{value['name']}, 年龄:{value['age']}, 城市:{value['city']}, 爱好:{value['hobby']}"
    else:
        return f"用户{user_id}不存在"


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_user_info],
    system_prompt="你是一个用户信息查询助手，你可以调用get_user_info查询用户的信息",
    # checkpionter是把过往的交互过程，保存在内存，或者DB，Redis
    checkpointer=InMemorySaver(),# 短期记忆存储
    store=store
)

config1 = {"configurable": {"thread_id": "session_1"}}
config2 = {"configurable": {"thread_id": "session_2"}}

# agent调用模型，必须是messages的结构体作为入参
response = agent.invoke({"messages": [{"role": "user", "content": "从长期记忆中，获取用户信息"}]}, config=config1)
print(type(response))
print(response)
print(response["messages"][-1].content)

print("================")

response_two = agent.invoke({"messages": [{"role": "user", "content": "从长期记忆中，获取用户信息"}]}, config=config2)
print(response_two["messages"][-1].content)
