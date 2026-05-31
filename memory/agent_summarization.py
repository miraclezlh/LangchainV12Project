from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_model, after_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from llm_init import ollama_llm_qwen

# 将记忆点存储在内存
checkpointer = InMemorySaver()


@tool
def get_weather(city: str) -> str:
    """
    获取天气情况

    :param city:
    :return:
    """
    return f"{city}的天气是晴天"


@before_model
def before_model(state: AgentState, runtime: ToolRuntime) -> dict | None:
    """
    调用大模型前，进行处理的函数

    :param state:
    :param runtime:
    :return:
    """
    print("before model state:", state)
    messages = state["messages"]
    # print("messages",messages)

    return {"messages": messages}


@after_model
def after_model(state: AgentState, runtime: ToolRuntime) -> dict | None:
    """
    调用大模型后，进行处理的函数

    :param state:
    :param runtime:
    :return:
    """
    print("after model state:", state)
    messages = state["messages"]

    return {"messages":messages}


# 创建Agent
# checkpointer是把过往的交互过程，保存在内存，或者DB，Redis
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_weather],
    checkpointer=checkpointer,
    middleware=[
        # 自定义中间件
        before_model,
        # 自定义中间件
        after_model,
        # 系统自带的中间件
        SummarizationMiddleware(
            model=ollama_llm_qwen,
            # 摘要触发条件：当消息数量达到5条的时候，对消息进行摘要总结
            trigger=("messages", 5),
            # 保留最近的2条消息，把前面的消息进行总结
            keep=("messages", 2),
            # 总结用的提示词，约定一些限制规则
            summary_prompt="请摘要以下内容：{messages}"
        )]
)

config = {"configurable": {"thread_id": "session01"}}

# agent调用模型，必须是messages的结构体作为入参
response = agent.invoke({
    "messages": [{"role": "user", "content": "我叫小飞侠，你是谁？"}]
}, config=config)
print(response["messages"][-1].content)
print("================")

response_two = agent.invoke({"messages": [{"role": "user", "content": "今天上海的天气如何?"}]}, config=config)
print(response_two["messages"][-1].content)
print("================")

response_three = agent.invoke({"messages": [{"role": "user", "content": "我的名字叫什么?"}]}, config=config)
print(response_three["messages"][-1].content)
