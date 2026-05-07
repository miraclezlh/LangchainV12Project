from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from llm_init import ollama_llm_qwen

"""
一次会话：包含多次对话，HumanMessage
checkpointer可以实现短期记忆



"""
# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[],
    # system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。",
    # checkpionter是把过往的交互过程，保存在内存，或者DB，Redis
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "session_1"}}

# agent调用模型，必须是mesages的结构体作为入参
response = agent.invoke({"messages": [{"role": "user", "content": "我叫小飞侠，你是谁？"}]}, config=config)
print(type(response))
print(response)
print(response["messages"][-1].content)

print("================")

response_two = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=config)
print(response_two["messages"][-1].content)
