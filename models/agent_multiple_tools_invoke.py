from langchain.agents import create_agent

from llm_init import ollama_llm_qwen
from tools.common_tools import search_news, get_stock_price

# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。"
)

# agent调用模型，必须是messages的结构体作为入参
response = agent.invoke({"messages": [{"role": "user", "content": "苹果公司今天的股价是多少？最近有什么新闻？"}]})
print(type(response))
print(response)
print(response["messages"][-1].content)

