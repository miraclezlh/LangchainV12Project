from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from llm_init import ollama_llm_two
from tools.common_tools import search_news, get_stock_price

# 创建Agent
agent = create_agent(
    model=ollama_llm_two,
    tools=[get_stock_price, search_news],
    system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。"
)

# human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
response = agent.invoke( {"messages": [{"role": "user", "content": "苹果公司今天的股价是多少？最近有什么新闻？"}]})
response = agent.invoke(human_message)
print(type(response))
print(response)
