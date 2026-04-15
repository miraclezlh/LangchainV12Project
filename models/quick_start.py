from tools.common_tools import get_weather
from langchain.agents import create_agent
from my_llm import ollama_llm, ollama_llm_two

"""
    创建agent，调用工具回答用户问题
"""

# 创建Agent
agent = create_agent(
    model=ollama_llm,
    tools=[get_weather],
    system_prompt="你是一个助手，你可以查询城市的天气。"
)

print(agent)

# 调用Agent
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "查询北京的天气"}]}
)
print(resp)

# 创建Agent
agent_two = create_agent(
    model=ollama_llm_two,
    tools=[get_weather],
    system_prompt="你是一个助手，你可以查询城市的天气。"
)
resp_two = agent_two.invoke(
    {"messages": [{"role": "user", "content": "查询北京的天气"}]}
)
print(resp_two)


