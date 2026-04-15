from langchain.agents import create_agent
from langchain_core.tools import tool

from my_llm import ollama_llm, ollama_llm_two

"""创建agent，调用工具回答用户问题"""


# 函数定义
@tool
def get_weather(city: str) -> str:
    # 模拟天气查询
    """获取给定城市的天气。"""
    return f"{city} 天气晴朗！"


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


