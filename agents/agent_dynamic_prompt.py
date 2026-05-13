from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from llm_init import ollama_llm_qwen
from tools.common_tools import search_news, get_stock_price

"""
@dynamic_prompt 是一个用于 ‌动态生成系统提示词（system prompt）‌ 的中间件装饰器
核心要点
‌作用时机‌：每次模型推理前执行，动态替换系统提示词。
‌覆盖规则‌：完全覆盖创建 Agent 时设置的静态 system_prompt。
‌作用范围‌：仅对当前模型调用有效，不持久化到 Agent 配置中。
"""
@dynamic_prompt
def dynamic_prompt_support(request: ModelRequest) -> str:
    if request.runtime.context["user_tp"] == "VIP":
        prompt = "你是一个专业的股票助手。在回答用户回答前，请称呼'用户为VIP您好:',再加上股票的新闻"
    else:
        prompt = "你是一个普通的股票助手。在回答用户回答前，请称呼'用户您好:',再加上股票的新闻"
    return prompt


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    middleware=[dynamic_prompt_support],
    system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。"
)

# agent调用模型，必须是messages的结构体作为入参
response = agent.invoke(
    {"messages": [{"role": "user", "content": "苹果公司今天的股价是多少？最近有什么新闻？"}]},
    # context={"user_tp": "normal"},
    context={"user_tp": "normal"}
)
print(type(response))  # agent调用大模型，返回类型为dict,<class 'dict'>
print(response)
print(response["messages"][-1].content)
