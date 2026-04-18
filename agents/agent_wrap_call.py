from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from llm_init import ollama_llm_qwen, ollama_llm_gemma
from tools.common_tools import search_news, get_stock_price


# 调用工具前后执行，钩子函数
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """动态修改模型"""
    print(type(request))
    print(request)

    message_count = len(request.state["messages"])
    if message_count > 2:
        request.override(model=ollama_llm_gemma)
    else:
        request.override(model=ollama_llm_qwen)
    return handler(request)


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    middleware=[dynamic_model_selection],
    system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。"
)

# agent调用模型，必须是mesages的结构体作为入参
response = agent.invoke({"messages": [{"role": "user", "content": "苹果公司今天的股价是多少？最近有什么新闻？"}]})
print(type(response))
print(response)
print(response["messages"][-1].content)
