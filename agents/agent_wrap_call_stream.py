from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from llm_init import ollama_llm_qwen, ollama_llm_qwen3_4b
from tools.common_tools import search_news, get_stock_price


# 调用大模型之前执行，钩子函数。动态调用，进行附属操作
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """动态修改模型"""
    # print(type(request)) # 类型为：ModelRequest对象
    # print(request)

    message_count = len(request.state["messages"])
    if message_count > 2:
        print("切换大模型为ollama_llm_qwen3_4b")
        model = ollama_llm_qwen3_4b
        # request.override(model=ollama_llm_gemma)
    else:
        print("切换大模型为ollama_llm_qwen")
        model = ollama_llm_qwen
        # request.override(model=ollama_llm_qwen)
    return handler(request.override(model=model))


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    middleware=[dynamic_model_selection],
    system_prompt="你是一个助手，你可以查询公司的股价，以及有关公司的新闻搜索。"
)

# agent调用模型，必须是messages的结构体作为入参
# 流式调用
resp_stream = agent.stream(
    {"messages": [{"role": "user", "content": "苹果公司今天的股价是多少？最近有什么新闻？"}]},
    stream_mode="messages"
)
print(type(resp_stream))  # agent调用大模型，stream的返回类型为<class 'generator'>

for token, metadata in resp_stream:
    if token.content:
        print(token.content, end="", flush=True)
# print(response)
# print(response["messages"][-1].content)
