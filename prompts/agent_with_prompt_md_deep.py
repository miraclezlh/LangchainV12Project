from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import HumanMessage

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


system_prompt_md="""
# 身份
- 你是一个科幻作家，根据用户的问题来创建一个太空之都

# 指令
- 不要返回markdown格式说明，仅返回代码
- 不要输出<think>的思考过程

# 示例
user:月球的首都是什么？
assistant:月华城-深嵌在月球静海环形山中的水晶穹顶都市
"""

# 创建Agent
agent_md = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    # middleware=[dynamic_model_selection],
    system_prompt=system_prompt_md
)

resp_stream_md = agent_md.stream(
    {"messages": [HumanMessage(content="金星的首都是什么")]},
    stream_mode="messages"
)
print(type(resp_stream_md))  # agent调用大模型，stream的返回类型为<class 'generator'>

# 输出的是非结构化字符串
for token, metadata in resp_stream_md:
    if token.content:
        print(token.content, end="", flush=True)
