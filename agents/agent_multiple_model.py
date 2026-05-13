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


# 创建Agent
agent = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    # middleware=[dynamic_model_selection],
    system_prompt="你是一个图片识别助手，请你详细描述图片资源。"
)

# agent调用模型，必须是messages的结构体作为入参
# 1.json格式
resp_stream = agent.stream(HumanMessage([
         {"type":"text","content":"描述以下图片的内容"},
         {"type":"image","url":"https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png"},
        ]))
for token in resp_stream:
    if token.content:
        print(token.content, end="", flush=True)
