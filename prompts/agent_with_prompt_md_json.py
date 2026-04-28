from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

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

# 定义class,格式化接收输出内容
class CapitalInfo(BaseModel):
    name:str
    location:str
    vibe:str
    economy:str

system_prompt_md="""
# 身份
- 你是一个科幻作家，根据用户的问题来创建一个太空之都

# 指令
- 务必以json格式输出，不要加任何markdown样式

# 示例
user:月球的首都是什么？
assistant:
{
"name":"月华城",
"location":"位于月球表面赤道附近的静海基地遗址附近，依托巨大的穹顶与地下网络构成",
"vibe":"高效,革新",
"economy":"能源开发,量子通信枢纽,尖端生物科技"
}
"""

# 创建Agent
agent_json = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    # middleware=[dynamic_model_selection],
    system_prompt=system_prompt_md,
    # response_format=CapitalInfo
)

# 创建Agent
agent_class = create_agent(
    model=ollama_llm_qwen,
    tools=[get_stock_price, search_news],
    # middleware=[dynamic_model_selection],
    system_prompt="你是一个科幻作家，根据用户的问题来创建一个太空之都",
    response_format=CapitalInfo
)

resp_json = agent_json.invoke(
    {"messages": [HumanMessage(content="金星的首都是什么")]}
)
print(type(resp_json))  # agent调用大模型
print(resp_json["messages"][-1].content)

resp_class = agent_class.invoke(
    {"messages": [HumanMessage(content="金星的首都是什么")]}
)
print(type(resp_class))  # agent调用大模型
print(resp_class)
print(resp_class["structured_response"])
