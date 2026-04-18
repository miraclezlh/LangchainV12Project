from langchain_core.messages import HumanMessage

from llm_init import ollama_llm_qwen
from tools.common_tools import get_weather

# chain为HumanMessage-->AIMessage-->ToolMessage-->AIMessage
messages = [HumanMessage(content="查询北京的天气")]

# 1.给模型，手动绑定工具Tool
ollama_llm_with_tools = ollama_llm_qwen.bind_tools([get_weather])

# 2.模型返回要调用的工具，模型不会直接调用工具
# first-->发送问题给大模型
resp = ollama_llm_with_tools.invoke(messages)
print(type(resp))
print(resp)
# second-->发送问题给大模型
messages.append(resp)

# 3.根据模型返回的调用工具的指令，调用工具
for tool_call in resp.tool_calls:
    # print(type(tool_call))
    # print(tool_call)
    if tool_call['name'] == "get_weather":
        city = tool_call['args']["city"]
        print(type(city))
        print(city)
        # tool_message = get_weather.invoke(city) # 返回的Tool的返回类型
        tool_message = get_weather.invoke(tool_call)  # 返回的是ToolMessage结构体
        print(type(tool_message))
        print(tool_message)
        messages.append(tool_message)
        # print(f"工具调用结果: {weather_info}")

# 4.最后，再次调用大模型LLM
result = ollama_llm_with_tools.invoke(messages)
print(messages)
print(type(result))
print(result)
