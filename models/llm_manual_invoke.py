from llm_init import ollama_llm_qwen
from tools.common_tools import get_weather

# 1.给模型，手动绑定工具Tool    2.大模型直接绑定Tools
ollama_llm_with_tools = ollama_llm_qwen.bind_tools([get_weather])

# 2.模型返回要调用的工具，模型不会直接调用工具
# 直接转入字符串，模型会自动默认，识别为：用户消息HumanMessage
# invoke阻塞式调用，
resp = ollama_llm_with_tools.invoke("查询北京的天气")
print(resp)

# 3.根据模型返回的调用工具的指令，调用工具
for tool_call in resp.tool_calls:
    print(type(tool_call))
    print(tool_call)
    if tool_call['name'] == "get_weather":
        city = tool_call['args']["city"]
        # print(type(city))
        # print(city)
        weather_info_str = get_weather.invoke(city) # 返回的Tool的返回类型，此函数为str字符串
        print(type(weather_info_str)) # str返回类型
        print(weather_info_str) # 直接返回天气情况

        weather_info = get_weather.invoke(tool_call) # 返回的是ToolMessage结构体
        print(type(weather_info))
        print(weather_info)
        # print(weather_info.content)
        print(f"工具调用结果: {weather_info.content}")

