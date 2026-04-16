from langchain_core.messages import HumanMessage

from llm_init import ollama_llm_two
from tools.common_tools import get_stock_price, search_news

# 1. 初始化模型并绑定工具
tools = [get_stock_price, search_news]
model_with_multiple_tools = ollama_llm_two.bind_tools(tools)

message = []
human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
# human_message = HumanMessage(content="比较一下微软和苹果的股价")
# human_message = HumanMessage(content="腾讯最近有什么重大新闻？")
# human_message = HumanMessage(content="海水为什么是咸的？")
message.append(human_message)

print(message)

# 2. 工具调用
while True:
    response = model_with_multiple_tools.invoke(message)

    message.append(response)

    print(message)

    # print(response)
    # print(response.tool_calls)
    # 如果有调用工具，处理工具调用响应
    # 3.开发者根据模型的响应，调用工具并获取结果
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_stock_price":
                stock_result = get_stock_price.invoke(tool_call)
                print("stock_result", stock_result)
                message.append(stock_result)
            if tool_call["name"] == "search_news":
                news_result = search_news.invoke(tool_call)
                print("news_result", news_result)
                message.append(news_result)
    else:
        print("没有工具调用，直接返回答案")
        break

print("response", response)
print(response.content)
