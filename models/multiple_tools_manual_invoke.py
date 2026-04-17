from langchain_core.messages import HumanMessage

from llm_init import ollama_llm_two
from tools.common_tools import get_stock_price, search_news

# 1. 初始化模型并绑定工具
tools = [get_stock_price, search_news]
model_with_multiple_tools = ollama_llm_two.bind_tools(tools)

message = []    # 手搓上下文记忆列表
human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
# human_message = HumanMessage(content="比较一下微软和苹果的股价")
# human_message = HumanMessage(content="腾讯最近有什么重大新闻？")
# human_message = HumanMessage(content="海水为什么是咸的？")
message.append(human_message)
# 第一次只包含：用户的问题HumanMessage
print(message)
print("==================")

count = 0
# 2. 工具调用,迭代循环
while True:
    count += 1

    response = model_with_multiple_tools.invoke(message)
    # 第二次包含：用户的问题HumanMessage + AI返回的结果AIMessage（包含待调用的工具列表）
    message.append(response)

    print(message)
    print("==================")

    print(type(response))
    print("response:",response)     # 打印返回的AIMessage的信息
    print(response.tool_calls)  # 打印返回的Tool工具列表
    print("==================")


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

print("=====================")
print("response:", response)
print(response.content)
