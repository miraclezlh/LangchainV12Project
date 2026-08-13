from langchain_core.messages import SystemMessage, HumanMessage

from llm_init import ollama_llm_qwen
from tools.common_tools import get_weather

"""
1.invoke 同步阻塞调用

2.stream 流式调用

3.batch invoke 批量调用
"""

# 1.给模型，手动绑定工具Tool    2.大模型直接绑定Tools
ollama_llm_with_tools = ollama_llm_qwen.bind_tools([get_weather])

# 2.模型返回要调用的工具，模型不会直接调用工具
# 直接转入字符串，模型会自动默认，识别为：用户消息HumanMessage

# 一.invoke调用
resp = ollama_llm_with_tools.invoke("查询北京的天气")
print(type(resp))
print(resp)

print("====================")

conversations = [
    {"role":"system","content":"你是一个翻译助手，可以将汉语翻译成英语"},
    {"role":"user","content":"我喜欢打乒乓球"}
]

resp_two = ollama_llm_with_tools.invoke(conversations)
print(type(resp_two))
print(resp_two)
print(resp_two.content)

print("====================")

conversations_two = [
    SystemMessage(content="你是一个翻译助手，可以将汉语翻译成英语"),
    HumanMessage(content="我喜欢打乒乓球")
]

resp_three = ollama_llm_with_tools.invoke(conversations_two)
print(type(resp_three))
print(resp_three)
print(resp_three.content)

print("====================")

# 二.stream 流式调用
resp_stream = ollama_llm_with_tools.stream(conversations_two)
for chunk in resp_stream:
    # AIMessageChunk
    # print(type(chunk))
    # print(chunk)
    print(chunk.content,end="",flush=True)

print("====================")

conversations_three = [
    "请你介绍一下自己",
    "飞机为什么会飞",
    "什么是大模型"
]

# 三.batch invoke,LLM都完成回答后,再返回给client(等待全部完成按原序返回列表)
resp_batch = ollama_llm_with_tools.batch(conversations_three)
for item in resp_batch:
    print(item)

print("====================")

# 三 .batch_as_completed 按任务完成顺序即时产出(索引，结果)元组‌
resp_batch = ollama_llm_with_tools.batch_as_completed(conversations_three)
for item in resp_batch:
    print(item)






