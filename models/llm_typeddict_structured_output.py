from typing import TypedDict, Annotated

from llm_init import ollama_llm_qwen

"""
1.Pydantic 输出形式

2.TypedDict 输出形式

3.JsonSchema 输出形式
"""

conversations = [
    {"role": "system", "content": "你是一个电影博主，可以评价电影"},
    {"role": "user", "content": "请介绍下电影:泰坦尼克号"}
]


# 定义结构化类型
class Actor(TypedDict):
    name: Annotated[str, "演员姓名"]
    role: Annotated[int, "演员角色"]

# 2.TypedDict
class Movie(TypedDict):
    title: Annotated[str, "电影的标题"]
    year: Annotated[int, "电影的标题"]
    director: Annotated[str, "电影的标题"]
    rating: Annotated[float, "电影的标题"]
    cast: Annotated[list[Actor], "电影的标题"]

ollama_llm_with_structured = ollama_llm_qwen.with_structured_output(Movie)
resp = ollama_llm_with_structured.invoke(conversations)
print(type(resp))
print(resp)



