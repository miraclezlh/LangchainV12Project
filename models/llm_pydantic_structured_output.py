
from pydantic import BaseModel, Field
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
class Actor(BaseModel):
    name: str = Field(description="演员姓名")
    role: int = Field(description="演员角色")


# 1.Pydantic
class Movie(BaseModel):
    title: str = Field(description="电影的标题")
    year: int = Field(description="电影的年份")
    director: str = Field(description="电影的导演")
    rating: float = Field(description="电影的评分")
    cast: list[Actor] = Field(description="电影演员列表")


ollama_llm_with_structured = ollama_llm_qwen.with_structured_output(Movie)
resp = ollama_llm_with_structured.invoke(conversations)
print(type(resp))
print(resp)




