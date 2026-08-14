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

# 定义结构化输出 JsonSchema
json_schema_def = {
    "title": "MovieInfo",
    "description": "电影信息",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "电影的标题"},
        "year": {"type": "integer", "description": "电影的年份"},
        "director": {"type": "string", "description": "电影的导演"},
        "rating": {"type": "number", "description": "电影的评分"},
        "cast": {
            "description": "电影演员列表",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "演员姓名"},
                    "role": {"type": "string", "description": "演员角色"}
                },
                "required": ["name", "role"]
            },
        }
    },
    "required": ["title", "year", "director", "rating"]
}

# 3.让大模型绑定JsonSchema
ollama_llm_with_structured = ollama_llm_qwen.with_structured_output(json_schema_def)
resp = ollama_llm_with_structured.invoke(conversations)
print(type(resp))
print(resp)
