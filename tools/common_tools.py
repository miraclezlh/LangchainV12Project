from langchain_core.tools import tool

# 函数定义
@tool
def get_weather(city: str) -> str:
    # 模拟天气查询
    """获取给定城市的天气。"""
    return f"{city} 天气晴朗！"