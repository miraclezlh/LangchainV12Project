
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

"""
    创建各类LLM大模型
"""

# 针对每一种模型供应商，不一样，不推荐
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
)

# Model Class单独创建llm客户端
ollama_llm = ChatOllama(
    base_url="http://localhost:11434",
    model='qwen3:8b'
)

# langchain整合了初始化模型，init_chat_model
ollama_llm_two = init_chat_model(
    api_key="ollama",
    base_url="http://localhost:11434",
    model='qwen3:8b',
    model_provider='ollama'
)

# @tool
# def get_weather(city: str) -> str:
#     # 函数：返回天气
#     return f"{city} 天气晴朗!"
#
# deepseek_llm.bind_tools("北京的天气怎么样?")




