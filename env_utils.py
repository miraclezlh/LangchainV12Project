import os

from dotenv import load_dotenv

# 从env加载环境变量,override=True,确保.env文件优先
load_dotenv(override=True)

# 从配置文件读取信息
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


