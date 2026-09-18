"""
远程运行的MCP server,通信使用streamableHTTP
"""
import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

ollama_llm_qwen = init_chat_model(
    api_key="ollama",
    base_url="http://localhost:11434",
    model='qwen3:8b',
    model_provider='ollama'
)


# 异步方法
async def main():
    # 1.创建client,配置多个MCP Server信息
    client = MultiServerMCPClient(
        {
            # stdio 本地进程
            "math_server": {
                "transport": "stdio",
                "command": "D:\\miniconda3\\envs\\langchain_study\\python.exe",
                "args": ["D:\\MyPythonProjects\\LangchainV12Project\\mcp_part\\01_mcp_demo.py"]

            },
            # streamableHTTP 远程Server
            "weather_server": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )

    # 2.从多个Server中加载工具，一定要await异步
    tools = await client.get_tools()
    print("tools:", tools)

    # 3.创建LangChain Agent
    agent = create_agent(
        model=ollama_llm_qwen,
        tools=tools
    )

    # 4.异步调用，运行agent
    math_result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": "3加上5等于多少"}
        ]
    })

    weather_result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": "北京天气如何"}
        ]
    })

    print("math_result:", math_result)
    print("weather_result:", weather_result)

    print("数学结果:", math_result["messages"][-1].content)
    print("天气结果:", weather_result["messages"][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
