from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

from llm_init import ollama_llm_qwen

"""
一次会话：包含多次对话，HumanMessage
checkpointer可以实现短期记忆

"""

DB_URI = "mysql+pymysql://root:111111@localhost:3306/langchain_db?useSSL=false&useUnicode=true&characterEncoding=utf8mb4"

"""
with 等于 try......catch......finally
自动释放资源‌：如文件、数据库连接、线程锁等，在 with 代码块结束后自动关闭或释放。
‌异常安全‌：无论代码块是否抛出异常，都会执行清理操作。
‌代码简洁‌：替代冗长的 try...finally 结构，提升可读性。
"""
with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    """
    自动创建表
    """
    checkpointer.setup()

    # 创建Agent
    agent = create_agent(
        model=ollama_llm_qwen,
        tools=[],
        # checkpionter是把过往的交互过程，保存在内存，或者DB，Redis
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "session_1"}}

    # agent调用模型，必须是mesages的结构体作为入参
    response = agent.invoke({"messages": [{"role": "user", "content": "我叫小飞侠，你是谁？"}]}, config=config)
    print(type(response))
    print(response)
    print(response["messages"][-1].content)

    print("================")

    response_two = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=config)
    print(response_two["messages"][-1].content)
