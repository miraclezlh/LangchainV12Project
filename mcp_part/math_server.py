"""
客户端以子进程形式启动，通过标准输入/输出协议(stdio)
"""
from fastmcp import FastMCP

mcp_server = FastMCP("math_server")


# 定义工具
@mcp_server.tool()
def add(a: int,b: int):
    """
    计算两个数之和
    Args:
        a: 第一个数
        b: 第二个数
    Returns:
        返回两数的和
    """
    return a+b

# 定义工具
@mcp_server.tool()
def multiply(a: int,b: int):
    """
    计算两个数之和
    Args:
        a: 第一个数
        b: 第二个数
    Returns:
        返回两数的积
    """
    return a*b

if __name__ == '__main__':
    # 后续以子进程的方式启动
    mcp_server.run(transport="stdio")


