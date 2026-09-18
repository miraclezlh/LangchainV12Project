"""
远程运行的MCP server,通信使用streamableHTTP
"""
from fastmcp import FastMCP

mcp_server = FastMCP("weather_server")

# 定义工具
@mcp_server.tool()
def get_weather(city: str):
    weather_data={
        "上海":"晴朗",
        "北京":"下雨",
        "广州":"暴风雪"
    }
    return weather_data.get(city)


if __name__ == '__main__':
    # 后续以streamable-http的方式启动,默认端口是8000
    mcp_server.run(transport="streamable-http")


