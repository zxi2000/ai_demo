from langchain_core.tools import tool

@tool
def weather_agent(city: str, token: str) -> str:
    """
    查询指定城市的天气信息，需要传入 city（城市名）和 token（用户标识）参数
    
    Args:
        city: 城市名（如：上海、北京）
        token: 用户唯一标识（用于日志或权限校验）
    
    Returns:
        天气信息字符串
    """
    # 这里应该是实际的天气查询逻辑
    # 示例实现：
    weather_data = {
        "上海": "晴，25-32℃",
        "北京": "多云，22-28℃", 
        "深圳": "阵雨，26-31℃"
    }
    
    default_weather = "晴，24-30℃"
    weather_info = weather_data.get(city, default_weather)
    
    print(f"【用户 {token}】{city}天气：{weather_info}")
    return f"【用户 {token}】{city}天气：{weather_info}"