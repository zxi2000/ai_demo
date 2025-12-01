from langchain_core.tools import tool
import datetime

@tool
def getcurrdate_agent() -> str:
    """
    得到当前时间
    
    Args:
        无
    
    Returns:
        得到返回时间，时间格式默认为"%Y-%m-%d %H:%M:%S
    """
    # 模拟日程查询逻辑
    current_time = datetime.datetime.now()
    return f"当前时间{current_time}"
    return current_time.strftime('%Y-%m-%d %H:%M:%S')
    