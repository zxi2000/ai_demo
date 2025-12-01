from langchain_core.tools import tool

@tool
def schedule_agent(date: str) -> str:
    """
    查询或管理用户的日程安排，需要传入 date（日期）和 token（用户标识）参数
    
    Args:
        date: 日期（格式：YYYY-MM-DD）
    
    Returns:
        日程信息字符串
    """
    # 模拟日程查询逻辑
    print(f" {date} 的日程：上午会议，下午开发，晚上学习")
    return f"{date} 的日程：上午会议，下午开发，晚上学习"