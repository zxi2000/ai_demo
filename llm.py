from models import  get_qwen_llm 

from typing import Dict, Any, Optional, List, Union
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage

from langchain.chat_models import init_chat_model

import importlib
from langchain_core.tools import Tool
import importlib.util


from langchain.messages import ToolMessage



#从dict里动态加载成tools
def load_dynamic_tools(agents_config: List[Dict[str, str]]) -> List[Tool]:
    """
    根据agents配置动态加载工具
    Args:
        agents_config: 格式如 [{"name":"weather_agent","description":"获取城市天气"}]
    Returns:
        加载后的Tool列表
    """
    tools = []
    for agent in agents_config:
        try:
            # 1. 解析配置
            agent_name = agent["name"]
            agent_desc = agent["description"]

            
            # 2. 动态导入模块和工具函数（name为函数名，模块名=name+".py"）
            # 因为模块文件是 weather_agent.py，导入时用 import weather_agent
            module_name =  agent_name  
            module = importlib.import_module(module_name)
            agent_func = getattr(module, agent_name)  # 从模块中获取同名函数
            
            # 确保工具已正确初始化
            if hasattr(agent_func, 'name') and hasattr(agent_func, 'run'):
                tools.append(agent_func)
                #print(f"成功加载工具: {agent_func.name} - {agent_func.description}")
            else:
                print(f"警告: 工具 {agent_name} 未正确初始化")
        except (ImportError, AttributeError) as e:
            print(f"加载工具 {agent_name} 失败: {e}")
    
    return tools



#调用 
#token代表用户token信息，body代表其它内容
def invoke(body: Dict[str, Any] ):


    #得到大模型
    llm = get_qwen_llm()

     #根据agent动态创建tools
    tools = load_dynamic_tools(body["agents"])

    messages = body.get('messages', [])  # Agent 要求输入是 messages 列表（格式：[{"role": "user", "content": "问题"}]）
 
    # else:
    agent = create_agent(
        model=llm,
        tools=tools
    )

    config: RunnableConfig = {"configurable": {"thread_id": 1}}

   
    #返回最后一条消息
    raw_response = agent.invoke({"messages":messages},config)
    return raw_response["messages"][-1]



# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":

    #根据传入的json串来调用大模型，这里有定义了3个tool
    #如果把getcurrdate_agent 这一段从里面json串里去除，则会看到今天的日期是大模型的建设日期 2023-10-04 的日程：上午会议，下午开发，晚上学习
    #需要修改message里的content,可以改不同的提示词
    test_body = {
        "agents": [
            {
                "name": "weather_agent",
                "description": "获取指定城市的天气信息"
            },
            {
                "name": "schedule_agent",
                "description": "查询用户指定日期的日程安排"
            },
            {
                "name": "getcurrdate_agent",
                "description": "得到当前时间或日期"
            }
        ],
        "temperature":"0.5",
        "messages":[{"role":"human","content":"今天上海天气怎么样，日程有什么安排"}]
    }

    
    #得到大模型
    message = invoke(test_body)
    print(message)