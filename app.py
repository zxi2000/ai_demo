import os
import sys
import importlib.util
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse

from typing import Dict, Any, Optional, List

#ignore warning
import warnings
warnings.filterwarnings("ignore")
from llm import invoke


# 加载环境变量
load_dotenv()
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('APP_PORT', '8000'))

app = FastAPI(
    title="llm service", 
    description="LLM with Agents",
    version="1.0.0"
)

# 2. 注册全局异常处理器，覆盖默认的HTTPException处理
# 如果没有这段，在调用接口请求时，返回值会出出一层 detail:{}
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # 直接返回exc.detail中的内容，不包裹detail字段
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail  # 这里直接使用detail作为响应体
    )


@app.post("/llm/invoke", summary="调用大模型")
async def llm_invoke(
    body: Dict[str, Any] 
):
    """
    调用大模型
    """
    try:
        result = invoke(body)
        return {"success":True, "obj": result}
        
    except HTTPException:
        raise
    except Exception as e:
       
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={  
                "success": False,
                "msg": f"执行失败: {str(e)}"
            }
        )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=APP_HOST, 
        port=APP_PORT
    )