from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

#取环境变量
load_dotenv()

# 加载环境变量
API_KEY = os.getenv('OPENAI_API_KEY', 'XXX')

def get_qwen_llm():

    """
        获取百炼qwen大模型
    """
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen-plus"
    api_key = API_KEY
    llm = ChatOpenAI(base_url=base_url, model=model, api_key = api_key)
    return llm


