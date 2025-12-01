from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

#取环境变量
load_dotenv()

def get_qwen_llm():

    """
        获取百炼qwen大模型
    """
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen-plus"
    api_key = "sk-3a299b62456a426f98a59e194d9297a6"
    llm = ChatOpenAI(base_url=base_url, model=model, api_key = api_key)
    return llm


def get_llm(base_url, model, api_key, temperature):

    """
        获取 实际大模型
    """
    #base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #model = "qwen-plus"
    #api_key = "sk-3a299b62456a426f98a59e194d9297a6"
    llm = ChatOpenAI(base_url=base_url, model=model, api_key = api_key, temperature = temperature)
    return llm



import uuid
from sentence_transformers import SentenceTransformer
import os  # 用于处理路径（确保跨平台兼容）


def get_local_embeddings(
        #model_name: str = "BAAI/bge-small-zh-v1.5"  # 改为 BGE 中文模型
        local_model_path: str = "./models/bge-small-zh-v1_5" 
    ):

    """
    用 bge-small-zh-v1.5 本地生成向量（中文效果优于 MiniLM，免费）
    :param texts: 纯文本列表（支持中文）
    :param model_name: BGE 中文模型名称（固定）
    :return: 向量列表（768 维）
    """

     # 转换为绝对路径（避免因运行目录不同导致路径错误）
    abs_model_path = os.path.abspath(local_model_path)
    
    # 校验本地模型目录是否存在
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"本地模型目录不存在：{abs_model_path}\n请确认模型已下载到该路径")
    
    # 校验模型文件是否完整（至少包含核心文件）
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(abs_model_path, f))]
    if missing_files:
        raise FileNotFoundError(f"本地模型缺少核心文件：{missing_files}\n请重新下载模型")

    # 加载模型（首次自动下载，后续复用缓存）
    try:
        model = SentenceTransformer(abs_model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{str(e)}")


def get_HuggingFace_embeddings(
    local_model_path: str = "./models/bge-small-zh-v1_5"
) -> HuggingFaceEmbeddings:
    """
    用 HuggingFaceEmbeddings 包装 SentenceTransformer 本地模型，适配 LangChain 接口
    :param local_model_path: 本地模型目录路径
    :return: LangChain 兼容的嵌入函数（带 embed_documents/embed_query 方法）
    """
    # 转换为绝对路径（避免因运行目录不同导致路径错误）
    abs_model_path = os.path.abspath(local_model_path)
    
    # 校验本地模型目录是否存在
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"本地模型目录不存在：{abs_model_path}\n请确认模型已下载到该路径")
    
    # 校验模型文件是否完整（至少包含核心文件）
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(abs_model_path, f))]
    if missing_files:
        raise FileNotFoundError(f"本地模型缺少核心文件：{missing_files}\n请重新下载模型")

    # 关键：用 HuggingFaceEmbeddings 包装本地 SentenceTransformer 模型
    # 自动实现 embed_documents 和 embed_query 方法
    embedding_function = HuggingFaceEmbeddings(
        model_name=abs_model_path,  # 传入本地模型路径
        model_kwargs={"device": "cpu"},  # 可选：指定设备（cpu/gpu，gpu需安装CUDA）
        encode_kwargs={"normalize_embeddings": True}  # 可选：归一化向量（推荐开启，提升检索效果）
    )
    return embedding_function

def get_qwen_embed(model, dashscope_api_key):   
    """
      获取 qwen embed模型
    """
    embed = DashScopeEmbeddings(model=model, dashscope_api_key = dashscope_api_key)

    return embed


def get_qwen_rerank():
    """
        获取 qwen rerank 模型
    """
    rerank = DashScopeRerank()
    return rerank



if __name__ == "__main__":
    embed = get_bge_embeddings() #get_qwen_embed()
    #print(embed.embed_query("你好"))
    print(embed.encode_query("你好"))