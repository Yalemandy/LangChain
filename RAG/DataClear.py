# 数据清洗

# 1. 创建一个清洗单个字符串的方法

# 2. 创建一个清洗document对象的方法

# 3. 批量调度的方法 处理的是目录下的所有的文件

import re
import unicodedata
from pathlib import Path
from typing import List

from llama_index.core import Document

import util
from config import base_path


def clean_text(text: str, suffix: str) -> str:
    """
    清洗单个字符串

    :param text: 目标文件
    :param suffix:  目标文件类型
    :return:  清洗好的字符串
    """
    # 隐形字符 可能会干扰LLM的判断
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\u200b\u200c\u200d]", "", text)

    # 全角转半角
    text = unicodedata.normalize("NFKC", text)

    # 统一换行符
    text = text.replace("\r\n","\n").replace("\r","\n")

    # 针对特殊类型文件处理
    if suffix.lower() in {".pptx", ".ppt"}:
        text = re.sub(r"^Title:\s*.+\s*$","", text, flags=re.MULTILINE)
        text = re.sub(r"^-{3,}\s*$","",text, flags=re.MULTILINE)
        text = re.sub(r"^\[Speaker Notes\]:\s*","",text,flags=re.MULTILINE)

    # 剔除md标记
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # 去掉 ** 保留里面的字
    text = text.replace("#","")                      # 简单粗暴去掉所有#

    # 去掉多余的空行
    text = re.sub(r"[ \t]*\n[ \t]*","\n", text)   # 第一步： 去掉每行首尾的空白
    text = re.sub(r"\n+","\n", text)              # 第二步： 连续空行压成一个换行

    return  text.strip()

def clean_doc(document: Document) -> Document:
    """
    清洗一个document对象

    :param document: 传进来的一个没有清洗的document对象
    :return:  清洗好的一个document对象
    """
    # 先获取文件类型
    suffix = Path(document.metadata.get("file_path")).suffix
    text = document.text

    cleaned_text = clean_text(text, suffix)

    # 返回清洗后的document对象
    return Document(text=cleaned_text, metadata=document.metadata)

def clean_all_formats(input_dir: Path) -> List[Document]:
    """
    批量清洗
    :param input_dir: 批量处理的路径
    :return: 清洗完成的document列表
    """
    # 将文件解析出来，得到一个Document对象的列表
    dir = util.parse_all_formats(input_dir)

    # 逐个清洗document对象
    cleaned_docs = [clean_doc(doc) for doc in dir]

    # 进行简单的验证
    for i, doc in enumerate(cleaned_docs):
        print(f"\n 第{i + 1}个对象 | 路径: {doc.metadata.get('file_path')}")
        print(doc.text[:200])

    # 返回清洗后的对象列表
    return cleaned_docs

if __name__ == "__main__":
    clean_all_formats(Path(base_path) / "data")