from llama_index.core import SimpleDirectoryReader
import re
import unicodedata
from pathlib import Path
from typing import List

from llama_index.core import Document

from DataClear import clean_doc


# 数据解析函数
def parse_all_formats(input_dir):
    """
    数据解析
    :param input_dir: 文件路径
    :return:  一个document数组
    """

    # 创建加载数据的对象
    reader = SimpleDirectoryReader(
        input_dir= input_dir
    )
    # 加载数据
    documents = reader.load_data()

    return documents

def clean_all_formats(input_dir: Path) -> List[Document]:
    """
    批量清洗
    :param input_dir: 批量处理的路径
    :return: 清洗完成的document列表
    """
    # 将文件解析出来，得到一个Document对象的列表
    dir = parse_all_formats(input_dir)

    # 逐个清洗document对象
    cleaned_docs = [clean_doc(doc) for doc in dir]

    # 返回清洗后的对象列表
    return cleaned_docs