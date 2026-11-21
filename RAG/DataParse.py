from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import os
# print(os.environ.get("DEEPSEEK_API_KEY"))  # 验证

# 数据解析

# 1. 将待处理的放到一个目录下

# 2. 对数据进行批量读取

# 3. 对拿到的数据进行结果预览

# 4. 结果返回


from llama_index.core import SimpleDirectoryReader

base_path = Path(__file__).parent

# 数据解析函数
def parse_all_formats(input_dir):

    # 创建加载数据的对象
    reader = SimpleDirectoryReader(
        input_dir= input_dir
    )

    # 加载数据
    documents = reader.load_data()

    # 数据预览
    for i, doc in enumerate(documents):
        file_path = doc.metadata.get("file_path")
        text = doc.text[:200]
        print(f"\n, 第{i+1}个对象｜路径：{file_path}")
        print(text)


    return documents

if __name__ == "__main__":
    parse_all_formats(base_path / "data")