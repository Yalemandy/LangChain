import os

import pandas

import config

source_dir = config.medical_all_data_path
target_dir = config.medical_data_path

# 随机抽样的比例
sample_ratio = 0.001

# 创建简化数据的目录
os.makedirs(target_dir, exist_ok=True)\

# 列出所有目录下的所有文件
for filename in os.listdir(source_dir):
    # 只处理csv
    if filename.lower().endswith(".csv"):
        csv_path = os.path.join(source_dir, filename)
        out_path = os.path.join(target_dir, filename)

        try:
            df = pandas.read_csv(csv_path)
            total_rows = len(df)

            if total_rows == 0:
                print(f"跳过{filename} 文件为空")
                continue

            # 原始数据的行数小于1000的时候，至少要取一条
            if total_rows * sample_ratio < 1:
                samples_num = 1
            samples_num = int(total_rows * sample_ratio)

            df_sample = df.sample(n=samples_num, random_state=42)

            # 取到的数据需要保存到csv里面
            df_sample.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"完成{filename} 总{total_rows}行， 抽取了 {samples_num}行")

        except Exception as e:
            print(f"简化{filename}失败 {str(e)}")