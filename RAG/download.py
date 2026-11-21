#模型下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    'BAAI/bge-m3',
    cache_dir='afeng./model/cache',
)

print("bge-m3模型下载ok")