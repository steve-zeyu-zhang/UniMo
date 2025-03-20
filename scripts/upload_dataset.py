from huggingface_hub import HfApi

api = HfApi()


api.upload_file(
    path_or_fileobj="dataset/AnimalML3D.zip",  # 本地 zip 文件路径
    path_in_repo="AnimalML3D.zip",          # 在仓库中的路径和文件名
    repo_id="sims81816/AnimalML3D",   # 数据集仓库的 ID
    repo_type="dataset"                     # 指定仓库类型为 dataset
)