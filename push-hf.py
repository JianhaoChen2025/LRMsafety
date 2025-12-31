from huggingface_hub import HfApi

api = HfApi()

# 设置参数
local_folder = "./dataset"  # 服务器上数据集的路径
repo_id = "chenjacike220/LRMsafety"        # 例如: "zhangsan/my-cool-dataset"

# 开始上传
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",  # 必须指定为 dataset，否则默认为 model
    commit_message="Initial dataset upload"
)