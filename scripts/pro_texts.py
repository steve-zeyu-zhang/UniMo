import json
import os

# 输入文件和输出目录（请根据实际情况修改）
input_file = "dataset/uni/uni/Coyote_20240821_230345_llama13B.txt"
output_folder = "dataset/uni/uni/texts"
os.makedirs(output_folder, exist_ok=True)

prefix = "/data/zbz5349/ICLR_2024/Motion_Avatar/data/zoo_300k/zoo_300k/Dog"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    video_path = item.get("video_path")

    video_path = video_path[len(prefix):]
    file_name = video_path.replace("/", "_")
    file_name = file_name[:-4] + ".txt"
    
    output_path = os.path.join(output_folder, file_name)
    
    content = item.get("predict")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(content)
    
    print(f"Created: {output_path}")

print("所有 txt 文件已生成。")
