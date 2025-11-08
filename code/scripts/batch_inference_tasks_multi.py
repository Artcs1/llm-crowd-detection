import os
import sys
import subprocess
from tqdm.auto import tqdm

# <json path> <frame dir> <vl model> 
#/home/pcascanteb/workspace/llm-crowd-detection/results/Qwen2.5-72B-Instruct/llm/detany_3D/p1 /home/pcascanteb/data/VBIG_dataset/videos_frames Qwen/Qwen2.5-VL-72B-Instruct
#/home/pcascanteb/workspace/llm-crowd-detection/results/Qwen2.5-32B-Instruct/llm/detany_3D/p1/ /home/pcascanteb/data/VBIG_dataset/videos_frames baidu/ERNIE-4.5-VL-28B-A3B-PT
json_dir = sys.argv[1]
video_dir = sys.argv[2]
model = sys.argv[3]

files = [os.path.join(json_dir, f, f+'.json') for f in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, f))]

for file in tqdm(files):
    print(f'Processing {file}')
    cmd = [
        'python', 'code/inference_tasks_multi.py',
        file,
        model,
        '--frame_ids', '1',
        '--frame_path', video_dir,
        '--max_tokens', '24000',
    ]
    subprocess.run(cmd)
