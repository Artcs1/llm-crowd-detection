import os
import sys
import subprocess
from tqdm.auto import tqdm


json_dir  = sys.argv[1]
video_dir = sys.argv[2]
model     = sys.argv[3]

print('JSON_DIR:',  json_dir)
print('VIDEO_DIR:', video_dir)
print('MODEL_DIR:', model)

files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f))]

depth_methods = ['detany_3D'] #['naive_3D_60FOV', 'unidepth_3D', 'detany_3D']
prompt_methods = ['p1'] # , 'p2', 'p3', 'p4']
settings = ['single'] #, 'full']

for setting in settings:
    for file in tqdm(files):
        for depth_method in depth_methods:
            for prompt_method in prompt_methods:                

                cmd = [
                    'python', 'code/fetch_groups_multi.py',
                    file,
                    setting,
                    'llm',
                    model,
                    '--frame_ids', '1',
                    '--depth_method', depth_method,
                    '--prompt_method', prompt_method,
                    '--frame_path', video_dir,
                    '--max_tokens', '24000',
                ]
                
                # print(f'Running command: {" ".join(cmd)}')
                subprocess.run(cmd)
