# LLM Crowd Detection

## Installation

### 1. Create a Conda Environment
```bash
conda create --name py10-dspy python=3.10
conda activate py10-dspy
```

### 2. Install Requirements
```bash
conda install -c conda-forge ffmpeg
conda install -c conda-forge jupyterlab
pip install vllm
pip install dspy
pip install pandas
```

## Pre-steps

- Serve the llm or vllm with vllm package (Examples in vllm_commands.txt). The name of the served model and args.model should be the same.
```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --api-key testkey --tensor-parallel-size 4
```

## Usage

### Main Scripts
The project includes four main scripts for LLM/VLM experiments:
- `fetch_llm_groups_singleframe.py` - Process a single frame
- `batch_fetch_llm_groups_singleframe.py` - Batch processing for a single frame
- `fetch_llm_groups_full_singleframe.py` - Process the video
- `batch_fetch_llm_groups_full_singleframe.py` - Batch processing for the whole video

Both scripts support:
- **LLM and VLM modes** (text-only and image modes)
- **Single frame or full frame processing**

### Command Syntax (Single frame)

```bash
python3 fetch_llm_groups_singleframe.py <filename> <mode> <model> <frame_id> [options]
```

### Required Arguments
| Argument   | Type | Description                                    |
|------------|------|------------------------------------------------|
| `filename` | str  | Path to JSON file containing metadata          |
| `mode`     | str  | Processing mode: `llm`, `vlm_text`, `vlm_image`|
| `model`    | str  | Name of the model served (typically via vLLM)  |
| `frame_id` | int  | Frame index to process                         |

### Optional Arguments
| Flag              | Type  | Default                      | Description                                                                                                           |
|-------------------|-------|------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `--depth_method`  | str   | `naive_3D_60FOV`             | Depth estimation method: `naive_3D_60FOV`, `naive_3D_110FOV`, `naive_3D_160FOV`, `unidepth_3D`, `detany_3D`           |
| `--prompt_method` | str   | `p1`                         | Prompt strategy: `baseline1` (only available with vlm_image), `p1` (3D), `p2` (3D+Direction), `p3` (3D+Transitive), `p4` (3D+Transitive+Direction)    |
| `--api_base`      | str   | `http://localhost:8000/v1`   | API base URL for model inference                                                                                      |
| `--api_key`       | str   | `testkey`                    | API authentication key                                                                                                |
| `--temperature`   | float | `0.6`                        | Sampling temperature (higher = more random, lower = more deterministic)                                               |
| `--max_tokens`    | int   | `32768`                      | Maximum tokens for model output                                                                                       |
| `--frame_path`    | str   | `VBIG_dataset/videos_frames` | Path to video frames directory for visualization                                                                      |
| `--save_image`    | flag  | `False`                      | Save processed images to disk (by defaul json results are saved without this option)                                  |

### Example Usage
```bash
python3 fetch_llm_groups_singleframe.py \
  VBIG_dataset/jsons_step5/Cusco_Peru_0003018_clip_002.json \
  vlm_image \
  Qwen/Qwen2.5-VL-72B-Instruct \
  10 \
  --prompt_method p1 \
  --save_image
```

**Hint**: The code assume that the image is located in "{args.frame_path}/{args.filename[:-5]}/{str(args.frame_id).zfill(5)}.jpeg". For the last example that will be equal to VBIG_dataset/videos_frames/Cusco_Peru_0003018_clip_002/00010.jpeg


### Command Syntax (Batch Single frame)
This command is similar to the previous one just consider that now you send a folder that should contain a bunch of .json. Same previous assumptions about the location of the image/video.

### Example Usage
```bash
python3 batch_fetch_llm_groups_singleframe.py \
  VBIG_dataset/jsons_step5/ \
  vlm_image \
  Qwen/Qwen2.5-VL-72B-Instruct \
  10 \
  --prompt_method p1 \
  --save_image
```


### Command Syntax (Multiple frame)
Here we sent a only json file. Same previous assumptions about the location of the image/video. Here prompt strategy are only `p1` and `baseline1` (for vlm_image only). Other input variable logic is the same. 

### Example Usage
```bash
python3 fetch_llm_groups_full_singleframe.py \
  VBIG_dataset/jsons_step5/Cusco_Peru_0003018_clip_002.json \
  vlm_image \
  Qwen/Qwen2.5-VL-72B-Instruct \
  10 \
  --prompt_method p1 \
  --save_image
```


### Command Syntax (Batch Multi frame)
This command is similar to the previous one just consider that now you send a folder that should contain a bunch of .json. Same previous assumptions about the location of the image/video. Here prompt strategy are only `p1` and `baseline1` (for vlm_image only). Other input variable logic is the same. 


### Example Usage
```bash
python3 batch_fetch_llm_groups_full_singleframe.py \
  VBIG_dataset/jsons_step5/ \
  vlm_image \
  Qwen/Qwen2.5-VL-72B-Instruct \
  10 \
  --prompt_method p1 \
  --save_image
```
