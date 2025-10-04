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

## Structure
*[To be documented]*

## Usage

### Main Scripts
The project includes two main scripts for LLM/VLM experiments:
- `fetch_llm_groups_singleframe.py` - Process a single frame
- `batch_fetch_llm_groups_singleframe.py` - Batch processing

Both scripts support:
- **LLM and VLM modes** (text-only and image modes)
- **Single frame or full frame processing**

### Command Syntax
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
| `--depth_method`  | str   | `naive_3D_60FOV`             | Depth estimation method: `naive_3D_60FOV`, `naive_3D_110FOV`, `naive_3D_160FOV`, `unidepth_3D`, `detany_3D`         |
| `--prompt_method` | str   | `p1`                         | Prompt strategy: `baseline1`, `p1` (3D), `p2` (3D+Direction), `p3` (3D+Transitive), `p4` (3D+Transitive+Direction)  |
| `--api_base`      | str   | `http://localhost:8000/v1`   | API base URL for model inference                                                                                      |
| `--api_key`       | str   | `testkey`                    | API authentication key                                                                                                |
| `--temperature`   | float | `0.6`                        | Sampling temperature (higher = more random, lower = more deterministic)                                               |
| `--max_tokens`    | int   | `32768`                      | Maximum tokens for model output                                                                                       |
| `--frame_path`    | str   | `VBIG_dataset/videos_frames` | Path to video frames directory for visualization                                                                      |
| `--save_image`    | flag  | `False`                      | Save processed images to disk                                                                                         |

### Example Usage
```bash
python3 fetch_llm_groups_singleframe.py \
  VBIG_dataset/jsons_step5/Cusco_Peru_0003018_clip_002.json \
  vlm_image \
  Qwen/Qwen2.5-VL-72B-Instruct \
  10 \
  --prompt_method baseline1 \
  --save_image
```
```

