# llm-crowd-detection

## Install

1. Create a conda enviorenment
```
conda create --name py10-dspy python=3.10
conda activate py10-dspy
```
2. Install requirements

```
conda install -c conda-forge ffmpeg
conda install -c conda-forge jupyterlab
pip install vllm
pip install dspy
```

## Structure

## How to use it

Files in code support experiments with llm and vlm (text only mode, and with image mode). Both have options to run only with one frame or the full frames. Options to run them is as follows:

```
python3 fetch_llm_groups_singleframe.py <filename> <mode> <model> <frame_id> [options]
```

* filename (str): a json which contains the metada
* mode (str): Options are llm, vlm_text, vlm_image
* model (str): Name of the model served by other package. We mainly served with vllm and try models as:
* frame_id (int): Frame index to process  

| Flag              | Type  | Default                      | Description                                                                                                           |
| ----------------- | ----- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--depth_method`  | str   | `naive_3D_60FOV`             | Depth estimation method. Choices: `naive_3D_60FOV`, `naive_3D_110FOV`, `naive_3D_160FOV`, `unidepth_3D`, `detany_3D`. |
| `--prompt_method` | str   | `p1`                         | Prompting strategy for LLM inference. Choices: `baseline1`, `p1`, `p2`, `p3`, `p4`.                                   |
| `--api_base`      | str   | `http://localhost:8000/v1`   | API base URL for model inference.                                                                                     |
| `--api_key`       | str   | `testkey`                    | API authentication key.                                                                                               |
| `--temperature`   | float | `0.6`                        | Sampling temperature (higher = more random, lower = more deterministic).                                              |
| `--max_tokens`    | int   | `32768`                      | Maximum tokens for model output.                                                                                      |
| `--frame_path`    | str   | `VBIG_dataset/videos_frames` | Path to video frames for visualization.                                                                               |
| `--save_image`    | flag  | `False`                      | Enable debug mode and save visualizations.                                                                            |


