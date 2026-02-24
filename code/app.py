from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Start Flask app with custom dataset folder")
parser.add_argument(
    "-d", "--dataset",
    type=str,
    required=True,
    help="Path to the dataset folder (e.g., predictions/BLENDER/results)"
)
args = parser.parse_args()


app = Flask(__name__)
ROOT_PATH = Path(__file__).parent / args.dataset

def get_subfolders(path):
    return [f.relative_to(ROOT_PATH).as_posix() for f in path.iterdir() if f.is_dir()]

def get_images(path):
    return [f.name for f in path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

@app.route("/")
def index():
    vlms = get_subfolders(ROOT_PATH)
    return render_template("index.html", vlms=vlms)

@app.route("/get_modes")
def get_modes():
    model = request.args.get("model")
    if not model:
        return jsonify([])
    model_path = ROOT_PATH / model
    if not model_path.exists():
        return jsonify([])
    modes = [f.name for f in model_path.iterdir() if f.is_dir()]
    return jsonify(modes)

@app.route("/get_extra_folders")
def get_extra_folders():
    model = request.args.get("model")
    mode = request.args.get("mode")
    if not model or not mode:
        return jsonify([])
    mode_path = ROOT_PATH / model / mode
    if not mode_path.exists():
        return jsonify([])
    extras = [f.name for f in mode_path.iterdir() if f.is_dir()]
    return jsonify(extras)

@app.route("/get_methods")
def get_methods():
    model = request.args.get("model")
    mode = request.args.get("mode")
    extra = request.args.get("extra")
    if not model or not mode or not extra:
        return jsonify([])
    extra_path = ROOT_PATH / model / mode / extra 
    if not extra_path.exists():
        return jsonify([])
    methods = [f.name for f in extra_path.iterdir() if f.is_dir()]
    return jsonify(methods)

@app.route("/get_inner_paths_by_method")
def get_inner_paths_by_method():
    model = request.args.get("model")
    mode = request.args.get("mode")
    extra = request.args.get("extra")
    method = request.args.get("method")
    if not all([model, mode, extra, method]):
        return jsonify([])
    method_path = ROOT_PATH / model / mode / extra / method/ "15"
    inner_paths = []
    if method_path.exists():
        for inner in method_path.iterdir():
            if inner.is_dir():
                rel_path = "/".join([mode, extra, method, "15", inner.name])
                inner_paths.append(rel_path)
    inner_paths.sort()
    return jsonify(inner_paths)

@app.route("/get_images_by_inner")
def get_images_by_inner():
    inner_path = request.args.get("inner")
    result = {}
    for model_folder in get_subfolders(ROOT_PATH):
        folder = ROOT_PATH / model_folder / Path(inner_path)
        print(folder)
        print(inner_path)
        if folder.exists():
            result[model_folder] = get_images(folder)
    return jsonify(result)

@app.route("/images/<path:filepath>")
def serve_image(filepath):
    return send_from_directory(ROOT_PATH, filepath)

if __name__ == "__main__":
    app.run(port=8888, debug=True)

