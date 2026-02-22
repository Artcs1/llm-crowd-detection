from utils import *
from prompts import *
from tqdm.auto import tqdm

def inference_task(lm, dspy_module, img_path, gbox):
    preds = dspy_module(image=dspy.Image.from_file(img_path), bbox=gbox)

    if lm.history:
        hist = lm.history[-1]
        hist['messages'][1]['content'][1]['image_url'] = None # Remove image data from history for brevity
        hist_str = str(hist)
    else:
        hist_str = ""
    return preds.output, hist_str


def parse_args_inference_tasks():
    # i want to pass an argument to divide res_json['annotations'] into N parts and run inference on each part separately
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('dir', type=str, help='Path to Folder containing JSON file with groups')
    parser.add_argument('model', type=str)
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=24000)
    parser.add_argument('--frame_path', type=str, default='VBIG_dataset/videos_frames', help='Path to frame for visualization')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to divide annotations into')
    parser.add_argument('--part_id', type=int, default=1, help='ID of the part to run inference on')
    return parser.parse_args()


def save_results(res_json, save_dir, filename, num_parts, part_id):
    os.makedirs(save_dir, exist_ok=True)
    if num_parts > 1:
        save_path = os.path.join(save_dir, f'{filename}_{part_id}_of_{num_parts}.json')
    else:
        save_path = os.path.join(save_dir, f'{filename}.json')
    print(f"Saving results to {save_path}")
    with open(save_path, 'w') as f:
        json.dump(res_json, f, indent=4)


def main():
    args = parse_args_inference_tasks()
    save_dir = os.path.join(args.dir, 'results_cultural', args.model.split('/')[-1])
    print(args)

    with open(os.path.join(args.dir, 'all_annotations.json'), 'r') as f:
        res_json = json.load(f)

    
    lm = dspy.LM('openai/'+ args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    activity_cot = dspy.ChainOfThought(RecognizeGroupActivity)
    clothing_cot = dspy.ChainOfThought(RecognizeGroupClothing)
    handholding_cot = dspy.ChainOfThought(RecognizeGroupHandholding)
    hugging_cot = dspy.ChainOfThought(RecognizeGroupHugging)

    # divide res_json['annotations'] into N parts and run inference on part_id (1-indexed) part
    total_annotations = len(res_json['annotations'])
    annotations_per_part = total_annotations // args.num_parts

    start_idx = (args.part_id - 1) * annotations_per_part
    end_idx = start_idx + annotations_per_part if args.part_id < args.num_parts else total_annotations

    for i, annotation in enumerate(tqdm(res_json['annotations'][start_idx:end_idx])):
        videoInfo = annotation['videoInfo']
        frame_path = os.path.join(args.dir, annotation['videoFolder'], str(videoInfo['annotationFrame']+1).zfill(5) + '.jpeg')

        for gbox in annotation['groups']:
            res = {}
            try:
                g_activity, hist_activity = inference_task(lm, activity_cot, frame_path, gbox['bbox'])
            except Exception as e:
                g_activity, hist_activity = "", ""
            try:
                g_clothing, hist_clothing = inference_task(lm, clothing_cot, frame_path, gbox['bbox'])
            except Exception as e:
                g_clothing, hist_clothing = "", ""

            try:
                g_handholding, hist_handholding = inference_task(lm, handholding_cot, frame_path, gbox['bbox'])
            except Exception as e:
                g_handholding, hist_handholding = "", ""

            try:
                g_hugging, hist_hugging = inference_task(lm, hugging_cot, frame_path, gbox['bbox'])
            except Exception as e:
                g_hugging, hist_hugging = "",""

            res['group_activity'] = g_activity
            res['group_clothing'] = g_clothing
            res['group_handholding'] = g_handholding
            res['group_hugging'] = g_hugging
            res['group_activity_LMhist'] = hist_activity
            res['group_clothing_LMhist'] = hist_clothing
            res['group_handholding_LMhist'] = hist_handholding
            res['group_hugging_LMhist'] = hist_hugging

            gbox['cultural_output'] = res
        
        if i % 25 == 0:            
            save_results(res_json, save_dir, 'annotations', args.num_parts, args.part_id)

    save_results(res_json, save_dir, 'annotations', args.num_parts, args.part_id)

        
 




if __name__ == "__main__":
    main()
