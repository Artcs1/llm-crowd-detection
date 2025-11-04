from utils import *
from prompts import *


def inference_task(lm, dspy_module, img_path, gbox):
    preds = dspy_module(image=dspy.Image.from_file(img_path), bbox=gbox)

    if lm.history:
        hist = lm.history[-1]
        hist['messages'][1]['content'][1]['image_url'] = None # Remove image data from history for brevity
        hist_str = str(hist)
    else:
        hist_str = ""
    return preds.output, hist_str


def main():

    args = parse_args_inference_tasks()

    with open(args.filename, 'r') as f:
        res_json = json.load(f)

    lm = dspy.LM('openai/'+ args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    activity_cot = dspy.ChainOfThought(RecognizeGroupActivity)
    clothing_cot = dspy.ChainOfThought(RecognizeGroupClothing)
    handholding_cot = dspy.ChainOfThought(RecognizeGroupHandholding)
    hugging_cot = dspy.ChainOfThought(RecognizeGroupHugging)

    for frame_id in res_json['frame_ids']:
        gdet_output = res_json['outputs'][str(frame_id)]
        img_path = f'{args.frame_path}/{args.filename.split("/")[-1][:-5]}/{str(frame_id).zfill(5)}.jpeg'
        gboxes, counts = compute_group_bbox(gdet_output['groups'], gdet_output['id_tobbox'], return_counts=True)

        frame_results = []
        for gbox, g_size in zip(gboxes, counts):
            if g_size < 2:
                continue  # Skip single-person groups
            res = {}
            g_activity, hist_activity = inference_task(lm, activity_cot, img_path, gbox)
            g_clothing, hist_clothing = inference_task(lm, clothing_cot, img_path, gbox)
            g_handholding, hist_handholding = inference_task(lm, handholding_cot, img_path, gbox)
            g_hugging, hist_hugging = inference_task(lm, hugging_cot, img_path, gbox)

            res['gbox'] = gbox
            res['group_size'] = g_size
            res['group_activity'] = g_activity
            res['group_clothing'] = g_clothing
            res['group_handholding'] = g_handholding
            res['group_hugging'] = g_hugging
            res['group_activity_LMhist'] = hist_activity
            res['group_clothing_LMhist'] = hist_clothing
            res['group_handholding_LMhist'] = hist_handholding
            res['group_hugging_LMhist'] = hist_hugging

            frame_results.append(res)

        res_json['outputs'][str(frame_id)]['cultural_tasks'] = frame_results


    save_dir = args.filename
    save_dir = '/'.join(save_dir.split('/')[:-2]).replace('results', 'results_cultural')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{args.filename.split("/")[-1][:-5]}.json')
    with open(save_path, 'w') as f:
        json.dump(res_json, f, indent=4)
    print(f'Saved results to {save_path}')


if __name__ == '__main__':
    main()
