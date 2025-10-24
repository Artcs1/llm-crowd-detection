from utils import *
from prompts import *


def inference_task(lm, dspy_module, img_path, gbox):
    preds = dspy_module(image=dspy.Image.from_file(img_path), bbox=gbox)
    return preds.output, lm.history[-1]


def main():

    args = parse_args_cultural_tasks()

    with open(args.filename, 'r') as f:
        res_json = json.load(f)

    lm = dspy.LM('openai/'+ args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)

    activity_cot = dspy.ChainOfThought(RecognizeGroupActivity)
    clothing_cot = dspy.ChainOfThought(RecognizeGroupClothing)
    handholding_cot = dspy.ChainOfThought(RecognizeGroupHandholding)

    for frame_id in res_json['frame_ids']:
        gdet_output = res_json['outputs'][str(frame_id)]
        img_path = f'{args.frame_path}/{args.filename.split("/")[-1][:-5]}/{str(frame_id).zfill(5)}.jpeg'
        gboxes = compute_group_bbox(gdet_output['groups'], gdet_output['id_tobbox'])

        frame_results = []
        for gbox in gboxes:
            res = {}
            g_size = len(gbox)
            g_activity, hist_activity = inference_task(lm, activity_cot, img_path, gbox)
            g_clothing, hist_clothing = inference_task(lm, clothing_cot, img_path, gbox)
            g_handholding, hist_handholding = inference_task(lm, handholding_cot, img_path, gbox)

            res['group_size'] = g_size
            res['group_activity'] = g_activity
            res['group_activity_LMhist'] = hist_activity
            res['group_clothing'] = g_clothing
            res['group_clothing_LMhist'] = hist_clothing
            res['group_handholding'] = g_handholding
            res['group_handholding_LMhist'] = hist_handholding

            frame_results.append(res)

        res_json['outputs'][str(frame_id)]['cultural_tasks'] = frame_results

    save_path = f'results_cultural/{args.filename.split("/")[-1][:-5]}_cultural_tasks.json'
    with open(save_path, 'w') as f:
        json.dump(res_json, f, indent=4)
    print(f'Saved results to {save_path}')


if __name__ == '__main__':
    main()