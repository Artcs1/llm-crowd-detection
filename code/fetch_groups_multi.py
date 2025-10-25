import dspy
import json


from utils import *

def main():

    args = parse_args_allframes()
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=lm)
    
    frameid_to_input_data = {}
    frameid_to_personid2bbox = {}
    frameid_to_output = {}
    
    if args.setting == 'single':
        dspy_cot, use_direction = get_dspy_cot(args.mode, args.prompt_method)
        for frame_id in args.frame_ids:
            frame_input_data, personid2bbox = get_frame_bboxes(data, use_direction, args.depth_method, frame_id) 
            frameid_to_input_data[frame_id] = frame_input_data
            frameid_to_personid2bbox[frame_id] = personid2bbox
    elif args.setting == 'full':
        dspy_cot, use_direction = get_full_dspy_cot(args.mode, args.prompt_method)
        all_frames, bboxes = get_allframes_bboxes(data, use_direction, args.depth_method)

    save_filename = args.filename.split('/')[-1][:-5]
    frameid_to_path = {}
    for frame_id in args.frame_ids:
        frame_path = f'{args.frame_path}/{save_filename}/{str(frame_id).zfill(5)}.jpeg'
        frameid_to_path[frame_id] = frame_path

    if args.setting == 'single':
        for frame_id in args.frame_ids:
            frame_input_data = frameid_to_input_data[frame_id]
            frame_path = frameid_to_path[frame_id]
            if args.mode == 'llm' or args.mode =='vlm_text':
                output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode)
            elif args.mode == 'vlm_image':
                output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode, frame_path)
            
            if output['error'] is not None:
                print(f"Error during inference: {output['error']}")
                return
            
            output['id_tobbox'] = frameid_to_personid2bbox[frame_id]
            frameid_to_output[frame_id] = output

    
    elif args.setting == 'full':
        for frame_id in args.frame_ids:
            if args.mode == 'llm' or args.mode == 'vlm_text':
                output = full_inference_wrapper(lm, dspy_cot, all_frames, frame_id, args.mode)
            elif args.mode == 'vlm_image':
                output = full_inference_wrapper(lm, dspy_cot, all_frames, frame_id, args.mode, frame_path)

            output['id_tobbox'] = bboxes[frame_id-1]        # 1-indexed
            frameid_to_output[frame_id] = output
        

    res_json = {}
    res_json['frame_ids'] = args.frame_ids
    res_json['outputs'] = frameid_to_output
    res_json['city'] = data['city']
    res_json['country'] = data['country']
    res_json['dataset'] = data['dataset']

    if args.setting == 'single':
        res_path = 'results'
    elif args.setting == 'full':
        res_path = 'results_full'
    save_json(res_json, res_path, save_filename, args.model, args.mode, args.depth_method, args.prompt_method)


if __name__ == "__main__":
    main()
