import dspy
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Metadata JSON file")
    parser.add_argument('filename', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1")
    parser.add_argument('--api_key', type=str, default="testkey")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    return parser.parse_args()


class IdentifyGroups(dspy.Signature):
    """"Identify sets of people that are very close to each other from the given 3D coordinates. Use all three coordinates in distance calculation. Select an appropriate grouping threshold based on all pairwise distances. Do not hallucinate empty sets."""
    detections: list[dict] = dspy.InputField(desc="list of dictionary objects with the keys: person_id, x, y, z.")
    groups: list[list[int]] = dspy.OutputField(desc="A list of lists of person_ids that are close to each other.")


def main():
    args = parse_args()
    print(args)
    with open(args.filename, 'r') as f:
        data = json.load(f)

    lm_detect_crowds = dspy.ChainOfThought(IdentifyGroups)
    lm = dspy.LM('openai/'+args.model, api_key='testkey', api_base='http://localhost:8000/v1', temperature=0.6, max_tokens=32768)
    dspy.configure(lm=lm)

    results = []
    for frame in data['frames']:
        outputs = {}
        outputs['frame_id'] = frame['frame_id']

        frame_input_data = []
        for i,det in enumerate(frame['detections']):
            frame_input_data.append({'person_id': det['track_id'], 'x': det['unidepth_3D'][0], 'y': det['unidepth_3D'][1], 'z': det['unidepth_3D'][2]})

        try:
            predictions = lm_detect_crowds(detections=frame_input_data)
            outputs['groups'] = predictions['groups']
            outputs['history'] = lm.history[-1]
            outputs['error'] = None
        except Exception as e:
            outputs['groups'] = None
            outputs['history'] = lm.history[-1]
            outputs['error'] = str(e)

        results.append(outputs)

    df = pd.DataFrame(results)
    # make output file name same as input but with csv extension
    output_file = args.filename.replace('.json', '_groups.csv')
    df.to_csv(output_file, index=False)

    

if __name__ == "__main__":
    main()
        