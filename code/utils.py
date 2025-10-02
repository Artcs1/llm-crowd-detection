import cv2 
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive
from prompts import IdentifyGroupsImage, IdentifyGroups_DirectionImage, IdentifyGroups_TransitiveImage, IdentifyGroups_DirectionTransitiveImage




def inference(dspy_module, input_text, mode='llm', image_path='optional'):
    is_error = False
    try:
        if mode == 'llm':
            predictions = dspy_module(detections=input_text)
        elif mode == 'vlm':
            predictions = dspy_module(image=dspy.Image.from_file(image_path), detections=input_text)
        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error


def inference_wrapper(lm, dspy_module, input_text, mode='llm', image_path='optional'):
    if mode == 'llm':
        output, is_error = inference(dspy_module, input_text, mode)
    elif mode == 'vlm':
        output, is_error = inference(dspy_module, input_text, mode, image_path)
    res = {}
    if not is_error:
        res['groups'] = output
        #res['hist'] = lm.history[-1]
        res['error'] = None
    else:
        res['groups'] = None
        #res['hist'] = lm.history[-1]
        res['error'] = output
    return res 

def get_dspy_cot(mode, prompt_method):
    
    use_direction = False
    if mode == 'llm':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_Direction)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_Transitive)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_DirectionTransitive)
            use_direction = True
    elif mode == 'vlm':
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(IdentifyGroupsImage)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_DirectionImage)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_TransitiveImage)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(IdentifyGroups_DirectionTransitiveImage)
            use_direction = True

    return (dspy_cot, use_direction)

