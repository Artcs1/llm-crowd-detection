import cv2 
import os
import dspy
import json
import argparse
import pandas as pd
from constants import CV2_COLORS
from tqdm.auto import tqdm
from prompts import IdentifyGroups, IdentifyGroups_Direction, IdentifyGroups_Transitive, IdentifyGroups_DirectionTransitive
from prompts import vlm_GroupsQAonlyImage, vlm_IdentifyGroupsImage, vlm_IdentifyGroups_DirectionImage, vlm_IdentifyGroups_TransitiveImage, vlm_IdentifyGroups_DirectionTransitiveImage
from prompts import vlm_IdentifyGroupsText, vlm_IdentifyGroups_DirectionText, vlm_IdentifyGroups_TransitiveText, vlm_IdentifyGroups_DirectionTransitiveText

def inference(dspy_module, input_text, mode='llm', image_path='optional'):
    is_error = False
    try:
        if mode == 'llm' or mode == 'vlm_text':
            predictions = dspy_module(detections=input_text)
        elif mode == 'vlm_image':
            predictions = dspy_module(image=dspy.Image.from_file(image_path), detections=input_text)
        output = predictions.groups
    except Exception as e:
        output = str(e)
        is_error = True
    return output, is_error


def inference_wrapper(lm, dspy_module, input_text, mode='llm', image_path='optional'):
    if mode == 'llm' or mode == 'vlm_text':
        output, is_error = inference(dspy_module, input_text, mode)
    elif mode == 'vlm_image':
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
    elif mode == 'vlm_image':
        if prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyImage)
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsImage)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionImage)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveImage)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveImage)
            use_direction = True
    elif mode == 'vlm_text':
        if prompt_method == 'baseline1':
            dspy_cot = dspy.ChainOfThought(vlm_GroupsQAonlyText)
        if prompt_method == 'p1':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroupsText)
        elif prompt_method == 'p2':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionText)
            use_direction = True
        elif prompt_method == 'p3':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_TransitiveText)
        elif prompt_method == 'p4':
            dspy_cot = dspy.ChainOfThought(vlm_IdentifyGroups_DirectionTransitiveText)
            use_direction = True



    return (dspy_cot, use_direction)

