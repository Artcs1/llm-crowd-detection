import os

from collections import Counter
from google import genai
from google.genai import types
import base64
import ast
import json
import pathlib
from glob import glob
from tqdm import tqdm

os.environ["GOOGLE_CLOUD_PROJECT"] = "stb-prj-paola-sbx"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

GCP_PROJECT = "stb-prj-paola-sbx"    # your GCP project ID
GCP_LOCATION = "global" # "us-central1"          # or "global"


import dspy

client = genai.Client(vertexai=True, project="stb-prj-paola-sbx", location="global",)

import io
from PIL import Image

import cv2
import argparse
import pandas as pd
from tqdm.auto import tqdm

from utils import *
from prompts import *

import datetime
import uuid


class GeminiVertexLM(dspy.LM):
    """DSPy-compatible LM wrapper for Gemini 3 Pro Preview via google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        project: str = None,
        location: str = None,
        max_tokens: int = 65535,
        temperature: float = 1.0,
        top_p: float = 0.95,
        thinking_budget: int = 32768, # 1024, # 0,
    ):
        # Pass a litellm-compatible name to the base class so it doesn't error.
        # We override __call__ so litellm is never actually invoked.
        super().__init__(model=f"vertex_ai/{model}")

        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget = thinking_budget

        # Initialize the google-genai client with Vertex AI
        self.client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _build_config(self) -> types.GenerateContentConfig:
        """Build the GenerateContentConfig matching your notebook's pattern."""
        config_kwargs = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",   threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",   threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",          threshold="OFF"),
            ],
        )
        if self.thinking_budget and self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        return types.GenerateContentConfig(**config_kwargs)

    def _messages_to_contents(self, messages):
      contents = []
      for msg in messages:
          role = msg.get("role", "user")
          if role in ("system", "developer"):
              role = "user"
          elif role == "assistant":
              role = "model"

          content = msg.get("content", "")
          parts = []

          if isinstance(content, str):
              parts.append(types.Part.from_text(text=content))

          elif isinstance(content, list):
              for item in content:
                  if isinstance(item, str):
                      parts.append(types.Part.from_text(text=item))
                  elif isinstance(item, dict):
                      if item.get("type") == "text":
                          parts.append(types.Part.from_text(text=item["text"]))
                      elif item.get("type") == "image_url":
                          url_data = item["image_url"]["url"]
                          if url_data.startswith("data:image"):
                              header, b64 = url_data.split(",", 1)
                              mime = header.split(";")[0].split(":")[1]
                              import base64
                              parts.append(types.Part.from_bytes(       # ← FIX
                                  data=base64.b64decode(b64), mime_type=mime
                              ))
                  elif isinstance(item, Image.Image):
                      parts.append(types.Part.from_bytes(               # ← FIX
                          data=self._pil_to_bytes(item), mime_type="image/png"
                      ))

          if parts:
              contents.append(types.Content(role=role, parts=parts))
      return contents

    # def __call__(self, prompt=None, messages=None, **kwargs):
    #     if messages:
    #         contents = self._messages_to_contents(messages)
    #     elif prompt:
    #         contents = [types.Content(
    #             role="user",
    #             parts=[types.Part.from_text(text=prompt)]                 # FIX
    #         )]
    #     else:
    #         raise ValueError("Either 'prompt' or 'messages' must be provided.")

    #     config = self._build_config()

    #     full_response = ""
    #     for chunk in self.client.models.generate_content_stream(
    #         model=self.model_name,
    #         contents=contents,
    #         config=config,
    #     ):
    #         if (chunk.candidates and chunk.candidates[0].content
    #                 and chunk.candidates[0].content.parts):
    #             for part in chunk.candidates[0].content.parts:
    #                 if part.text:
    #                     full_response += part.text

    #     return [full_response]
    def __call__(self, prompt=None, messages=None, **kwargs):
      if messages:
          contents = self._messages_to_contents(messages)
      elif prompt:
          contents = [types.Content(
              role="user",
              parts=[types.Part.from_text(text=prompt)]
          )]
      else:
          raise ValueError("Either 'prompt' or 'messages' must be provided.")

      config = self._build_config()

      # Stream response
      full_response = ""
      for chunk in self.client.models.generate_content_stream(
          model=self.model_name,
          contents=contents,
          config=config,
      ):
          if (chunk.candidates and chunk.candidates[0].content
                  and chunk.candidates[0].content.parts):
              for part in chunk.candidates[0].content.parts:
                  if part.text:
                      full_response += part.text

      outputs = [full_response]

      # ── Record history so lm.history / dspy.inspect_history() work ──
      entry = {
          "prompt": prompt,
          "messages": messages,
          "kwargs": kwargs,
          "response": full_response,
          "outputs": outputs,
          "usage": {
              "prompt_tokens": 0,
              "completion_tokens": 0,
              "total_tokens": 0,
          },
          "cost": None,
          "timestamp": datetime.datetime.now().isoformat(),
          "uuid": str(uuid.uuid4()),
          "model": self.model,
          "response_model": self.model_name,
          "model_type": "chat",
      }
      self.update_history(entry)

      return outputs

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

    
    #lm = dspy.LM('openai/'+ args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
    #dspy.configure(lm=lm)

    lm = GeminiVertexLM(
        model="gemini-3-pro-preview",
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        max_tokens=65535, # 24000,
        temperature=1.0,
        thinking_budget=256#1024, # 0,      # set >0 to enable thinking (e.g., 256, 1024)
    )

    dspy.configure(
        lm=lm,
        adapter=dspy.JSONAdapter()
    )



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
