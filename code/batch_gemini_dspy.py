from utils import *

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

def main():


    args = parse_args()

    collected_files = [os.path.join(args.filename,f) for f in os.listdir(args.filename) if os.path.isfile(os.path.join(args.filename, f))]
    collected_files.sort()
    
    collected_files = collected_files[int(args.start):]

    for current_file in tqdm(collected_files):
        try:
            #print(current_file)
        #if True:
            with open(current_file, 'r') as f:
                data = json.load(f)
        
                lm = GeminiVertexLM(
                    model="gemini-3-pro-preview",
                    project=GCP_PROJECT,
                    location=GCP_LOCATION,
                    max_tokens=65535, # 24000,
                    temperature=1.0,
                    thinking_budget=1024, # 0,      # set >0 to enable thinking (e.g., 256, 1024)
                )
            
                dspy.configure(
                    lm=lm,
                    adapter=dspy.JSONAdapter()
                )
            
            #lm = dspy.LM('openai/'+args.model, api_key=args.api_key, api_base=args.api_base, temperature=args.temperature, max_tokens=args.max_tokens)
            #dspy.configure(lm=lm)
            
            if args.setting == 'single':
                dspy_cot, use_direction = get_dspy_cot(args.mode, args.prompt_method)
                frame_input_data, personid2bbox = get_frame_bboxes(data, use_direction, args.depth_method, args.frame_id, args.prompt_method)
            elif args.setting == 'full':
                dspy_cot, use_direction = get_full_dspy_cot(args.mode, args.prompt_method)
                all_frames, bboxes = get_allframes_bboxes(data, use_direction, args.depth_method, args.prompt_method)
        
            save_filename = current_file.split('/')[-1][:-5]
            frame_path = f'{args.frame_path}/{save_filename}/{str(args.frame_id).zfill(5)}.jpeg'
        
            if args.setting == 'single':
                if args.mode == 'llm' or args.mode =='vlm_text':
                    output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode)
                elif args.mode == 'vlm_image':
                    output = inference_wrapper(lm, dspy_cot, frame_input_data, args.mode, frame_path)
            elif args.setting == 'full':
                if args.mode == 'llm' or args.mode == 'vlm_text':
                    output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode)
                elif args.mode == 'vlm_image':
                    output = full_inference_wrapper(lm, dspy_cot, all_frames, args.frame_id, args.mode, frame_path)
        
            if output['error'] is not None:
                print(f"Error during inference: {output['error']}")
                continue#return
            
        
            output['frame_id'] = args.frame_id
        
            if args.setting == 'single':
                output['id_tobbox'] = personid2bbox
                res_path = 'predictions/'+ args.frame_path.split('/')[-2] + '/results'
                save_frame(output, personid2bbox, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method, args.frame_id)
            elif args.setting == 'full':
                output['id_tobbox'] = bboxes[args.frame_id-1]
                res_path = 'predictions/'+ args.frame_path.split('/')[-2] + '/results_full'
                save_full_frame(output, bboxes, res_path, save_filename, frame_path, args.save_image, args.model, args.mode, args.depth_method, args.prompt_method, args.frame_id)
        
        except Exception as e:
            print(f'Fail in: {current_file}, seting: {args.setting},  mode: {args.mode}, model: {args.model}, prompt: {args.prompt_method}')
    
if __name__ == "__main__":
    main()
