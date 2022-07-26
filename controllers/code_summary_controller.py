from controllers.printing import print_to_console
import os 
from flask import json
from flask.wrappers import Request
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
# from fastT5 import export_and_get_onnx_model, get_onnx_model
# import time

class CodeSummaryController:
    cache = {}

    def get_summary(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        text = data.get("content", "")
        if not text:
            return None
            
        text = text[:512]
        hash_key = hash(text)
        if hash_key in self.cache:
            return {"result": self.cache[hash_key]}

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = "Salesforce/codet5-base-multi-sum"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        device_tokenizer = tokenizer(text, return_tensors="pt").to(device)
        input_ids = device_tokenizer.input_ids
        generated_ids = model.generate(input_ids, max_length=20)
        result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.cache[hash_key] = result

        print_to_console("Summary - code:", text)
        print_to_console("Summary - result:", result)

        return {"result": result}

    
    def get_summaries(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        code_parts = data.get("content", "")
        if not code_parts:
            return None
        
        result = []
        not_cached_parts = {}
        for id, part in enumerate(code_parts):
            hash_key = hash(part)
            if hash_key in self.cache:
                result.append(self.cache[hash_key])
            else:
                not_cached_parts[id] = part[:512]
        missing_parts = list(not_cached_parts.values())
        if len(missing_parts) != 0:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_name = "Salesforce/codet5-base-multi-sum"
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model.to(device)
            device_tokenizer = tokenizer(missing_parts, return_tensors="pt", padding=True).to(device)
            input_ids = device_tokenizer.input_ids
            generated_ids = model.generate(input_ids, max_length=20)
            counter = 0
            for id in list(not_cached_parts.keys()):
                original_code = code_parts[id]
                summary = tokenizer.decode(generated_ids[counter], skip_special_tokens=True)
                hash_key = hash(original_code)
                self.cache[hash_key] = summary
                result.insert(id, summary)
                counter += 1
        
        return {"result": result}