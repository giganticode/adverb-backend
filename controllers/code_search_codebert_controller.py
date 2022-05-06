import os
from flask import json
from flask.wrappers import Request
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from tqdm import tqdm

class CodeSearchCodeBertController:
    
    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        batch_size = data.get("batch_size", 8)
        if not content or not search_text:
            return None

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained(os.path.join(os.getcwd(), "models", "codebert-base"), local_files_only=True)
        model.to(device)
        
        query = search_text
        query_vec = model(tokenizer(query,return_tensors='pt').to(device).input_ids)[1]
             
        codePartsCounter = 0
        tensors = []
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            codePartsCounter += 1
            code = lines[i : (i + batch_size)]
            code = " ".join(code).replace("\r\n", " ").replace("\n", " ")[:512]
            code_vec =  model(tokenizer(code, return_tensors="pt").to(device).input_ids)[1]
            tensors.append(code_vec)
            i += batch_size + 1

        code_vecs = torch.cat(tensors, 0)
        scores = torch.einsum("ab,cb->ac", query_vec, code_vecs)
        scores = torch.softmax(scores, -1)

        search_lines = []
        for i in range(codePartsCounter):
            score = scores[0, i].item()
            line = i * batch_size
            if score > 0.9:
                search_lines.append(line)

        return { "result": {"search_text": search_text, "search_lines": search_lines, "batch_size": batch_size} }