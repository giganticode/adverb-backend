from flask import json
from flask.wrappers import Request
import torch
from transformers import AutoModel, AutoTokenizer

class CodeSearchController:

    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        if not content or not search_text:
            return None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        model.to(device)
        
        nl_tokens = tokenizer.tokenize(search_text)
        code_tokens = tokenizer.tokenize(content)
        tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]

        return { "result": context_embeddings }