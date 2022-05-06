import utils.logging
from flask import json
from flask.wrappers import Request
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM, pipeline

class CodeSymbolController:

    def get_symbol_name(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        text = data.get("content", "")
        if not text:
            return None
        text = text[:512]
        
        model_type = data.get("modelType", 2)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
       
        if model_type == 0:
            utils.logging.log("New symbol name - model:", "huggingface/CodeBERTa-small-v1")
            model = RobertaForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
            model.to(device)
            tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1 if device == "cpu" else 0)
            result = fill_mask(text)
            result = list(map(lambda x: x["token_str"].strip(), result))
        elif model_type == 1:
            utils.logging.log("New symbol name - model:", "microsoft/codebert-base-mlm")
            model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
            model.to(device)
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1 if device == "cpu" else 0)
            result = fill_mask(text)
            result = list(map(lambda x: x["token_str"].strip(), result))
        else:
            utils.logging.log("New symbol name - model:", "Salesforce/codet5-base")
            model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
            model.to(device)
            tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
            device_tokenizer = tokenizer(text, return_tensors="pt").to(device)
            input_ids = device_tokenizer.input_ids
            generated_ids = model.generate(input_ids, max_length=8)
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        utils.logging.log("New symbol name - code:", text)
        utils.logging.log("New symbol name - result:", result)

        return { "result": result }