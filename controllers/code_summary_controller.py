from flask import json
from flask.wrappers import Request
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
# from fastT5 import export_and_get_onnx_model, get_onnx_model
# import time

class CodeSummaryController:
    def get_summary(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        text = data.get("content", "")
        if not text:
            return None
        text = text[:512]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = "Salesforce/codet5-base-multi-sum"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

        #################################################
        ### DEFAULT APPROACH USING NORMAL HUGGINGFACE ###
        #################################################
        # start = time.time()
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        device_tokenizer = tokenizer(text, return_tensors="pt").to(device)
        input_ids = device_tokenizer.input_ids
        generated_ids = model.generate(input_ids, max_length=20)
        result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # latency = time.time() - start
        # print("Inference time = {} ms".format(latency * 1000, '.2f'))
        # print(result)

        ################################################
        ### FASTT5-APPROACH: NOT FASTER; MUCH SLOWER ###
        ################################################
        # start = time.time()
        # output_path = "models/codet5/"
        # try:
        #     model = get_onnx_model(model_name, onnx_models_path=output_path)
        # except:
        #     model = export_and_get_onnx_model(model_name, custom_output_path=output_path)
        # generated_ids = model.generate(input_ids, max_length=20)
        # result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # latency = time.time() - start
        # print("Inference time = {} ms".format(latency * 1000, '.2f'))
        # print(result)

        return {"result": result}
