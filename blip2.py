# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

img_url = input('Image URL : ') 
raw_image = Image.open(requests.get(img_url, stream=True).raw)

print('Image loaded')

question = "What is a detailed description of this image?"
qtext = f"Question: {question} Answer:"
inputs = processor(raw_image, qtext, return_tensors="pt").to("cuda", torch.float16)
print('Inputs loaded')

out = model.generate(**inputs)
print(out)
print(processor.decode(out[0], skip_special_tokens=True).strip())
