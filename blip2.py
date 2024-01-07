# pip install accelerate
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

img_url = input('Image URL : ') 
raw_image = Image.open(requests.get(img_url, stream=True).raw)

print('Image loaded')

question = "What would be the main keywords you would use to describe this image?"
qtext = f"Question: {question} Answer:"
inputs = processor(raw_image, qtext, return_tensors="pt").to("cuda")
print('Inputs loaded')

out = model.generate(**inputs, max_new_tokens=1000)
print(out)
print(processor.decode(out[0], skip_special_tokens=True).strip())
