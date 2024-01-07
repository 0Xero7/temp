# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b". torch.dtype=torch.float16).to("cuda")

img_url = input('Image URL : ') 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "describe the main keywords you can gather from this image"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
