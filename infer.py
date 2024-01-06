import requests
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# image = Image.open(open("/Users/soumya/Desktop/SS/Screenshot 2023-08-02 at 1.54.25 AM.png", "rb"))

model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").to("cpu")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

prompt = "<grounding> Describe the main features of this image"

def main():
    while True:
        urls = input('Paste image URLs : ') #https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png
        url_split = urls.split(' ')

        # The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
        images = [ Image.open(requests.get(url, stream=True).raw) for url in url_split ]

        # The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
        # image.save("new_image.jpg")
        # image = Image.open("new_image.jpg")

        start = time.time_ns()
        prompts = [ prompt for _ in range(len(images)) ]
        inputs = processor(text=prompts, images=images, return_tensors="pt")

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Specify `cleanup_and_extract=False` in order to see the raw model generation.
        gens = [ processor.post_process_generation(generated_text) for generated_text in generated_texts ]
        pt = [ t[0] for t in gens ] 
        et = [ t[1] for t in gens ]
        # processed_text, entities = processor.post_process_generation(generated_text)

        print(pt)
        # `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

        # By default, the generated  text is cleanup and the entities are extracted.
        # processed_text, entities = processor.post_process_generation(generated_text)

        # print(processed_text)
        # `An image of a snowman warming himself by a fire.`

        print(et)
        # `[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]`

        end = time.time_ns()

        print(f'Done in {(end - start) / 1_000_000} ms')


if __name__ == '__main__':
    main()

