from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

# MPS device (for MacBook M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Load image
image = Image.open("/Users/vijayrajgohil/Downloads/unicorns.jpeg").convert("RGB")

# Prepare input
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
task_prompt = "<s_docvqa><s_question>What is all the content of this document?<s_answer>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

# Run inference
outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
result = processor.token2json(result)

print("🔍 Predicted JSON:")
print(result)



