from donut import DonutModel
from PIL import Image
import torch

# Load model
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

# Load your scanned ID image
image = Image.open("path_to_your_image.jpg").convert("RGB")

# Generate JSON output
output = model.inference(image, task_prompt="document question answering")

print("Predicted JSON:")
print(output)
