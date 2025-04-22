from donut import DonutModel
from PIL import Image
import torch

# Load model
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", ignore_mismatched_sizes=True)
model.eval()

# Load your scanned ID image
image = Image.open("/Users/vijayrajgohil/Downloads/unicorns.jpeg").convert("RGB")

# Generate JSON output
output = model.inference(image, task_prompt="document question answering")

print("Predicted JSON:")
print(output)
