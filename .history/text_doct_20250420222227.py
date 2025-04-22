from donut import DonutModel
from PIL import Image
import torch








# Load model
model = DonutModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", ignore_mismatched_sizes=True)
