import torch
from PIL import Image

import clip


print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN101", device=device)
image_path = r"F:\datasets\FGVC10\snake\SnakeCLEF2023-medium_size\1990\Amphiesma_stolatum\59067968.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(["CN",
                      "US",
                      "UK",
                      "CL",
                      "JP",
                      'CN']).to(device)
print(text.shape)
# print(text)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(text_features)
    print(text_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]