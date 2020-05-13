# https://pytorch.org/hub/pytorch_vision_resnet/

import torch
import urllib
from PIL import Image
from torchvision import transforms

# Download pretrained model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")

try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

# sample execution
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

with torch.no_grad():
    output = model(input_batch)

print(output.squeeze().max(0))

traced_script_module = torch.jit.trace(model, input_batch)

traced_script_module.save("model.pt")
