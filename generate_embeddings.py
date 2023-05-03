import torch
import torchvision.models as models 
import skimage.io as io
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

# set up
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')
with open('./data/coco/annotations/captions_train2014.json', 'r') as f:
    train_data = json.load(f)
with open('./data/coco/annotations/captions_val2014.json', 'r') as f:
    val_data = json.load(f)
train_data = train_data['annotations']
val_data = val_data['annotations']

print(f"{len(train_data) + len(val_data)} captions loaded from json ")

# prepare embeddings
train_embeddings = []
train_captions = []
val_embeddings = []
val_captions = []

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
encoder = encoder.to(device)


print("Training Embeddings")
for i in tqdm(range(len(train_data))):
    d = train_data[i]
    img_id = d["image_id"]
    filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        
    # preprocess image
    image = io.imread(filename)
    image = Image.fromarray(image)
    if image.mode == 'L':
        image = image.convert('RGB')
    image = processor(image, return_tensors='pt').pixel_values.to(device)

    # encode image
    with torch.no_grad():
        prefix = encoder(image).last_hidden_state.mean(dim=1)

    d["embedding"] = i
    train_embeddings.append(prefix)
    train_captions.append(d)
    if (i + 1) % 100000 == 0 or (i + 1) == len(train_data):
        with open(f"oscar_split_train_{i + 1}.pkl", 'wb') as f:
            pickle.dump({"captions": train_captions, "embedding": torch.cat(train_embeddings, dim=0)}, f)

print("Validation Embeddings")
for i in tqdm(range(len(val_data))):
    d = val_data[i]
    img_id = d["image_id"]
    filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        
    # preprocess image
    image = io.imread(filename)
    image = Image.fromarray(image)
    if image.mode == 'L':
        image = image.convert('RGB')
    image = processor(image, return_tensors='pt').pixel_values.to(device)

    # encode image
    with torch.no_grad():
        prefix = encoder(image).last_hidden_state.mean(dim=1)

    d["embedding"] = i
    val_embeddings.append(prefix)
    val_captions.append(d)
    if (i + 1) % 100000 == 0 or (i + 1) == len(val_data):
        with open(f"oscar_split_val_{i + 1}.pkl", 'wb') as f:
            pickle.dump({"captions": val_captions, "embedding": torch.cat(val_embeddings, dim=0)}, f)
            
print('Done')