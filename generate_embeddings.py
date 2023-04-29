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
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
output_path = "./data/coco/oscar_split_train.pkl"
with open('./data/coco/annotations/train_caption.json', 'r') as f:
    data = json.load(f)
print("%0d captions loaded from json " % len(data))

# prepare embeddings
all_embeddings = []
all_captions = []
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
encoder = encoder.cuda()

num_examples = len(data)

for i in tqdm(range(num_examples)):
    d = data[i]
    img_id = d["image_id"]
    filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
    if not os.path.isfile(filename):
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
    all_embeddings.append(prefix)
    all_captions.append(d)
    if (i + 1) % 100000 == 0 or (i + 1) == len(data):
        with open(f"oscar_split_train_{i + 1}.pkl", 'wb') as f:
            pickle.dump({"captions": all_captions, "embedding": torch.cat(all_embeddings, dim=0)}, f)
            
print('Done')
print("%0d embeddings saved " % len(all_embeddings))