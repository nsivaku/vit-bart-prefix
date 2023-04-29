import json

with open('./data/coco/annotations/captions_val2014.json', 'r') as f:
    data = json.load(f)

json.dump(data['annotations'], open('./data/coco/annotations/val_caption.json', 'w'), indent="")