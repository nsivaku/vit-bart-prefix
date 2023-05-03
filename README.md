## Getting Started

Heavily inspired by rmokady/CLIP_prefix_caption

### Creating the model

1) Activate the conda environment detailed in environment.yml 
2) Download the COCO 2014 training and validation sets into data/coco directory 
3) Run generate_embeddings.py 
4) Run caption_tokens.py 
5) Run train.py

### Predicting a caption for a new image

1) Add an image file of your choice to the images directory
2) Go to predict.py and change image_path variable in main accordingly
3) Run predict.py
