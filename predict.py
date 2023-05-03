from cog import BasePredictor, Input, Path
from PIL import Image
from train import PrefixModel
import torch
from transformers import ViTImageProcessor, ViTModel, BartTokenizerFast
import skimage.io as io

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cpu")
        self.preprocess = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')


        model = PrefixModel(prefix_length=10, clip_length=10)
        model.load_state_dict(torch.load('./checkpoints/vit_bart_latest-001.pt', map_location=torch.device('cpu')))
        model.eval()
        model = model.to(self.device)
        self.model = model

    def predict(self, image):
        """Run a single prediction on the model"""
        image = io.imread(image)
        image = Image.fromarray(image)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.preprocess(image, return_tensors='pt').pixel_values
        prefix = self.encoder(image).last_hidden_state.mean(dim=1)
        with torch.no_grad():
            prefix_embed = self.model.img_project(prefix).reshape(1, 10, -1)
        output = self.model.bart.generate(inputs_embeds=prefix_embed, max_new_tokens=20)[0]
        return self.tokenizer.decode(output.tolist()[2:-1])

def main():
    predictor = Predictor()
    predictor.setup()
    image_path = './images/spill.jpg'
    prediction = predictor.predict(image_path)
    print(prediction)
    
if __name__ == '__main__':
    main()
    
    
    
    
