from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np
from train import PrefixModel
import torch
import torch.nn.functional as nnf
import torchvision.models as models
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
        model.load_state_dict(torch.load('./checkpoints/vit_bart_latest-005.pt', map_location=torch.device('cpu')))
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
        # print(prefix_embed.shape)
        # generator = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        # inputs_embeds = # a 3D tensor, [batch, seq_length, dim]
        # attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)
        decoder_input_ids = torch.ones((prefix_embed.shape[0], 1), dtype=torch.long)*2
        # output_ids = generator.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=inputs_embeds, max_length=100,num_beams=4)
        # return self.model.bart(inputs_embeds=prefix_embed, decoder_input_ids=decoder_input_ids)
        # outputs = self.model.bart(inputs_embeds=prefix_embed, decoder_inputs_embeds=prefix_embed)
        # logits = outputs.logits[:, 9: -1]
        
        # # return outputs
        return generate2(self.model, self.tokenizer, embed=prefix_embed)

def generate(model, tokenizer, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=70, top_p=0.8, temperature=1.0, stop_token: str = '.'):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.bart.model.shared(tokens)
                print(generated)

            for i in range(entry_length):
                decoder_inputs_embeds = generated.new_zeros(generated.shape)
                decoder_inputs_embeds[:, 1:] = generated[:, :-1].clone()
                decoder_inputs_embeds[:, 0] = model.bart.get_input_embeddings().weight[2]
                outputs = model.bart(inputs_embeds=generated, decoder_inputs_embeds=decoder_inputs_embeds)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                # print(logits.shape)
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # print(next_token)
                next_token_embed = model.bart.model.shared(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list

def generate2(model, tokenizer, beam_size: int = 5, tokens=None, prompt=None, embed=None, entry_length=70, temperature=1.0, stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.bart(inputs_embeds=generated, decoder_inputs_embeds=generated)
            
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.bart.model.shared(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            # next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
            #     generated.shape[0], 1, -1
            # )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

def main():
    predictor = Predictor()
    predictor.setup()
    image_path = './images/spill.jpg'
    prediction = predictor.predict(image_path)
    print(prediction)
    
if __name__ == '__main__':
    main()
    
    
    
    
