from dataset import CocoDataset
from transformer_mapper import TransformerMapper
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizerFast, get_linear_schedule_with_warmup, EvalPrediction
import evaluate
from typing import Optional, Tuple
import argparse
import os
import sys

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class PrefixModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor):
        prefix_projections = self.img_project(prefix).view(-1, self.prefix_length, self.bart_embedding_size)
        labels = torch.clone(tokens)
        labels[labels[:, :] == self.bart.config.pad_token_id] = -100
        out = self.bart(inputs_embeds=prefix_projections, labels=labels)
        return out
         

    def __init__(self, prefix_length: int, clip_length: int = 10, prefix_size: int = 768, num_layers: int = 8):
        super(PrefixModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prefix_length = prefix_length
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(self.device)
        self.bart_embedding_size = self.bart.model.shared.weight.shape[1]
        self.img_project = MLP((prefix_size, (self.bart_embedding_size * prefix_length) // 2, self.bart_embedding_size * prefix_length))
        # self.img_project = TransformerMapper(prefix_size, self.bart_embedding_size, prefix_length, clip_length, num_layers)
    
class StaticBartPrefixModel(PrefixModel):
    
    def parameters(self):
       return self.img_project.parameters()
    
    def train(self, mode: bool = True):
        super(PrefixModel, self).train(mode)
        self.bart.eval()
        return self
        
def train(dataset: CocoDataset, val_dataset: CocoDataset, model: PrefixModel, lr: float = 2e-5, warmup_steps: int = 3000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 3
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    summary_writer = SummaryWriter(log_dir='./tensorboard')
    
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

    def compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
        rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
        bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
        return {
                **rouge_result, 
                "bleu": round(bleu_result["bleu"] * 100, 4), 
                "gen_len": bleu_result["translation_length"] / len(predictions)
        }
    
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch + 1}")
        
        
        model.train()
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc='vit_bart_latest')
        for idx, (tokens, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, prefix = tokens.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()
        
        
        model.eval()
        valid_loss = 0
        predictions, labels = [], []
        progress = tqdm(total=len(val_dataloader), desc='evaluating')
        for idx, (tokens, prefix) in enumerate(val_dataloader):
            with torch.no_grad():
                tokens, prefix = tokens.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix)
                loss = outputs.loss
                valid_loss += loss
                logits = outputs.logits.detach().cpu()
                predictions.extend(logits.argmax(dim=-1).tolist())
                tokens = tokens.detach().cpu()
                labels.extend(tokens.tolist())
            progress.update()
        progress.close()
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_metrics(eval_prediction)
        print(f"\nEpoch: {epoch + 1}, Valid Loss: {valid_loss / len(val_dataloader)}, BLEU: {metrics['bleu']:.4f}, ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}\n")
        summary_writer.add_scalar("valid_loss", valid_loss / len(val_dataloader))
        summary_writer.add_scalar("bleu", metrics["bleu"])
        summary_writer.add_scalar("rouge1", metrics["rouge1"])
        summary_writer.add_scalar("rouge2", metrics["rouge2"])
        summary_writer.add_scalar("rougeL", metrics["rougeL"])
        
        
        torch.save(
            model.state_dict(),
            os.path.join("./checkpoints", f"vit_bart_latest-{epoch + 1:03d}.pt"),
        )
    return model


def main():
    parser = argparse.ArgumentParser();
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args();
    dataset = CocoDataset(prefix_length=args.prefix_length, normalize_prefix=args.normalize_prefix)
    val_dataset = CocoDataset(prefix_length=args.prefix_length, normalize_prefix=args.normalize_prefix, type='val')
    model = PrefixModel(prefix_length=args.prefix_length, clip_length=args.prefix_length_clip)
    train(dataset, val_dataset, model)

if __name__=='__main__':
    main()