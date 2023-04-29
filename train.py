from dataset import CocoDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as nnf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BartForConditionalGeneration, get_linear_schedule_with_warmup
from typing import Optional, Tuple
import argparse
import os
import sys

# class MlpTransformer(nn.Module):
#     def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
#         super().__init__()
#         out_d = out_d if out_d is not None else in_dim
#         self.fc1 = nn.Linear(in_dim, h_dim)
#         self.act = act
#         self.fc2 = nn.Linear(h_dim, out_d)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class MultiHeadAttention(nn.Module):

#     def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim_self // num_heads
#         self.scale = head_dim ** -0.5
#         self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
#         self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
#         self.project = nn.Linear(dim_self, dim_self)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, y=None, mask=None):
#         y = y if y is not None else x
#         b, n, c = x.shape
#         _, m, d = y.shape
#         # b n h dh
#         queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
#         # b m 2 h dh
#         keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
#         keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
#         attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
#         if mask is not None:
#             if mask.dim() == 2:
#                 mask = mask.unsqueeze(1)
#             attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
#         attention = attention.softmax(dim=2)
#         out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
#         out = self.project(out)
#         return out, attention

# class TransformerLayer(nn.Module):

#     def forward_with_attention(self, x, y=None, mask=None):
#         x_, attention = self.attn(self.norm1(x), y, mask)
#         x = x + x_
#         x = x + self.mlp(self.norm2(x))
#         return x, attention

#     def forward(self, x, y=None, mask=None):
#         x = x + self.attn(self.norm1(x), y, mask)[0]
#         x = x + self.mlp(self.norm2(x))
#         return x

#     def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
#                  norm_layer: nn.Module = nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim_self)
#         self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
#         self.norm2 = norm_layer(dim_self)
#         self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

# class Transformer(nn.Module):

#     def forward_with_attention(self, x, y=None, mask=None):
#         attentions = []
#         for layer in self.layers:
#             x, att = layer.forward_with_attention(x, y, mask)
#             attentions.append(att)
#         return x, attentions

#     def forward(self, x, y=None, mask=None):
#         for i, layer in enumerate(self.layers):
#             if i % 2 == 0 and self.enc_dec: # cross
#                 x = layer(x, y)
#             elif self.enc_dec:  # self
#                 x = layer(x, x, mask)
#             else:  # self or cross
#                 x = layer(x, y, mask)
#         return x

#     def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
#                  mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
#         super(Transformer, self).__init__()
#         dim_ref = dim_ref if dim_ref is not None else dim_self
#         self.enc_dec = enc_dec
#         if enc_dec:
#             num_layers = num_layers * 2
#         layers = []
#         for i in range(num_layers):
#             if i % 2 == 0 and enc_dec:  # cross
#                 layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
#             elif enc_dec:  # self
#                 layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
#             else:  # self or cross
#                 layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
#         self.layers = nn.ModuleList(layers)

# class TransformerMapper(nn.Module):

#     def forward(self, x):
#         x = self.linear(x).view(x.shape[0], self.clip_length, -1)
#         prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
#         prefix = torch.cat((x, prefix), dim=1)
#         out = self.transformer(prefix)[:, self.clip_length:]
#         return out

#     def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
#         super(TransformerMapper, self).__init__()
#         self.clip_length = clip_length
#         self.transformer = Transformer(dim_embedding, num_layers, num_layers)
#         self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
#         self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


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

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # embedding_text = self.bart.model.shared(tokens)
        prefix_projections = self.img_project(prefix).view(-1, self.prefix_length, self.bart_embedding_size)
        # embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        # decoder_input_ids = torch.ones((embedding_cat.shape[0], 1), dtype=torch.long) * self.bart.config.decoder_start_token_id
        # decoder_input_ids = decoder_input_ids.to(self.device)
        labels = torch.clone(tokens)
        labels[labels[:, :] == self.bart.config.pad_token_id] = -100
        decoder_input_ids = shift_tokens_right(labels, self.bart.config.pad_token_id)
        out = self.bart(inputs_embeds=prefix_projections, labels=tokens)
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
        
def train(dataset: CocoDataset, model: PrefixModel, lr: float = 2e-5, warmup_steps: int = 3000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    epochs = 5
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch + 1}")
        model.train()
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc='vit_bart_latest')
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            loss = outputs.loss
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join("./checkpoints", "vit_bart_latest.pt"),
                )
        progress.close()
        if epoch % 1 == 0 or epoch == epochs - 1:
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
    model = PrefixModel(prefix_length=args.prefix_length, clip_length=args.prefix_length_clip)
    train(dataset, model)

if __name__=='__main__':
    main()