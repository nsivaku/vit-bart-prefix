import torch
from torch.utils.data import Dataset
from transformers import BartTokenizerFast
import pickle
from typing import Tuple
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class CocoDataset(Dataset):
    
    def __init__(self, prefix_length: int, normalize_prefix=False, type='train'):
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        if type == 'train':
            with open('./data/coco/oscar_split_train.pkl', 'rb') as f:
                all_data = CPU_Unpickler(f).load()
        if type == 'val':
            with open('./data/coco/oscar_split_val.pkl', 'rb') as f:
                all_data = CPU_Unpickler(f).load()
        self.prefixes = all_data['embedding']
        anns = all_data['captions']
        self.image_ids = [ann['image_id'] for ann in anns]
        self.captions = [ann['caption'] for ann in anns]
        
        if type == 'train':
            with open('./data/coco/oscar_split_tokens_train.pkl', 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        if type == 'val':
            with open('./data/coco/oscar_split_tokens_val.pkl', 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64))) # 1 is pad_token_id for bart
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, prefix
    
    def __len__(self) -> int:
        return len(self.captions_tokens)