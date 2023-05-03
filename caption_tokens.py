import pickle
import torch
from transformers import BartTokenizerFast

with open('./data/coco/oscar_split_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('./data/coco/oscar_split_val.pkl', 'rb') as f:
    val_data = pickle.load(f)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

captions_tokens = []
caption2embedding = []
max_seq_len = 0

for caption in train_data['captions']:
    captions_tokens.append(torch.tensor(tokenizer.encode(caption['caption']), dtype=torch.int64))
    caption2embedding.append(caption['embedding'])
    max_seq_len = max(max_seq_len, captions_tokens[-1].shape[0])

with open("./data/coco/oscar_split_tokens_train.pkl", 'wb') as f:
    pickle.dump([captions_tokens, caption2embedding, max_seq_len], f)
    
captions_tokens = []
caption2embedding = []
max_seq_len = 0

for caption in val_data['captions']:
    captions_tokens.append(torch.tensor(tokenizer.encode(caption['caption']), dtype=torch.int64))
    caption2embedding.append(caption['embedding'])
    max_seq_len = max(max_seq_len, captions_tokens[-1].shape[0])

with open("./data/coco/oscar_split_tokens_val.pkl", 'wb') as f:
    pickle.dump([captions_tokens, caption2embedding, max_seq_len], f)