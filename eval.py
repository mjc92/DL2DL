import torch
import json
import numpy as np
import os
from models.seq2seq import Seq2Seq
from torch.autograd import Variable
from data_loader import get_loader

# sample path
sample_path = './data/task1/predicted'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

# load vocabulary file and loader
with open('./data/word2id.json', 'r') as f:
    word2id = json.load(f)
id2word = {i: w for w, i in word2id.items()}
src_root = './data/task1/source/'
trg_root = './data/task1/target/'
data_loader = get_loader(src_root, trg_root, 10, num_workers=2, shuffle=False)

model_save_path = './data/model.pth'
vocab_size = len(word2id)
model = Seq2Seq(vocab_size)
model.cuda()
model.load_state_dict(torch.load(model_save_path))

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

outputs = []
filenames = []
total_samples = 0
correct_samples = 0
for i, (sources, targets, source_lengths, target_lengths, names) in enumerate(data_loader):
    
    sources = to_var(sources, volatile=True)
    targets_out = targets[:, 1:]
    max_length = targets_out.size(1)
    
    # Forward pass
    h, c = model.encode(sources, source_lengths)
    batch_size = sources.size(0)
    input = to_var(torch.LongTensor([word2id['<SOS>']] * batch_size)).unsqueeze(1)  # (batch_size, 1)
    output = model.sample(input, h, c, 130)
    
    outputs.append(output)
    filenames.extend(names)
    
    # Compute accuracy
    mask = (targets_out != 0).float()
    correct = (output[:,:mask.size(1)].data.cpu() == targets_out).float() * mask  
    # (batch_size, max_length)
    mask_sum = torch.sum(mask, dim=1)                            
    # (batch_size)
    correct_sum =torch.sum(correct, dim=1)
    score = (mask_sum == correct_sum).numpy().sum()
    total_samples+=batch_size
    correct_samples+=score
print('Total score: %d/%d' %(correct_samples,total_samples))
    
predicted = torch.cat(outputs, dim=0)
predicted = predicted.cpu().data.numpy()

num_data = predicted.shape[0]

sampled = []
for i, sentence in enumerate(predicted):
    tmp = ""
    for idx in sentence:
        token = id2word[idx]
        if token == '<EOS>':
            break
        tmp += token
    sampled.append(tmp)

    with open(os.path.join(sample_path, '{:05}.txt'.format(filenames[i])), 'w') as f:
        f.write(tmp)