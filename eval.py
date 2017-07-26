import torch
import argparse
import json
import numpy as np
import os
from models.seq2seq import Seq2Seq
from torch.autograd import Variable
from data_loader import get_loader

parser = argparse.ArgumentParser()
# parser.add_argument("--vocab_size",type=int, default=100, help='vocab_size')
parser.add_argument("--src_root",type=str, default='./data/task1/source/', help='directory for source data')
parser.add_argument("--trg_root",type=str, default='./data/task1/target/', help='directory for target data')
parser.add_argument("--sample_dir",type=str, default='./data/task1/predicted', help='directory save sampled data')
parser.add_argument("--saved_dir",type=str, default='./data/', help='directory for saved data')
parser.add_argument("--model_name",type=str, default='model.pth', help='file name of saved model')
parser.add_argument("--num_epochs",type=int, default=10, help='number of epochs')
parser.add_argument("--log_step",type=int, default=100, help='number of steps to write on log')
parser.add_argument("--batch_size",type=int, default=10, help='minibatch size')
parser.add_argument("--embed_size",type=int, default=128, help='embedding size')
parser.add_argument("--hidden_size",type=int, default=512, help='hidden size')
parser.add_argument("--lr",type=float, default=0.001, help='learning rate')
args = parser.parse_args()

# sample path
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)

# load vocabulary file and loader
with open('./data/word2id.json', 'r') as f:
    word2id = json.load(f)
id2word = {i: w for w, i in word2id.items()}
data_loader = get_loader(args.src_root, args.trg_root, args.batch_size, 
                         num_workers=2, shuffle=False)

vocab_size = len(word2id)
model = Seq2Seq(vocab_size)
model.cuda()
model.load_state_dict(torch.load(os.path.join(args.saved_dir,args.model_name)))

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

    with open(os.path.join(args.sample_dir, '{:05}.txt'.format(filenames[i])), 'w') as f:
        f.write(tmp)