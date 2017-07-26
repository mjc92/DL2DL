import torch
import torch.nn as nn
import json
from models.seq2seq import Seq2Seq
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
# from data_loader import get_loader
from data_loader import get_loader
from packages.functions import reset_grad, to_var
import argparse
import os


parser = argparse.ArgumentParser()
# parser.add_argument("--vocab_size",type=int, default=100, help='vocab_size')
parser.add_argument("--src_root",type=str, default='./data/task1/source/', help='directory for source data')
parser.add_argument("--trg_root",type=str, default='./data/task1/target/', help='directory for target data')
parser.add_argument("--saved_dir",type=str, default='./data/', help='directory for saved data')
parser.add_argument("--model_name",type=str, default='model.pth', help='file name of saved model')
parser.add_argument("--num_epochs",type=int, default=10, help='number of epochs')
parser.add_argument("--log_step",type=int, default=100, help='number of steps to write on log')
parser.add_argument("--batch_size",type=int, default=10, help='minibatch size')
parser.add_argument("--embed_size",type=int, default=128, help='embedding size')
parser.add_argument("--hidden_size",type=int, default=512, help='hidden size')
parser.add_argument("--lr",type=float, default=0.001, help='learning rate')
args = parser.parse_args()

with open('./data/word2id.json', 'r') as f:
    word2id = json.load(f)
    
data_loader = get_loader(args.src_root, args.trg_root, args.batch_size, num_workers=2)

vocab_size = len(word2id)
model = Seq2Seq(vocab_size=vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size)

model.cuda()

log_step = args.log_step
num_epochs = args.num_epochs

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for e in range(num_epochs):
    for i, (sources, targets, source_lengths, target_lengths,_) in enumerate(data_loader):
        
        # convert tensor to variable
        sources = to_var(sources)
        targets_in = to_var(targets[:, :-1])
        targets_out = to_var(targets[:, 1:])
        target_lengths = [x-1 for x in target_lengths] 
        
        # Forward pass
        h, c = model.encode(sources, source_lengths)
        output = model.decode(targets_in, target_lengths, h, c)
        
        # Compute loas
        targets_out = pack(targets_out, target_lengths, batch_first=True)[0]
        loss = criterion(output, targets_out)
        
        # Backward + Optimize
        reset_grad(model)
        loss.backward()
        optimizer.step()
        
        if (i+1) % log_step:
            print ("Epoch [{}/{}], Step [{}/{}], loss: {:.4}" \
                   .format(e+1, num_epochs, i+1, len(data_loader), loss.data[0]))

torch.save(model.state_dict(), os.path.join(args.saved_dir,args.model_name))