import torch
import torch.nn as nn
import json
from model import Seq2Seq
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from data_loader import get_loader


with open('./data/word2id.json', 'r') as f:
    word2id = json.load(f)
    
src_root = './data/task1/source/'
trg_root = './data/task1/target/'
data_loader = get_loader(src_root, trg_root, 10, num_workers=2)

vocab_size = len(word2id)
model = Seq2Seq(vocab_size)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def reset_grad():
    model.zero_grad()

model.cuda()

log_step = 100
num_epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for e in range(num_epochs):
    for i, (sources, targets, source_lengths, target_lengths) in enumerate(data_loader):
        
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
        reset_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % log_step:
            print ("Epoch [{}/{}], Step [{}/{}], loss: {:.4}" \
                   .format(e+1, num_epochs, i+1, len(data_loader), loss.data[0]))

torch.save(model.state_dict(), './data/model.pth')