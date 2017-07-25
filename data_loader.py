import torch
import os
from torch.utils import data


"""
Sample usage case

src_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/source/'
trg_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/target/'
data_loader = get_loader(src_root, trg_root, 2)
src,trg = dat_iter.next()

"""


class TextFolder(data.Dataset):
    def __init__(self, src_root, trg_root):
        """Initializes paths and preprocessing module"""
        self.src_root = src_root
        self.trg_root = trg_root
        self.max_len = 100
        self.source_filenames = os.listdir(src_root)
        self.target_filenames = os.listdir(trg_root)
        self.load_dict()
    
    def __getitem__(self, index):
        source_filename = self.source_filenames[index]
        target_filename = self.target_filenames[index]
        
        with open(os.path.join(self.src_root,source_filename)) as f:
            src = f.read()
        with open(os.path.join(self.trg_root,target_filename)) as f:
            trg = f.read()
            
        # tokenize source and target
        src_tokens = self.tokenize(self.preprocess(src))
        trg_tokens = self.tokenize(self.preprocess(trg),trg=True)
        
        # change lists of words to lists of idxs
        src_tokens = self.wordlist2idxlist(src_tokens)
        trg_tokens = self.wordlist2idxlist(trg_tokens)
        return src_tokens, trg_tokens
    
    def __len__(self):
        return len(self.source_filenames)
    
    def preprocess(self, string):
        string = string.replace('\n',' \n ').replace('\t',' \t ').replace('(',' ( ')
        string = string.replace(')',' ) ').replace(',',' , ').replace('nn.','nn . ')
        string = string.replace('  ',' ')
        return string.strip()
    
    def load_dict(self):
        # load dictionary
        import json
        with open('data/word2id.json') as f:
            txt = f.read()
            self.w2i = json.loads(txt)
            
    def wordlist2idxlist(self,wordlist):
        out = []
        for word in wordlist:
            if word in self.w2i:
                out.append(self.w2i[word])
            else:
                out.append(self.w2i['<UNK>'])
        return out

    def tokenize(self, string, trg=False):
        out = string.split(' ')
        if trg:
            out = ['<SOS>'] + out + ['<EOS>']
#         while(len(out)<self.max_len):
#             out.append('<PAD>')
#         return out[:self.max_len]
        return torch.LongTensor(get_vocab(out))

def collate_fn(data):
    sources, targets = zip(*data)
    source_lengths = [len(x) for x in sources]
    target_lengths = [len(x) for x in targets]
    sources_out = torch.zeros(len(sources),max(source_lengths)).long()
    targets_out = torch.zeros(len(targets),max(target_lengths)).long()
    for i in range(len(sources)):
        source_end = source_lengths[i]
        target_end = target_lengths[i]
        sources_out[i,:source_end] = sources[i]
        targets_out[i,:target_end] = targets[i]
#     prin(len(sources))
#     print(sources)
    return sources_out, targets_out, source_lengths, target_lengths

def get_loader(src_root, trg_root, batch_size, num_workers=2):
    dataset = TextFolder(src_root, trg_root)
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader

