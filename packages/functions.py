import torch
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def reset_grad(model):
    model.zero_grad()
    
def extract(text): # extracts model structure from source code
    import re
    text = text.split('\n')
    numbers = []
    activation = []
    for line in text:
        if 'Leaky' not in line:
            numbers.extend(re.findall(r'\d+',line))
        if ('LU' in line):
            activation.append(line.split('(')[0].strip())
    numbers = [x for i,x in enumerate(numbers) if i%2==0]
    activation = list(set(activation))[0]
    result = line.split('(')[0].strip()
    return ' '.join(numbers), activation, result