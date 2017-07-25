import torch 
import torch.nn as nn
import numpy as np
import os


num_data = 100        # number of data pair to generate (source, target)
source_path = './data/task1/source/'
target_path = './data/task1/target/'
if not os.path.exists(source_path):
    os.makedirs(source_path)
if not os.path.exists(target_path):
    os.makedirs(target_path)

# options
activations = ['nn.LeakyReLU(0.2)','nn.ReLU()','nn.PReLU()', 'nn.ELU()']
names = ['model','net','network','main']
input_dims = ['in_dim','x_dim','input_dim']
hidden_dims = [100, 200, 16, 32, 64, 128]
output_dims = ['out_dim']
final_functions = [',\n\tnn.Sigmoid()', ',\n\tnn.Tanh()',',\n\tnn.Softmax()', '']
cases = [[[1,1,1,1],[1,1,2,2]],
        [[1,1,1,1,1,1],[1,1,2,2,4,4],[1,1,2,2,1,1]],
        [[1,1,1,1,1,1,1,1],[1,1,2,2,4,4,2,2]]]


def get_random():
    # select options randomly
    activation = activations[np.random.randint(len(activations))]
    name = names[np.random.randint(len(names))]
    input_dim = input_dims[np.random.randint(len(input_dims))]
    hidden_dim = hidden_dims[np.random.randint(len(hidden_dims))]
    output_dim = output_dims[np.random.randint(len(output_dims))]
    final_function = final_functions[np.random.randint(len(final_functions))]
    return activation, name, input_dim, hidden_dim, output_dim, final_function

def change_case(case):
    # get wrong case
    case_wrong = case.copy()
    idx = np.random.randint(len(case))
    change_idx = np.random.randint(2)
    if case[idx]==1:
        case_wrong[idx]=[2,4][change_idx]
    elif case[idx]==2:
        case_wrong[idx]=[1,4][change_idx]
    elif case[idx]==4:
        case_wrong[idx]=[1,2][change_idx]
    return case_wrong

def get_cases(hidden_dim):
    # For given hidden_dim (e.g. 100), generate case_correct and case_wrong
    num_sets = np.random.randint(0,3)  # 3 ~ 5
    case_correct = cases[num_sets][np.random.randint(len(cases[num_sets]))]
    case_wrong = change_case(case_correct)
    
    case_correct = list(np.array(case_correct) * hidden_dim)
    case_wrong = list(np.array(case_wrong) * hidden_dim)
    return case_correct, case_wrong, num_sets

def generate_data(filename, input_dim, output_dim, case, num_sets, activation, final_function):
    with open(filename, 'w') as f:

        data = ""
        for i in range(num_sets + 3):
            # Input layer
            if i == 0:
                data += '{} = nn.Sequential(\n\tnn.Linear({}, {}),\n\t{},' \
                    .format(name, input_dim, case[i], activation)
            # Output layer
            elif i == (num_sets + 2):
                data += '\n\tnn.Linear({}, {}){})'.format(case[-1], output_dim, final_function)
            # Hidden layer
            else:
                data += '\n\tnn.Linear({}, {}),\n\t{},' \
                    .format(case[i*2 - 1], case[i*2], activation)

        # File write
        f.write(data)
        
for n in range(num_data):
    
    source_file = os.path.join(source_path, '{:05}.txt'.format(n+1))
    target_file = os.path.join(target_path, '{:05}.txt'.format(n+1))
    
    # Get random sets and correct / wrong case
    activation, name, input_dim, hidden_dim, output_dim, final_function = get_random()
    case_correct, case_wrong, num_sets = get_cases(hidden_dim)
    
    # Generate data
    generate_data(source_file, input_dim, output_dim, case_wrong, num_sets, activation, final_function)
    generate_data(target_file, input_dim, output_dim, case_correct, num_sets, activation, final_function)