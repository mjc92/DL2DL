import json
import os
import argparse
from collections import Counter


def build_word2id(root, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        root: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    filenames = os.listdir(root)
    counter = Counter()
    
    for i, filename in enumerate(filenames):
        with open(os.path.join(root, filename), 'r') as f:
            sentence = f.read()
            sentence = sentence.replace('\n',' \n ').replace('\t',' \t ').replace('(',' ( ')
            sentence = sentence.replace(')',' ) ').replace(',',' , ').replace('nn.','nn . ')
            sentence = sentence.replace('  ',' ')
            counter.update(tokens)

        if i % 1000 == 0:
            print("[{}] Tokenized the sentences.".format(i))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<SOS>'] = 1
    word2id['<EOS>'] = 2
    word2id['<UNK>'] = 3
    
    # if word frequency is less than 'min_word_count', then the word is discarded
    words = [word for word, count in counter.items() if count >= min_word_count]
    
    # add the words to the word2id dictionary
    for i, word in enumerate(words):
        word2id[word] = i + 4
    
    return word2id


def main(config):
    
    # build word2id dictionaries for source and target sequences
    word2id = build_word2id(config.root, config.min_word_count)
    
    # save word2id dictionaries
    with open(config.output_path, 'w') as f:
        json.dump(word2id, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/task1/source/')
    parser.add_argument('--output_path', type=str, default='./data/word2id.json')
    parser.add_argument('--min_word_count', type=int, default=1)
    config = parser.parse_args()
    print (config)
    main(config)