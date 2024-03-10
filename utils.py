import os
import random
import torch
import numpy as np

from unidecode import unidecode

def get_vocab(ascii_letters):
    vocab = dict()
    for i, letter in enumerate(ascii_letters + " .,:;-'"):
        vocab.update({letter:i})

    # Vocab
    vocab_size = len(vocab) 
    print(f'Total Vocab Size: {vocab_size}')
    
    return vocab, vocab_size
    
    
def get_lang2label(data_dir):
    all_files = os.listdir(data_dir)
    num_langs = len(all_files)
    lang2label  = {file_name.split('.')[0]: torch.tensor([i], dtype = torch.long) for i, file_name in enumerate(all_files)}
    
    total_names = 0
    for lang_file in all_files:
        with open(f"{data_dir}/{lang_file}", "r") as f:
            for name in f:
                total_names += 1
       

    #Output: {'Czech': tensor(0),
            #'German': tensor(1),...}
    print(f"Total Languages (classes): {num_langs}") 
    print(f"TOtal names in all languages: {total_names}")
    return lang2label, num_langs

#Function to create one-hot vectors for name
def name2tensor(name,vocab):
    '''
    Converts a name to a tensor of size -> (len(name),1,len(vocab))
    '''
    base_tensor = torch.zeros(len(name),1,  len(vocab))
    #*the extra dimension in the above tensor is bcos pytorch expects everything in a batch.
    for i, chars in enumerate(name):
        idx = vocab[chars]
        base_tensor[i][0][idx] = 1 
        
    return base_tensor


def create_dataset(data_dir, all_files, vocab, lang2label):
    names = 0
    c = 0
    tensor_names= []
    tensor_labels = []
    actual_names= []
    for file in all_files:
        with open(os.path.join(data_dir, file)) as f:
            lang = file.split('.')[0] #Arabic.txt -> Arabic
            names = [unidecode(name.rstrip()) for name in f] #All names for Arabic -> [name1, name2, ...]
            for name in names:
                c += 1
                try:
                    actual_names.append(name)
                    tensor_names.append(name2tensor(name, vocab)) # This is a one-hot vector for every character
                    tensor_labels.append(lang2label[lang])  #These are integer labels
                except KeyError:
                    print('Key Not Present')
                    print(name)
                    pass
    print(f'Total Names in all files: {c}')
    return actual_names, tensor_names, tensor_labels



def split_train_test(actual_names, tensor_names, tensor_labels, test_size = 0.1):
    total_data = list(zip(actual_names, tensor_names, tensor_labels))

    random.shuffle(total_data)
    
    print(len(total_data[:int(len(total_data)*(1-test_size))][0]))

    train_actual_names, train_tensor_names, train_tensor_labels = zip(*total_data[:int(len(total_data)*(1-test_size))])
    test_actual_names, test_tensor_names, test_tensor_labels = zip(*total_data[int(len(total_data)*(1-test_size)):])
    return train_actual_names, train_tensor_names, train_tensor_labels, \
        test_actual_names, test_tensor_names, test_tensor_labels