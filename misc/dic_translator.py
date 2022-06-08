import numpy as np
import pandas as pd

from deep_translator import GoogleTranslator
import argparse
from tqdm import tqdm
import ast
import os
import re

def file_reader(path):
    a_file = open(path, "r")

    vocab = []
    for line in a_file:
        stripped_line = line.strip()
        vocab.append(stripped_line)

    a_file.close()
    return list(set(vocab))
    # print(list_of_lists)

def trans_dict(wordlist, stored, lang=None):
    # print(stored)
    translated_wordlist = {}
    # ic(wordlist)
    supported_langs = ["su", "jw", "ms", "en", "fr", "it"]
    
    if lang not in supported_langs:
        raise ValueError('Language Unavailable')

    translator = GoogleTranslator(source="id", target=lang)
    
    alpha = []
    non_alpha = {}
    translated_store = {}
    for word in wordlist:
        if word in stored.keys():
            translated_store[word] = word
        
        else:
            word = word.lower()
            word = re.sub(r'[^A-Za-z0-9 ]+', '', word)

            if word.isdigit():
                non_alpha[word] = word
            elif any(c.isalpha() for c in word):
                alpha.append(word)
            else:
                non_alpha[word] = word
    
    trans_alpha = translator.translate_batch(alpha)
    translated_alpha = dict(zip(alpha, translator.translate_batch(trans_alpha)))
    
    translated_wordlist = {**non_alpha, **translated_alpha, **translated_store}
    # if len(translated_wordlist)%10 == 0:
    return translated_wordlist


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", required=True, type=str)

    return parser.parse_args()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    splits = []
    for i in range(0, len(lst), n):
        splits.append(lst[i:i + n])
    return splits

def read_dict(filename):
    with open(filename) as f:
        data = f.read()
      
    # reconstructing the data as a dictionary
    d = ast.literal_eval(data)
    # print(type(d))
    return d

if __name__ == "__main__":
    args = get_args()
    
    unique = file_reader("allvocab.txt")
    splitted_vocab = chunks(unique,1000)
    
    filename = "new_dict_"+str(args.lang)+".txt"
    with open(filename) as f:
        data = f.read()
        
    stored_filename = os.getcwd() + r"/dicts/dict_"+str(args.lang)+".txt"
    stored_dict = read_dict(stored_filename)
    
        
    cache = ast.literal_eval(data)
    len_splitted = len(cache)//1000
    if len_splitted > 0:
        total_trans = cache
        splitted_vocab = splitted_vocab[len_splitted:]
    else:
        total_trans = {}

    for split in tqdm(splitted_vocab):
        trans = trans_dict(
            wordlist = split,
            stored = stored_dict,
            lang = args.lang
        )
        total_trans = {**total_trans, **trans}
        with open(filename, 'w', encoding="utf-8") as f:
            print(total_trans, file=f)