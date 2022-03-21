import os, sys
import gc
import swifter

from sklearn.metrics import accuracy_score
import time

gc.collect()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  

import random
import numpy as np
# import jax.numpy as np
import pandas as pd

import torch
from torch import optim
import torch.nn.functional as F
torch.cuda.empty_cache()


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw')

stop_words_set = []
for w in stopwords.words('indonesian'):
    stop_words_set.append(w)

import math
import re
import copy

from operator import itemgetter
from deep_translator import GoogleTranslator

from utils.utils_init_dataset import set_seed, load_dataset_loader
from utils.utils_semantic_use import USE
from utils.utils_data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader, EmotionDetectionDataset, EmotionDetectionDataLoader
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, init_model, logit_prob, load_word_index
from utils.utils_forward_fn import forward_sequence_classification

# debugger
from icecream import ic

# !pip install pandarallel
from pandarallel import pandarallel
pandarallel.initialize()

from tqdm.notebook import tqdm
tqdm.pandas()
pd.set_option('display.max_colwidth', None)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")


model_target="IndoBERT"
downstream_task="emotion"
attack_strategy="adversarial"
finetune_epoch=15
num_sample=50
# exp_name=
perturbation_technique="codemixing"
perturb_ratio=0.4
perturb_lang="en"
seed=26092020




set_seed(seed)
# use = USE()

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  
tokenizer, config, model = init_model(model_target, downstream_task)
w2i, i2w = load_word_index(downstream_task)

train_dataset, train_loader, train_path = load_dataset_loader(downstream_task, 'train', tokenizer)
valid_dataset, valid_loader, valid_path = load_dataset_loader(downstream_task, 'valid', tokenizer)
test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)
# print(model.cuda())
finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, finetune_epoch)

# exp_dataset.to_csv(os.getcwd() + r'/result/'+exp_name+".csv", index=False)



model.eval()
torch.set_grad_enabled(False)

total_loss, total_correct, total_labels = 0, 0, 0
list_hyp, list_label = [], []

pbar = tqdm(test_loader, leave=True, total=len(test_loader))
for i, batch_data in enumerate(pbar):
    _, batch_hyp, _ = forward_sequence_classification(finetuned_model, batch_data[:-1], i2w=i2w, device='cuda')
    list_hyp += batch_hyp

# Save prediction
df = pd.DataFrame({'label':list_hyp}).reset_index()
df.to_csv('pred.txt', index=False)

print(df)