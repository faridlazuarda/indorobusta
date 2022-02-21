import os, sys
import gc

gc.collect()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'tugas-akhir-repository/')

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F

torch.cuda.empty_cache()

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

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
from utils.utils_forward_fn import forward_sequence_classification
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, init_model

# debugger
from icecream import ic

from tqdm.notebook import tqdm
tqdm.pandas()
pd.set_option('display.max_colwidth', None)



def codemix_perturbation(words, target_lang, words_perturb):
    """
    'su': 'sundanese'
    'jw': 'javanese
    'ms': 'malay','
    'en': 'english',
    """
    
    translator = GoogleTranslator(source='id', target=target_lang)
    
    supported_langs = ["su", "jw", "ms", "en"]
    
    if target_lang not in supported_langs:
        raise ValueError('Language Unavailable')
    
    new_words = words.copy()
    
    if len(words_perturb) >= 1:
        for perturb_word in words_perturb:
            new_words = [translator.translate(word) if word == perturb_word[1] and word.isalpha() else word for word in new_words]

    sentence = ' '.join(new_words)

    return sentence

def synonym_replacement(words, words_perturb):
    return None

# fungsi untuk mencari kandidat lain ketika sebuah kandidat perturbasi kurang dari sim_score_threshold
def swap_minimum_importance_words(words_perturb, top_importance_words):
    def get_minimums(word_tups):
        arr = []
        for wt in word_tups:
            if wt[2] == min(top_importance_words, key = lambda t: t[2])[2]:
                arr.append(wt)
        return arr
    minimum_import = get_minimums(top_importance_words)
    unlisted = list(set(words_perturb).symmetric_difference(set(top_importance_words)))

    len_wp = len(top_importance_words)
    len_ul = len(unlisted)
    
    res = []
    for i in range(len_wp):
        if top_importance_words[i] in minimum_import:
            temp_wp = list(copy.deepcopy(top_importance_words))
            temp_wp.pop(i)
            swapped_wp = np.array([(temp_wp) for i in range(len_ul)])
            for j in range(len(swapped_wp)):
                temp_sm = np.vstack((swapped_wp[j], tuple(unlisted[j])))
                
                res.append(temp_sm.tolist())
                
    return res

def attack(text_ls,
           true_label,
           predictor,
           tokenizer,
           att_ratio,
           lang_codemix,
           attack_strategy,
           sim_predictor=None,
           sim_score_threshold=0.5,
           sim_score_window=15,
           batch_size=32, 
           import_score_threshold=-1.):
    
    label_dict = {
        'positive': 0, 
        'neutral': 1, 
        'negative': 2}
    
    original_text = text_ls
    subwords = tokenizer.encode(text_ls)
    subwords = torch.LongTensor(subwords).view(1, -1).to(predictor.device)

    logits = predictor(subwords)[0]
    orig_label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    
    orig_probs = F.softmax(logits, dim=-1).squeeze()
    orig_prob = F.softmax(logits, dim=-1).squeeze()[orig_label].detach().cpu().numpy()
        
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        text_ls = word_tokenize(text_ls)
        text_ls = [word for word in text_ls if word.isalnum()]
        len_text = len(text_ls)
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        
        leave_1_texts = [' '.join(text_ls[:ii] + [tokenizer.mask_token] + text_ls[min(ii + 1, len_text):]) for ii in range(len_text)]
                
        leave_1_probs = []
        leave_1_probs_argmax = []
        num_queries += len(leave_1_texts)
        for text_leave_1 in leave_1_texts:
            subwords_leave_1 = tokenizer.encode(text_leave_1)
            subwords_leave_1 = torch.LongTensor(subwords_leave_1).view(1, -1).to(predictor.device)
            logits_leave_1 = predictor(subwords_leave_1)[0]
            orig_label_leave_1 = torch.topk(logits_leave_1, k=1, dim=-1)[1].squeeze().item()
            
            leave_1_probs_argmax.append(orig_label_leave_1)
            leave_1_probs.append(F.softmax(logits_leave_1, dim=-1).squeeze().detach().cpu().numpy())
            
        leave_1_probs = torch.tensor(leave_1_probs).to("cuda:0")
        
        orig_prob_extended=np.empty(len_text)
        orig_prob_extended.fill(orig_prob)
        orig_prob_extended = torch.tensor(orig_prob_extended).to("cuda:0")
        
        arr1 = orig_prob_extended - leave_1_probs[:,orig_label] + float(leave_1_probs_argmax != orig_label)
        arr2 = (leave_1_probs.max(dim=-1)[0].to("cuda:0") - orig_probs[leave_1_probs_argmax].to("cuda:0"))
        
        import_scores = arr1*arr2
        import_scores = [im * -1 for im in import_scores]
        
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx], score.item()))
            except Exception as e:
                print(e)
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
        
        num_perturbation = math.floor(len(words_perturb)*att_ratio)
        
#       top words perturb berisi list kata terpenting yang tidak akan diswitch ketika first_codemix_sim_score < sim_score_threshold
        top_words_perturb = words_perturb[:num_perturbation]
        
        
        if attack_strategy == "codemixing":
            perturbed_text = codemix_perturbation(text_ls, lang_codemix, words_perturb)
        elif attack_strategy == "synonym_replacement":
            perturbed_text = synonym_replacement(text_ls, words_perturb)
        
        first_perturbation_sim_score = sim_predictor.semantic_sim(original_text, perturbed_text)
                
#       cek semantic similarity
#       kalo top wordsnya cuma 1 diskip
        if len(top_words_perturb) > 1:
            words_perturb_candidates = []
            if first_perturbation_sim_score < sim_score_threshold:
                words_perturb_candidates.append(top_words_perturb)
                swapped = swap_minimum_importance_words(words_perturb, top_words_perturb)
                for s in swapped:
                    words_perturb_candidates.append(s)

                words_perturb_candidates = [[tuple(w) for w in wpc] for wpc in words_perturb_candidates]

                candidate_comparison = {}
                for wpc in words_perturb_candidates:
                    if attack_strategy == "codemixing":
                        perturbed_candidate = codemix_perturbation(text_ls, lang_codemix, words_perturb)
                    elif attack_strategy == "synonym_replacement":
                        perturbed_candidate = synonym_replacement(text_ls, words_perturb)
                    
                    perturbed_candidate_sim_score = sim_predictor.semantic_sim(original_text, perturbed_candidate)
                    candidate_comparison[perturbed_candidate] = (perturbed_candidate_sim_score, wpc[-1][-1])

                sorted_candidate_comparison = sorted(candidate_comparison.keys(), key=lambda x: (candidate_comparison[x][0], candidate_comparison[x][1]), reverse=True)
                perturbed_text = sorted_candidate_comparison[0]
        else:
            if first_perturbation_sim_score < sim_score_threshold:
                perturbed_text = original_text
        
        if sim_predictor.semantic_sim(original_text, perturbed_text) < sim_score_threshold:
            perturbed_text = original_text
        
        return perturbed_text

def run_codemixing():
    return None

def run_synonym_replacement():
    return None

def main(
    model_target,
    downstream_task,
    attack_strategy,
    perturbation_technique,
    perturb_ratio,
    num_sample,
    seed=26092020
):
    set_seed(seed)

    use = USE()

    print("\nModel initialization..")
    tokenizer, config, model = init_model(model_target)
    
    if downstream_task == "sentiment":
        w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL
        print("\nLoading dataset..")
        train_dataset, train_loader = load_dataset_loader(downstream_task, 'train', tokenizer)
        valid_dataset, valid_loader = load_dataset_loader(downstream_task, 'valid', tokenizer)
        test_dataset, test_loader = load_dataset_loader(downstream_task, 'test', tokenizer)
        
        text0 = 'lokasi di alun alun masakan padang ini cukup terkenal dengan kepala ikan kakap gule , biasa saya pesan nasi bungkus padang berisikan rendang , ayam pop dan perkedel . porsi banyak dan mengenyangkan'
        text1 = 'meski masa kampanye sudah selesai , bukan berati habis pula upaya mengerek tingkat kedipilihan elektabilitas .'
        text2 = 'kamar nya sempit tidak ada tempat menyimpan barang malah menambah barang . by the way ini kipas2 mau diletakkan mana . mana uchiwa segede ini pula .'


        print("\nTest initial model on sample text..")
        text_logit(text0, model, tokenizer, i2w)
        text_logit(text1, model, tokenizer, i2w)
        text_logit(text2, model, tokenizer, i2w)
        
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, 5)
        
        print("\nAttacking text using codemixing...")
        codemixed0 = attack(text0, 0, finetuned_model, tokenizer, perturb_ratio, 'jw', attack_strategy, sim_predictor=use)
        codemixed1 = attack(text1, 1, finetuned_model, tokenizer, perturb_ratio, 'en', attack_strategy, sim_predictor=use)
        codemixed2 = attack(text2, 2, finetuned_model, tokenizer, perturb_ratio, 'su', attack_strategy, sim_predictor=use)

        print("\nCalculating logit on codemixed data...")
        text_logit(codemixed0, finetuned_model, tokenizer, i2w)
        text_logit(codemixed1, finetuned_model, tokenizer, i2w)
        text_logit(codemixed2, finetuned_model, tokenizer, i2w)
        
        print("\nCalculating similarity score...")
        print(use.semantic_sim(text0, codemixed0))
        print(use.semantic_sim(text1, codemixed1))
        print(use.semantic_sim(text2, codemixed2))
        
    elif downstream_task == "emotion":
        w2i, i2w = EmotionDetectionDataset.LABEL2INDEX, EmotionDetectionDataset.INDEX2LABEL
        train_dataset, train_loader = load_dataset_loader(downstream_task, 'train', tokenizer)
        valid_dataset, valid_loader = load_dataset_loader(downstream_task, 'valid', tokenizer)
        test_dataset, test_loader = load_dataset_loader(downstream_task, 'test', tokenizer)
        
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, 5)
        
        # prob_before = 

if __name__ == "__main__":
    main(
        model_target="IndoBERT",
        downstream_task="sentiment",
        attack_strategy="codemixing",
        perturbation_technique="adversarial",
        perturb_ratio=0.2,
        num_sample=0,
        seed=26092020
    )

