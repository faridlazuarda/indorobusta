from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, init_model, logit_prob, load_word_index
import time
from nltk.tokenize import word_tokenize
import torch
import torch.nn.functional as F
import numpy as np
import math
from .attack_helper import get_synonyms, codemix_perturbation, synonym_replacement, swap_minimum_importance_words

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words_set = []
for w in stopwords.words('indonesian'):
    stop_words_set.append(w)

    
def attack(text_ls,
           true_label,
           predictor,
           tokenizer,
           att_ratio,
           attack_strategy,
           lang_codemix=None,
           sim_predictor=None,
           sim_score_threshold=0.5,
           sim_score_window=15,
           batch_size=32, 
           import_score_threshold=-1.):
    
    start_time = time.time()
    
    label_dict = {
        'positive': 0, 
        'neutral': 1, 
        'negative': 2}
    
    original_text = text_ls
    orig_label, orig_probs, orig_prob = logit_prob(text_ls, predictor, tokenizer)
        
#     SEK SALAAHHHHHH
    if true_label != orig_label:
        running_time = round(time.time() - start_time, 2)
        # perturbed_text, perturbed_semantic_sim, orig_label, orig_prob, perturbed_label, perturbed_prob, running_time
        return original_text, 1.000, orig_label, orig_prob, orig_label, orig_prob, running_time
    else:
        text_ls = word_tokenize(text_ls)
        text_ls = [word for word in text_ls if word.isalnum()]
        len_text = len(text_ls)
        half_sim_score_window = (sim_score_window - 1) // 2
        # num_queries = 1
        
        leave_1_texts = [' '.join(text_ls[:ii] + [tokenizer.mask_token] + text_ls[min(ii + 1, len_text):]) for ii in range(len_text)]
                
        leave_1_probs = []
        leave_1_probs_argmax = []
        # num_queries += len(leave_1_texts)
        for text_leave_1 in leave_1_texts:
            subwords_leave_1 = tokenizer.encode(text_leave_1)
            subwords_leave_1 = torch.LongTensor(subwords_leave_1).view(1, -1).to(predictor.device)
            logits_leave_1 = predictor(subwords_leave_1)[0]
            orig_label_leave_1 = torch.topk(logits_leave_1, k=1, dim=-1)[1].squeeze().item()
            
            leave_1_probs_argmax.append(orig_label_leave_1)
            leave_1_probs.append(F.softmax(logits_leave_1, dim=-1).squeeze().detach().cpu().numpy())
            
        leave_1_probs = torch.tensor(leave_1_probs).to("cuda:1")
        
        orig_prob_extended=np.empty(len_text)
        orig_prob_extended.fill(orig_prob)
        orig_prob_extended = torch.tensor(orig_prob_extended).to("cuda:1")
        
        arr1 = orig_prob_extended - leave_1_probs[:,orig_label] + float(leave_1_probs_argmax != orig_label)
        arr2 = (leave_1_probs.max(dim=-1)[0].to("cuda:1") - orig_probs[leave_1_probs_argmax].to("cuda:1"))
        
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
        
        if num_perturbation < 1:
            num_perturbation = 1
        
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
        
        perturbed_semantic_sim = sim_predictor.semantic_sim(original_text, perturbed_text)
        if perturbed_semantic_sim < sim_score_threshold:
            perturbed_text = original_text
            perturbed_semantic_sim = 1.000
        
        perturbed_label, perturbed_probs, perturbed_prob = logit_prob(perturbed_text, predictor, tokenizer)
        
        running_time = round(time.time() - start_time, 2)
        
        return perturbed_text, perturbed_semantic_sim, orig_label, orig_prob, perturbed_label, perturbed_prob, running_time