import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import copy
from deep_translator import GoogleTranslator
import numpy as np
import os
import ast

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('omw')
# nltk.download('stopwords')

stop_words_set = []
for w in stopwords.words('indonesian'):
    stop_words_set.append(w)

from icecream import ic
    

def get_synonyms(word):
    if word in stop_words_set:
        return [word]

    word_lemmas = wordnet.lemmas(word, lang="ind")
    
    hypernyms = []
    for lem in word_lemmas:
        hypernyms.append(lem.synset().hypernyms())

    if not any(hypernyms):
        return [word]
    
    lemma_corp = []
    
    for hypernym in hypernyms:
        if(len(hypernym) < 1):
            continue
        else:
            lemma_corp.append(hypernym[0].lemmas(lang="ind"))
            
    lemmas = set()
    for list_lemmas in lemma_corp:
        if(len(list_lemmas) < 1):
            lemmas.add(word)
        else:
            for l in list_lemmas:
                lemmas.add(l.name())
    
    clean_synonyms = set()
    for syn in lemmas.copy():
        synonym = syn.replace("_", " ").replace("-", " ").lower()
        synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
        clean_synonyms.add(synonym) 
    if word in clean_synonyms:
        clean_synonyms.remove(word)
    
    if len(list(clean_synonyms)) < 1:
        return [word]
    else:
        return list(clean_synonyms)

def read_dict(filename):
    with open(filename) as f:
        data = f.read()
      
    # reconstructing the data as a dictionary
    d = ast.literal_eval(data)
    # print(type(d))
    return d

def translate_batch(wordlist,translator, target_lang):
    translated = []
    # print(wordlist)
    for w in wordlist:
        # print(translator[w])
        if w in translator.keys():
            if translator[w] is None:
                online_trans = GoogleTranslator(source="id", target=target_lang)
                trans = online_trans.translate(w)
            else:
                trans = translator[w]
        else:
            online_trans = GoogleTranslator(source="id", target=target_lang)
            trans = online_trans.translate(w)
        if not trans.isalpha():
            trans = w
        translated.append(trans)
    # ic(translated)
    return translated

def codemix_perturbation_cache(words, target_lang, words_perturb):
    translator = read_dict(os.getcwd() + r"/dicts/dict_"+target_lang+".txt")
    new_wp = []
    for wp in words_perturb:
        if wp[1].isalpha():
            new_wp.append(wp[1])
    
    # ic(new_wp)
    if len(new_wp) == 1:
        if new_wp[0] in translator.keys():
            new_wp_trans = dict(zip(new_wp, translator[new_wp[0]]))
        else:
            online_trans = GoogleTranslator(source="id", target=target_lang)
            new_wp_trans = dict(zip(new_wp, online_trans.translate(new_wp[0])))
    elif len(new_wp) == 0:
        return ' '.join(words), {}
        
    new_wp_trans = dict(zip(new_wp, translate_batch(new_wp,translator,target_lang)))
    
    supported_langs = ["su", "jw", "ms", "en", "fr", "it"]
    
    if target_lang not in supported_langs:
        raise ValueError('Language Unavailable')
    
    new_words = words.copy()
    
    if len(words_perturb) >= 1:
        for perturb_word in new_wp_trans.keys():
            new_words = [new_wp_trans[perturb_word] if word == perturb_word and word.isalpha() else word for word in new_words]

    # ic(words, words_perturb, new_words, new_wp_trans)
    return ' '.join(new_words), new_wp_trans
    
def codemix_perturbation(words, target_lang, words_perturb):
    """
    'su': 'sundanese'
    'jw': 'javanese'
    'ms': 'malay'
    'en': 'english'
    """
    translator = GoogleTranslator(source="id", target=target_lang)
    new_wp = []
    for wp in words_perturb:
        if wp[1].isalpha():
            new_wp.append(wp[1])
    
    if len(new_wp) == 1:
        new_wp_trans = dict(zip(new_wp, translator.translate(new_wp[0])))
    elif len(new_wp) == 0:
        return ' '.join(words), {}
        
    new_wp_trans = dict(zip(new_wp, translator.translate_batch(new_wp)))
    
    supported_langs = ["su", "jw", "ms", "en", "fr", "it"]
    
    if target_lang not in supported_langs:
        raise ValueError('Language Unavailable')
    
    new_words = words.copy()
    
    if len(words_perturb) >= 1:
        for perturb_word in new_wp_trans.keys():
            new_words = [new_wp_trans[perturb_word] if word == perturb_word and word.isalpha() else word for word in new_words]

    return ' '.join(new_words), new_wp_trans

def synonym_replacement(words, words_perturb):    
    # new_words = words.copy()
    new_wp = []
    for wp in words_perturb:
        if wp[1].isalpha():
            new_wp.append(wp[1])
    
    if len(new_wp) == 1:
        new_wp_trans = dict(zip(new_wp, get_synonyms(new_wp[0])[0]))
    elif len(new_wp) == 0:
        return ' '.join(words), {}
    
    wp_trans=[]
    for wp in new_wp:
        wp_trans.append(get_synonyms(wp)[0])
    
    new_wp_trans = dict(zip(new_wp, wp_trans))
    
    new_words = words.copy()
    
    if len(words_perturb) >= 1:
        for perturb_word in new_wp_trans.keys():
            new_words = [new_wp_trans[perturb_word] if word == perturb_word and word.isalpha() else word for word in new_words]
    
    return ' '.join(new_words), new_wp_trans

# fungsi untuk mencari kandidat lain ketika sebuah kandidat perturbasi kurang dari sim_score_threshold
def swap_minimum_importance_words(words_perturb, top_importance_words):
    def get_minimums(word_tups):
        arr = []
        for wt in word_tups:
            if wt[2] == min(top_importance_words, key = lambda t: t[2])[2]:
                arr.append(wt)
        return arr
    
    # get list of words with minimum importance score
    minimum_import = get_minimums(top_importance_words)
    unlisted = list(set(words_perturb).symmetric_difference(set(top_importance_words)))
    
    ic(unlisted)
    
    len_wp = len(top_importance_words)
    len_ul = len(unlisted)
    
    res = []
    for i in range(len_wp):
        if top_importance_words[i] in minimum_import:
            temp_wp = list(copy.deepcopy(top_importance_words))
            if len(temp_wp) == 0:
                break
            
            swapped_wp = np.array([(temp_wp) for i in range(len_ul)])
            
            ic(swapped_wp)
            for j in range(len(swapped_wp)):
                temp_sm = np.vstack((swapped_wp[j], tuple(unlisted[j])))
                
                res.append(temp_sm.tolist())
            temp_wp.pop(i)
                
    return res

def trans_dict(wordlist, perturbation_technique, lang=None):
    translated_wordlist = {}
    # ic(wordlist)
    if perturbation_technique == "codemixing":
        translator = GoogleTranslator(source="id", target=lang)
        
        for word in wordlist:
            if word.isalpha():
                translated_wordlist[word] = translator.translate(word)
            else:
                translated_wordlist[word] = word
    else:
        for word in wordlist:
            translated_wordlist[word] = get_synonyms(word)[0]
            
    return translated_wordlist