import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import copy
from deep_translator import GoogleTranslator
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw')
stop_words_set = []
for w in stopwords.words('indonesian'):
    stop_words_set.append(w)

    


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

def codemix_perturbation(words, target_lang, words_perturb):
    """
    'su': 'sundanese'
    'jw': 'javanese'
    'ms': 'malay'
    'en': 'english'
    """
    
    translator = GoogleTranslator(source="id", target=target_lang)
    
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
    new_words = words.copy()
       
    if len(words_perturb) >= 1:
        for perturb_word in words_perturb:
            new_words = [get_synonyms(word)[0] if word == perturb_word[1] and word.isalpha() else word for word in new_words]

    sentence = ' '.join(new_words)
    
    return sentence

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