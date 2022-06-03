import os, sys
import gc

gc.collect()

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from sklearn.metrics import accuracy_score
import swifter

from tqdm.notebook import tqdm
tqdm.pandas()

from utils.utils_init_dataset import set_seed
from utils.utils_semantic_use import USE
from utils.utils_data_utils import EmotionDetectionDataset as EmotionDetectionDatasetOrig
from utils.utils_data_utils import DocumentSentimentDataset as DocumentSentimentDatasetOrig
from utils.utils_data_utils import DocumentSentimentDataLoader, EmotionDetectionDataLoader

from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, logit_prob, load_word_index, init_model
from utils.get_args import get_args
from utils.utils_forward_fn import forward_sequence_classification

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

from torch.utils.data import Dataset, DataLoader

from attack.adv_attack import attack
import os, sys
from icecream import ic
import pandas as pd
import numpy as np
import argparse


#####
# Emotion Twitter
#####
class EmotionDetectionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
    INDEX2LABEL = {0: 'sadness', 1: 'anger', 2: 'love', 3: 'fear', 4: 'happy'}
    NUM_LABELS = 5
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        # dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset[["perturbed_text", "label"]]

    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        
    def __getitem__(self, index):
        perturbed_text, label = self.data.loc[index,'perturbed_text'], self.data.loc[index,'label']        
        subwords = self.tokenizer.encode(perturbed_text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(label), perturbed_text
    
    def __len__(self):
        return len(self.data)


#####
# Document Sentiment Prosa
#####
class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
    INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}
    NUM_LABELS = 3
    
    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        # dataset['label'] = dataset['label'].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset[["perturbed_text", "sentiment"]]
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        perturbed_text, sentiment = data['perturbed_text'], data['sentiment']
        subwords = self.tokenizer.encode(perturbed_text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['perturbed_text']
    
    def __len__(self):
        return len(self.data)    
    


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_target", required=True, type=str, default="IndoBERT", help="Choose between IndoBERT | XLM-R | mBERT")
    # parser.add_argument("--downstream_task", required=True, type=str, default="sentiment", help="Choose between sentiment or emotion")
    # parser.add_argument("--exp_name", required=True, type=str, help="choose experiment name")
    parser.add_argument("--seed", type=int, default=26092020, help="Seed")

    return parser.parse_args()



def load_dataset_loader(dataset_id, ds_type, tokenizer, path=None):
    # print(dataset_id, ds_type, tokenizer, path)
    
    dataset_path = None
    dataset = None
    loader = None
    if(dataset_id == 'sentiment'):
        if(ds_type == "train"):
            if path is None:
                path = './dataset/smsa-document-sentiment/train_preprocess.tsv'
            # ic(path)
            dataset = DocumentSentimentDataset(path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            if path is None:
                path = './dataset/smsa-document-sentiment/valid_preprocess.tsv'
            # ic(path)
            dataset = DocumentSentimentDataset(path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/smsa-document-sentiment/test_preprocess_masked_label.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    elif(dataset_id == 'emotion'):
        if(ds_type == "train"):
            if path is None:
                path = './dataset/emot-emotion-twitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            if path is None:
                path = './dataset/emot-emotion-twitter/valid_preprocess.csv'
            dataset = EmotionDetectionDataset(path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/emot-emotion-twitter/test_preprocess_masked_label.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
    
    return dataset, loader, dataset_path



def main(        
        model_target,
        downstream_task,
        exp_name,
        seed
    ):
    
    set_seed(seed)
    # use = USE()

    #     baca hasil perturb -> finetuning pake perturbed training -> predict data valid
    tokenizer, config, model = init_model(model_target, downstream_task, seed)
    w2i, i2w = load_word_index(downstream_task)
    
    trainpath = os.getcwd() + r'/result/seed'+str(seed)+"/train/"+exp_name+"-train"+".csv"
    validpath = os.getcwd() + r'/result/seed'+str(seed)+"/valid/"+exp_name+"-valid"+".csv"
    
#     ic(trainpath)
#     ic(validpath)
    
    _, train_loader, _ = load_dataset_loader(downstream_task, 'train', tokenizer, trainpath)
    _, valid_loader, _ = load_dataset_loader(downstream_task, 'valid', tokenizer, validpath)
    # _t, valid_loader_orig, _ = load_dataset_loader(downstream_task, 'valid', tokenizer)
    # test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)
    
    
    
    if "sentiment" in trainpath: 
        # text = 'text'
        orig_col_label = 'sentiment'
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, epochs=5)
        LABEL2INDEX = {'positive': 0, 'neutral': 1, 'negative': 2}
        valid_path_orig = './dataset/smsa-document-sentiment/valid_preprocess.tsv'
        valid_dataset_orig = DocumentSentimentDatasetOrig(valid_path_orig, tokenizer, lowercase=True)
        valid_loader_orig = DocumentSentimentDataLoader(dataset=valid_dataset_orig, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
    else: 
        # text = 'tweet'
        orig_col_label = 'label'
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, epochs=10)
        LABEL2INDEX = {'sadness': 0, 'anger': 1, 'love': 2, 'fear': 3, 'happy': 4}
        valid_path_orig = './dataset/emot-emotion-twitter/valid_preprocess.csv'
        valid_dataset_orig = EmotionDetectionDatasetOrig(valid_path_orig, tokenizer, lowercase=True)
        valid_loader_orig = EmotionDetectionDataLoader(dataset=valid_dataset_orig, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
    
    valid_df = pd.read_csv(os.getcwd() + r'/result/seed'+str(seed)+"/valid/"+exp_name+"-valid"+".csv")
    
    # prediksi adv_training vs original finetuned model on perturbed data
    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(pbar):
        # ic(batch_data)
        _, batch_hyp, _ = forward_sequence_classification(finetuned_model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp

    # Save prediction
    temp_df_perturb = pd.DataFrame({'label':list_hyp}).reset_index()
    temp_df_perturb['label'] = temp_df_perturb['label'].apply(lambda sen: LABEL2INDEX[sen])
    
    
    # prediksi adv_training vs original finetuned model on original data
    
    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(valid_loader_orig, leave=True, total=len(valid_loader_orig))
    for i, batch_data in enumerate(pbar):
        # ic(batch_data)
        _, batch_hyp, _ = forward_sequence_classification(finetuned_model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp

    # Save prediction
    temp_df_orig = pd.DataFrame({'label':list_hyp}).reset_index()
    temp_df_orig['label'] = temp_df_orig['label'].apply(lambda sen: LABEL2INDEX[sen])
    
    
    valid_df = pd.read_csv(os.getcwd() + r'/result/seed'+str(seed)+"/valid/"+exp_name+"-valid"+".csv")

    # valid_df["adv_pred"] = df["label"]
    valid_df.insert(loc=9, column='adv_pred', value=temp_df_perturb["label"].values)
    valid_df.insert(loc=10, column='adv_pred_on_orig', value=temp_df_orig["label"].values)
    
    adv_training = accuracy_score(valid_df[orig_col_label], valid_df['adv_pred'])
    delta_acc = valid_df.before_attack_acc.values[0] - valid_df.after_attack_acc.values[0]
    delta_adv = valid_df.before_attack_acc.values[0] - adv_training 
    
    acc_adv_training_on_orig = accuracy_score(valid_df[orig_col_label], valid_df['adv_pred_on_orig'])
    
    valid_df.loc[valid_df.index[0], 'adv_training'] = adv_training
    valid_df.loc[valid_df.index[0], 'delta_acc'] = delta_acc
    valid_df.loc[valid_df.index[0], 'delta_adv'] = delta_adv
    valid_df.loc[valid_df.index[0], 'acc_adv_training_on_orig'] = acc_adv_training_on_orig

    valid_df.to_csv(os.getcwd() + r'/adversarial-training/result/seed'+str(seed)+"/"+str(exp_name)+".csv", index=False)

    # valid_df.to_csv("valid_test.csv", index=False)


def get_intersect(seed):
    path_train = str(os.getcwd()) + "/result/seed" + str(seed) + "/" + "train" + "/"
    path_valid = str(os.getcwd()) + "/result/seed" + str(seed) + "/" + "valid" + "/"

    dir_list_train = [f[:-10] for f in os.listdir(path_train)]
    dir_list_valid = [f[:-10] for f in os.listdir(path_valid)]
    
    return list(set(dir_list_train) & set(dir_list_valid))
  

if __name__ == "__main__":   
    args = get_args()
    
    model_map = {
        'indobert': 'IndoBERT',
        'indobertlarge': 'IndoBERT-Large',
        'xlmr': 'XLM-R',
        'xlmrlarge': 'XLM-R-Large',
        'mbert': 'mBERT'
    }
    
    intersect = get_intersect(args.seed)
    
    path_adv = os.getcwd() + r'/adversarial-training/result/seed'+str(args.seed)+"/"
    dir_list_adv = [f[:-4] for f in os.listdir(path_adv) if "ipynb" not in f]
    
    for exp_name in intersect:    
        if exp_name in intersect and exp_name not in dir_list_adv and 'ipynb' not in exp_name:
            print(exp_name)
            names = exp_name.split("-")
            model_tgt = names[0]
            downstream_task = names[1]
            model_target = model_map[model_tgt]
            print(exp_name.split("-"))
            main(
                model_target=model_target,
                downstream_task=downstream_task,
                exp_name=exp_name,
                seed=args.seed
            )

    
    
    
    
    
    

# def init_model(id_model, downstream_task, seed):
#     if id_model == "IndoBERT":
#         tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
#         config = BertConfig.from_pretrained('indobenchmark/indobert-base-p2')
#         if downstream_task == "sentiment":
#             config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         elif downstream_task == "emotion":
#             config.num_labels = EmotionDetectionDataset.NUM_LABELS
#         else:
#             return "Task does not match"
        
#         model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', config=config)
#         # model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
#     elif id_model == "XLM-R":
#         tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
#         config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
#         if downstream_task == "sentiment":
#             config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         elif downstream_task == "emotion":
#             config.num_labels = EmotionDetectionDataset.NUM_LABELS
#         else:
#             return "Task does not match"
        
#         model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', config=config)
#         # model = XLMRobertaForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
#     elif id_model == "XLM-R-Large":
#         tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
#         config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
#         if downstream_task == "sentiment":
#             config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         elif downstream_task == "emotion":
#             config.num_labels = EmotionDetectionDataset.NUM_LABELS
#         else:
#             return "Task does not match"

#         model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', config=config)
#         # model = XLMRobertaForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))

#     elif id_model == "mBERT":
#         tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
#         config = BertConfig.from_pretrained("bert-base-multilingual-uncased")
#         if downstream_task == "sentiment":
#             config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         elif downstream_task == "emotion":
#             config.num_labels = EmotionDetectionDataset.NUM_LABELS
#         else:
#             return "Task does not match"
        
#         model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', config=config)
#         # model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
#     elif id_model == "IndoBERT-Large":
#         tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-large-p2")
#         config = BertConfig.from_pretrained("indobenchmark/indobert-large-p2")
#         if downstream_task == "sentiment":
#             config.num_labels = DocumentSentimentDataset.NUM_LABELS
#         elif downstream_task == "emotion":
#             config.num_labels = EmotionDetectionDataset.NUM_LABELS
#         else:
#             return "Task does not match"
        
#         model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-large-p2", config=config)
#         # model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
    
#     return tokenizer, config, model