import os, sys
import gc

gc.collect()

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from sklearn.metrics import accuracy_score
import swifter

from tqdm.notebook import tqdm
tqdm.pandas()

from utils.utils_init_dataset import set_seed
from utils.utils_semantic_use import USE
from utils.utils_data_utils import DocumentSentimentDataLoader, EmotionDetectionDataLoader
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, logit_prob, load_word_index, init_model
from utils.get_args import get_args

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

from torch.utils.data import Dataset, DataLoader

from attack.adv_attack import attack
import os, sys
from icecream import ic
import pandas as pd
import numpy as np





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

    parser.add_argument("--model_target", required=True, type=str, default="IndoBERT", help="Choose between IndoBERT | XLM-R | mBERT")
    parser.add_argument("--downstream_task", required=True, type=str, default="sentiment", help="Choose between sentiment or emotion")
    parser.add_argument("--exp_name", required=True, type=str, help="choose experiment name")
    parser.add_argument("--seed", type=int, default=26092020, help="Seed")

    return parser.parse_args()



def load_dataset_loader(dataset_id, ds_type, tokenizer, path=None):
    print(dataset_id, ds_type, tokenizer, path)
    
    dataset_path = None
    dataset = None
    loader = None
    if(dataset_id == 'sentiment'):
        if(ds_type == "train"):
            # dataset_path = './dataset/smsa-document-sentiment/train_preprocess.tsv'
            ic(path)
            dataset = DocumentSentimentDataset(path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            # dataset_path = './dataset/smsa-document-sentiment/valid_preprocess.tsv'
            ic(path)
            dataset = DocumentSentimentDataset(path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/smsa-document-sentiment/test_preprocess_masked_label.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    elif(dataset_id == 'emotion'):
        if(ds_type == "train"):
            # dataset_path = './dataset/emot-emotion-twitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            # dataset_path = './dataset/emot-emotion-twitter/valid_preprocess.csv'
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
    
    ic(trainpath)
    ic(validpath)
    
    train_dataset, train_loader, _ = load_dataset_loader(downstream_task, 'train', tokenizer, trainpath)
    valid_dataset, valid_loader, _ = load_dataset_loader(downstream_task, 'valid', tokenizer, validpath)
    # test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)
    
    
    
    if "sentiment" in trainpath: 
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, epochs=5)
    else: 
        finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, epochs=10)
    
if __name__ == "__main__":   
#     args = get_args()
    
#     main(
#         args.model_target,
#         args.downstream_task,
#         args.exp_name,
#         args.seed
#     )

    exp_name = "xlmr-sentiment-codemixing-fr-adv-0.8"
    # exp_name = "mbert-emotion-sr-adv-0.8-train"
    names = exp_name.split("-")
    
    print(exp_name.split("-"))
    
    model_tgt = names[0]
    downstream_task = names[1]
    seed = 26092020
    
    model_map = {
        'indobert': 'IndoBERT',
        'indobertlarge': 'IndoBERT-Large',
        'xlmr': 'XLM-R',
        'xlmrlarge': 'XLM-R-Large',
        'mbert': 'mBERT'
    }
    
    model_target = model_map[model_tgt]
    
    main(
        model_target,
        downstream_task,
        exp_name,
        seed
    )