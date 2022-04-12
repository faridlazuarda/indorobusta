import numpy as np
import random

import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

from utils.utils_forward_fn import forward_sequence_classification
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader, EmotionDetectionDataset, EmotionDetectionDataLoader
from utils.utils_semantic_use import USE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def load_dataset_loader(dataset_id, ds_type, tokenizer):
    dataset_path = None
    dataset = None
    loader = None
    if(dataset_id == 'sentiment'):
        if(ds_type == "train"):
            dataset_path = './dataset/smsa-document-sentiment/train_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            dataset_path = './dataset/smsa-document-sentiment/valid_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/smsa-document-sentiment/test_preprocess_masked_label.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    elif(dataset_id == 'emotion'):
        if(ds_type == "train"):
            dataset_path = './dataset/emot-emotion-twitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            dataset_path = './dataset/emot-emotion-twitter/valid_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/emot-emotion-twitter/test_preprocess_masked_label.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    return dataset, loader, dataset_path