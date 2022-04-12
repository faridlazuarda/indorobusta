import torch
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(device)

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
tqdm.pandas()

from .utils_metrics import document_sentiment_metrics_fn
from .utils_forward_fn import forward_sequence_classification
from .utils_data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader, EmotionDetectionDataset, EmotionDetectionDataLoader


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def logit_prob(text_ls, predictor, tokenizer):
    original_text = text_ls
    # print(text_ls)
    subwords = tokenizer.encode(text_ls)
    subwords = torch.LongTensor(subwords).view(1, -1).to(predictor.device)

    logits = predictor(subwords)[0]
    orig_label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    
    orig_probs = F.softmax(logits, dim=-1).squeeze()
    orig_prob = F.softmax(logits, dim=-1).squeeze()[orig_label].detach().cpu().numpy()
    
    return orig_label, orig_probs, orig_prob

    
def load_word_index(downstream_task):
    w2i, i2w = None, None
    if downstream_task == 'sentiment':
        w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL
        return w2i, i2w
    elif downstream_task == 'emotion':
        w2i, i2w = EmotionDetectionDataset.LABEL2INDEX, EmotionDetectionDataset.INDEX2LABEL
        return w2i, i2w
    else:
        return None

def init_model(id_model, downstream_task, seed):
    if id_model == "IndoBERT":
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2', local_files_only=True)
        config = BertConfig.from_pretrained('indobenchmark/indobert-base-p2')
        if downstream_task == "sentiment":
            config.num_labels = DocumentSentimentDataset.NUM_LABELS
        elif downstream_task == "emotion":
            config.num_labels = EmotionDetectionDataset.NUM_LABELS
        else:
            return "Task does not match"
        
        # model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p2', config=config)
        model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
    elif id_model == "XLM-R":
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', local_files_only=True)
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
        if downstream_task == "sentiment":
            config.num_labels = DocumentSentimentDataset.NUM_LABELS
        elif downstream_task == "emotion":
            config.num_labels = EmotionDetectionDataset.NUM_LABELS
        else:
            return "Task does not match"
        
        # model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', config=config)
        model = XLMRobertaForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
    elif id_model == "mBERT":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', local_files_only=True)
        config = BertConfig.from_pretrained("bert-base-multilingual-uncased")
        if downstream_task == "sentiment":
            config.num_labels = DocumentSentimentDataset.NUM_LABELS
        elif downstream_task == "emotion":
            config.num_labels = EmotionDetectionDataset.NUM_LABELS
        else:
            return "Task does not match"
        
        # model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', config=config)
        model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
        
    elif id_model == "IndoBERT-Large":
        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-large-p2", local_files_only=True)
        config = BertConfig.from_pretrained("indobenchmark/indobert-large-p2")
        if downstream_task == "sentiment":
            config.num_labels = DocumentSentimentDataset.NUM_LABELS
        elif downstream_task == "emotion":
            config.num_labels = EmotionDetectionDataset.NUM_LABELS
        else:
            return "Task does not match"
        
        # model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-large-p2", config=config)
        model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(id_model)+"-"+str(downstream_task))
    
    return tokenizer, config, model

def text_logit(text, model, tokenizer, i2w):
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
    
    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

    # print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')
    return i2w[label], F.softmax(logits, dim=-1).squeeze()[label] * 100

def fine_tuning_model(base_model, i2w, train_loader, valid_loader, epochs=5):
    optimizer = optim.Adam(base_model.parameters(), lr=3e-6)
    base_model = base_model.cuda()
    
    # Train
    n_epochs = epochs
    for epoch in range(n_epochs):
        base_model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward base_model
            loss, batch_hyp, batch_label = forward_sequence_classification(base_model, batch_data[:-1], i2w=i2w, device=device)

            # Update base_model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                total_train_loss/(i+1), get_lr(optimizer)))

        # Calculate train metric
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
            total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        # Evaluate on validation
        base_model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]        
            loss, batch_hyp, batch_label = forward_sequence_classification(base_model, batch_data[:-1], i2w=i2w, device=device)

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = document_sentiment_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))

        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
            total_loss/(i+1), metrics_to_string(metrics)))
    return base_model

def eval_model(model, test_loader, i2w):
    # Evaluate on test
    model.eval()
    torch.set_grad_enabled(False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device=device)
        list_hyp += batch_hyp

    # Save prediction
    df = pd.DataFrame({'label':list_hyp}).reset_index()
    df.to_csv('pred.txt', index=False)

    print(df)