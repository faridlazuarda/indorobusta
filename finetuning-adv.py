import os, sys
import gc

gc.collect()

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"

from sklearn.metrics import accuracy_score
import swifter

from tqdm.notebook import tqdm
tqdm.pandas()

from utils.utils_init_dataset import set_seed
from utils.utils_semantic_use import USE
from utils.utils_data_utils import DocumentSentimentDataLoader, EmotionDetectionDataLoader
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, init_model, logit_prob, load_word_index
from utils.get_args import get_args

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

from attack.adv_attack import attack
import os, sys





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
        tweet, label = self.data.loc[index,'tweet'], self.data.loc[index,'label']        
        subwords = self.tokenizer.encode(tweet, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(label), tweet
    
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
        text, sentiment = data['text'], data['sentiment']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['text']
    
    def __len__(self):
        return len(self.data)    
    


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_target", required=True, type=str, default="IndoBERT", help="Choose between IndoBERT | XLM-R | mBERT")
    parser.add_argument("--downstream_task", required=True, type=str, default="sentiment", help="Choose between sentiment or emotion")
    parser.add_argument("--exp_name", required=True, type=str, help="choose experiment name")
    parser.add_argument("--seed", type=int, default=26092020, help="Seed")

    return parser.parse_args()



def load_dataset_loader(dataset_id, ds_type, tokenizer, path):
    dataset_path = None
    dataset = None
    loader = None
    if(dataset_id == 'sentiment'):
        if(ds_type == "train"):
            # dataset_path = './dataset/smsa-document-sentiment/train_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            # dataset_path = './dataset/smsa-document-sentiment/valid_preprocess.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/smsa-document-sentiment/test_preprocess_masked_label.tsv'
            dataset = DocumentSentimentDataset(dataset_path, tokenizer, lowercase=True)
            loader = DocumentSentimentDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    elif(dataset_id == 'emotion'):
        if(ds_type == "train"):
            # dataset_path = './dataset/emot-emotion-twitter/train_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=True)  
        elif(ds_type == "valid"):
            # dataset_path = './dataset/emot-emotion-twitter/valid_preprocess.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)
        elif(ds_type == "test"):
            dataset_path = './dataset/emot-emotion-twitter/test_preprocess_masked_label.csv'
            dataset = EmotionDetectionDataset(dataset_path, tokenizer, lowercase=True)
            loader = EmotionDetectionDataLoader(dataset=dataset, max_seq_len=512, batch_size=32, num_workers=80, shuffle=False)

    return dataset, loader, dataset_path



def main(
        model_target,
        downstream_task,
        attack_strategy,
        finetune_epoch,
        num_sample,
        exp_name,
        perturbation_technique,
        perturb_ratio,
        dataset,
        perturb_lang="id",
        seed=26092020
    ):
    
    set_seed(seed)
    use = USE()

    #     baca hasil perturb -> finetuning pake perturbed training -> predict data valid
    tokenizer, config, model = init_model(model_target, downstream_task, seed)
    w2i, i2w = load_word_index(downstream_task)
    
    
    trainpath = exp_name = model_target+"-"+downstream_task+"-"+attack_strategy+"-"+perturb_lang+"-adv"+"-"+str(perturb_ratio)+"-"+"train"
    
    validpath = exp_name = model_target+"-"+downstream_task+"-"+attack_strategy+"-"+perturb_lang+"-adv"+"-"+str(perturb_ratio)+"-"+"valid"

    
    train_dataset, train_loader, train_path = load_dataset_loader(downstream_task, 'train', tokenizer, path)
    valid_dataset, valid_loader, valid_path = load_dataset_loader(downstream_task, 'valid', tokenizer, path)
    test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)
    
    
    

    finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, finetune_epoch)
    
    # finetuned_model.save_pretrained(os.getcwd() + r"/models/seed"+str(seed)+ "/"+str(model_target)+"-"+str(downstream_task))
    
#     if model_target == "IndoBERT" or model_target == "mBERT" or model_target == "IndoBERT-Large":
#         finetuned_model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(model_target)+"-"+str(downstream_task))
#     else:
#         finetuned_model = XLMRobertaForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(model_target)+"-"+str(downstream_task))
    
#     if dataset == "valid":
#         exp_dataset = valid_dataset.load_dataset(valid_path).iloc[388:393]
#     elif dataset == "train":
#         exp_dataset = train_dataset.load_dataset(train_path)
#     # exp_dataset = dd.from_pandas(exp_dataset, npartitions=10)
#     text,label = None,None

#     # print(perturbation_technique)
#     if downstream_task == 'sentiment':
#         text = 'text'
#         label = 'sentiment'
#         exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'pred_label', 'pred_proba', 'perturbed_label', 'perturbed_proba', 'translated_word(s)', 'running_time(s)']] = exp_dataset.progress_apply(
#             lambda row: attack(
#                 text_ls = row.text,
#                 true_label = row.sentiment,
#                 predictor = finetuned_model,
#                 tokenizer = tokenizer, 
#                 att_ratio = perturb_ratio,
#                 perturbation_technique = perturbation_technique,
#                 lang_codemix = perturb_lang,
#                 sim_predictor = use), axis=1, result_type='expand'
#         )
#     elif downstream_task == 'emotion':
#         text = 'tweet'
#         label = 'label'
#         exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'pred_label', 'pred_proba', 'perturbed_label', 'perturbed_proba', 'translated_word(s)', 'running_time(s)']] = exp_dataset.progress_apply(
#             lambda row: attack(
#                 text_ls = row.tweet,
#                 true_label = row.label,
#                 predictor = finetuned_model,
#                 tokenizer = tokenizer, 
#                 att_ratio = perturb_ratio,
#                 perturbation_technique = perturbation_technique,
#                 lang_codemix = perturb_lang,
#                 sim_predictor = use), axis=1, result_type='expand'
#         )
        
if __name__ == "__main__":   
    args = get_args()
    
    main(
        args.model_target,
        args.downstream_task,
        args.exp_name,
        args.seed
    )