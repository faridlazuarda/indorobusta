import os, sys
import gc

gc.collect()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import torch
# torch.cuda.set_device(device)

from sklearn.metrics import accuracy_score
import swifter

from tqdm.notebook import tqdm
tqdm.pandas()

from utils.utils_init_dataset import set_seed, load_dataset_loader
from utils.utils_semantic_use import USE
from utils.utils_data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader, EmotionDetectionDataset, EmotionDetectionDataLoader
from utils.utils_metrics import document_sentiment_metrics_fn
from utils.utils_init_model import text_logit, fine_tuning_model, eval_model, init_model, logit_prob, load_word_index
from utils.get_args import get_args

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification

from attack.adv_attack import attack
import os, sys


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

    tokenizer, config, model = init_model(model_target, downstream_task, seed)
    w2i, i2w = load_word_index(downstream_task)
    
    train_dataset, train_loader, train_path = load_dataset_loader(downstream_task, 'train', tokenizer)
    valid_dataset, valid_loader, valid_path = load_dataset_loader(downstream_task, 'valid', tokenizer)
    test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)

    finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, finetune_epoch)
    finetuned_model.save_pretrained(os.getcwd() + r"/models/seed"+str(seed)+ "/"+str(model_target)+"-"+str(downstream_task))
    
    if model_target == "IndoBERT" or model_target == "mBERT" or model_target == "IndoBERT-Large":
        finetuned_model = BertForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(model_target)+"-"+str(downstream_task))
    else:
        finetuned_model = XLMRobertaForSequenceClassification.from_pretrained(os.getcwd() + r"/models/seed"+str(seed) + "/"+str(model_target)+"-"+str(downstream_task))
    
    if dataset == "valid":
        exp_dataset = valid_dataset.load_dataset(valid_path).iloc[388:393]
    elif dataset == "train":
        exp_dataset = train_dataset.load_dataset(train_path)
    # exp_dataset = dd.from_pandas(exp_dataset, npartitions=10)
    text,label = None,None

    # print(perturbation_technique)
    if downstream_task == 'sentiment':
        text = 'text'
        label = 'sentiment'
        exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'pred_label', 'pred_proba', 'perturbed_label', 'perturbed_proba', 'translated_word(s)', 'running_time(s)']] = exp_dataset.progress_apply(
            lambda row: attack(
                text_ls = row.text,
                true_label = row.sentiment,
                predictor = finetuned_model,
                tokenizer = tokenizer, 
                att_ratio = perturb_ratio,
                perturbation_technique = perturbation_technique,
                lang_codemix = perturb_lang,
                sim_predictor = use), axis=1, result_type='expand'
        )
    elif downstream_task == 'emotion':
        text = 'tweet'
        label = 'label'
        exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'pred_label', 'pred_proba', 'perturbed_label', 'perturbed_proba', 'translated_word(s)', 'running_time(s)']] = exp_dataset.progress_apply(
            lambda row: attack(
                text_ls = row.tweet,
                true_label = row.label,
                predictor = finetuned_model,
                tokenizer = tokenizer, 
                att_ratio = perturb_ratio,
                perturbation_technique = perturbation_technique,
                lang_codemix = perturb_lang,
                sim_predictor = use), axis=1, result_type='expand'
        )
        
if __name__ == "__main__":   
    
    now_seed = 26092020
    now_seed_2 = 24032022
    now_seed_3 = 42
    
    # main(
    #     model_target="IndoBERT-Large",
    #     downstream_task="sentiment",
    #     attack_strategy="adversarial",
    #     finetune_epoch=5,
    #     num_sample=5,
    #     exp_name="coba",
    #     perturbation_technique="codemixing",
    #     perturb_ratio=0.7,
    #     dataset="valid",
    #     perturb_lang="en",
    #     seed=now_seed
    # )
    
    main(
        model_target="IndoBERT-Large",
        downstream_task="emotion",
        attack_strategy="adversarial",
        finetune_epoch=10,
        num_sample=5,
        exp_name="coba",
        perturbation_technique="codemixing",
        perturb_ratio=0.7,
        dataset="valid",
        perturb_lang="en",
        seed=now_seed
    )
    
    main(
        model_target="IndoBERT-Large",
        downstream_task="sentiment",
        attack_strategy="adversarial",
        finetune_epoch=5,
        num_sample=5,
        exp_name="coba",
        perturbation_technique="codemixing",
        perturb_ratio=0.7,
        dataset="valid",
        perturb_lang="en",
        seed=now_seed_2
    )
    
    main(
        model_target="IndoBERT-Large",
        downstream_task="emotion",
        attack_strategy="adversarial",
        finetune_epoch=10,
        num_sample=5,
        exp_name="coba",
        perturbation_technique="codemixing",
        perturb_ratio=0.7,
        dataset="valid",
        perturb_lang="en",
        seed=now_seed_2
    )
    
    main(
        model_target="IndoBERT-Large",
        downstream_task="sentiment",
        attack_strategy="adversarial",
        finetune_epoch=5,
        num_sample=5,
        exp_name="coba",
        perturbation_technique="codemixing",
        perturb_ratio=0.7,
        dataset="valid",
        perturb_lang="en",
        seed=now_seed_3
    )
    
    main(
        model_target="IndoBERT-Large",
        downstream_task="emotion",
        attack_strategy="adversarial",
        finetune_epoch=10,
        num_sample=5,
        exp_name="coba",
        perturbation_technique="codemixing",
        perturb_ratio=0.7,
        dataset="valid",
        perturb_lang="en",
        seed=now_seed_3
    )
    
    
    # main(
    #     model_target="IndoBERT",
    #     downstream_task="sentiment",
    #     attack_strategy="adversarial",
    #     finetune_epoch=5,
    #     num_sample=5,
    #     exp_name="coba",
    #     perturbation_technique="codemixing",
    #     perturb_ratio=0.7,
    #     dataset="valid",
    #     perturb_lang="en",
    #     seed=now_seed
    # )
    
#     main(
#         model_target="XLM-R",
#         downstream_task="sentiment",
#         attack_strategy="adversarial",
#         finetune_epoch=5,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed
#     )
    
#     main(
#         model_target="mBERT",
#         downstream_task="sentiment",
#         attack_strategy="adversarial",
#         finetune_epoch=5,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed
#     )
    
#     main(
#         model_target="IndoBERT",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed
#     )
    
#     main(
#         model_target="XLM-R",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed
#     )
    
#     main(
#         model_target="mBERT",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed
#     )
    
    
    
    
    
    
#     main(
#         model_target="IndoBERT",
#         downstream_task="sentiment",
#         attack_strategy="adversarial",
#         finetune_epoch=5,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )
    
#     main(
#         model_target="XLM-R",
#         downstream_task="sentiment",
#         attack_strategy="adversarial",
#         finetune_epoch=5,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )
    
#     main(
#         model_target="mBERT",
#         downstream_task="sentiment",
#         attack_strategy="adversarial",
#         finetune_epoch=5,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )
    
#     main(
#         model_target="IndoBERT",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )
    
#     main(
#         model_target="XLM-R",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )
    
#     main(
#         model_target="mBERT",
#         downstream_task="emotion",
#         attack_strategy="adversarial",
#         finetune_epoch=10,
#         num_sample=5,
#         exp_name="coba",
#         perturbation_technique="codemixing",
#         perturb_ratio=0.7,
#         dataset="valid",
#         perturb_lang="en",
#         seed=now_seed_2
#     )