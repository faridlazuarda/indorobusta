import os, sys
import gc

gc.collect()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  

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
    perturb_lang="id",
    seed=26092020
):
    set_seed(seed)
    use = USE()

    tokenizer, config, model = init_model(model_target, downstream_task)
    w2i, i2w = load_word_index(downstream_task)
    
    train_dataset, train_loader, train_path = load_dataset_loader(downstream_task, 'train', tokenizer)
    valid_dataset, valid_loader, valid_path = load_dataset_loader(downstream_task, 'valid', tokenizer)
    test_dataset, test_loader, test_path = load_dataset_loader(downstream_task, 'test', tokenizer)

    finetuned_model = fine_tuning_model(model, i2w, train_loader, valid_loader, finetune_epoch)

    exp_dataset = valid_dataset.load_dataset(valid_path).head(num_sample)

    text,label = None,None
    if downstream_task == 'sentiment':
        text = 'text'
        label = 'sentiment'
        exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'orig_label', 'orig_prob', 'perturbed_label', 'perturbed_prob', 'running_time(s)']] = exp_dataset.swifter.apply(
            lambda row: attack(
                row.text,
                row.sentiment,
                finetuned_model,
                tokenizer, 0.2,
                "codemixing",
                perturb_lang,
                use), axis=1, result_type='expand'
        )
    elif downstream_task == 'emotion':
        text = 'tweet'
        label = 'label'
        exp_dataset[['perturbed_text', 'perturbed_semantic_sim', 'orig_label', 'orig_prob', 'perturbed_label', 'perturbed_prob', 'running_time(s)']] = exp_dataset.swifter.apply(
            lambda row: attack(
                row.tweet,
                row.label,
                finetuned_model,
                tokenizer, 0.2,
                "codemixing",
                perturb_lang,
                use), axis=1, result_type='expand'
        )

    before_attack = accuracy_score(exp_dataset[label], exp_dataset['orig_label'])
    after_attack = accuracy_score(exp_dataset[label], exp_dataset['perturbed_label'])

    exp_dataset.loc[exp_dataset.index[0], 'before_attack_acc'] = before_attack
    exp_dataset.loc[exp_dataset.index[0], 'after_attack_acc'] = after_attack
    exp_dataset.to_csv(os.getcwd() + r'/result/'+exp_name+".csv", index=False)   
    


if __name__ == "__main__":
    args = get_args()
    
    main(
        args.model_target,
        args.downstream_task,
        args.attack_strategy,
        args.finetune_epoch,
        args.num_sample,
        args.exp_name,
        args.perturbation_technique,
        args.perturb_ratio,
        args.perturb_lang,
    )