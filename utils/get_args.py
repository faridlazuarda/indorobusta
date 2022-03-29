import argparse


model_target="IndoBERT", # IndoBERT, XLM-R, mBERT
downstream_task="emotion", # sentiment, emotion
attack_strategy="synonym_replacement", # codemixing, synonym replacement
perturbation_technique="adversarial", # adversarial, random
perturb_ratio=0.2, # 0.2, 0.4, 0.6, 0.8
finetune_epoch=5,
num_sample=5,
result_file="test-indobert-emotion-synonym_replacement-adversarial-0.2",
seed=26092020


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_target", required=True, type=str, default="IndoBERT", help="Choose between IndoBERT | XLM-R | mBERT")
    parser.add_argument("--downstream_task", required=True, type=str, default="sentiment", help="Choose between sentiment or emotion")
    parser.add_argument("--attack_strategy", required=True, type=str, default="adversarial", help="Choose between adversarial or random")
    parser.add_argument("--finetune_epoch", required=True, type=int, default=5, help="Number of epochs when fine-tune")
    parser.add_argument("--num_sample",type=int, default=10, help="Number of sample to be perturbed from valid_dataset on each downstream task respectfully [SmSA 1000 | EmoT 400]")
    parser.add_argument("--exp_name", required=True, type=str, default="exp", help="file to write the experiment result")
    parser.add_argument("--perturbation_technique", required=True, type=str, default="codemixing", help="Choose between codemixing and synonym replacement")
    parser.add_argument("--perturb_ratio", required=True, type=float, default=0.2 , help="Ratio of words change in a single attack [0.2 | 0.4 | 0.6 | 0.8]")    
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--perturb_lang", type=str, default="id" , help="Perturbation language only when using codemixing, choose between su | jw | ms | en")
    parser.add_argument("--seed", type=int, default=26092020, help="Seed")

    return parser.parse_args()