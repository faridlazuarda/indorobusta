import argparse



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_target", required=True, help="")
    parser.add_argument("--downstream_task", required=True, help="")
    parser.add_argument("--attack_strategy", required=True, help="")
    parser.add_argument("--perturbation_technique", required=True, help="")
    parser.add_argument("--perturb_ratio", required=True, help="")
    parser.add_argument("--finetune_epoch", required=True, help="")
    parser.add_argument("--num_sample", required=True, help="")
    parser.add_argument("--result_file", required=True, help="")
    parser.add_argument("--seed")

    return parser.parse_args()

main(
        model_target="IndoBERT", # IndoBERT, XLM-R, mBERT
        downstream_task="sentiment", # sentiment, emotion
        attack_strategy="codemixing", # codemixing, synonym replacement
        perturbation_technique="adversarial", # adversarial, random
        perturb_ratio=0.6, # 0.2, 0.4, 0.6, 0.8
        finetune_epoch=5,
        num_sample=5,
        result_file="test-indobert-sentiment-codemixing-adversarial-0.6",
        seed=26092020
    )
    
    main(
        model_target="IndoBERT", # IndoBERT, XLM-R, mBERT
        downstream_task="emotion", # sentiment, emotion
        attack_strategy="codemixing", # codemixing, synonym replacement
        perturbation_technique="adversarial", # adversarial, random
        perturb_ratio=0.6, # 0.2, 0.4, 0.6, 0.8
        finetune_epoch=5,
        num_sample=5,
        result_file="test-indobert-emotion-codemixing-adversarial-0.6",
        seed=26092020
    )
    
    main(
        model_target="IndoBERT", # IndoBERT, XLM-R, mBERT
        downstream_task="sentiment", # sentiment, emotion
        attack_strategy="synonym_replacement", # codemixing, synonym replacement
        perturbation_technique="adversarial", # adversarial, random
        perturb_ratio=0.6, # 0.2, 0.4, 0.6, 0.8
        finetune_epoch=5,
        num_sample=5,
        result_file="test-indobert-sentiment-synonym_replacement-adversarial-0.6",
        seed=26092020
    )
    
    main(
        model_target="IndoBERT", # IndoBERT, XLM-R, mBERT
        downstream_task="emotion", # sentiment, emotion
        attack_strategy="synonym_replacement", # codemixing, synonym replacement
        perturbation_technique="adversarial", # adversarial, random
        perturb_ratio=0.6, # 0.2, 0.4, 0.6, 0.8
        finetune_epoch=5,
        num_sample=5,
        result_file="test-indobert-emotion-synonym_replacement-adversarial-0.6",
        seed=26092020
    )
