python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy codemixing --finetune_epoch 5 --num_sample 10 --exp_name indobert-sentiment-codemixing-en-adv-0.6-sample10  --perturbation_technique adversarial --perturb_ratio 0.4 --perturb_lang en 

python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy codemixing --finetune_epoch 5 --num_sample 10 --exp_name indobert-sentiment-codemixing-jw-adv-0.6-sample10 --perturbation_technique adversarial --perturb_ratio 0.4 --perturb_lang jw 

python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy codemixing --finetune_epoch 5 --num_sample 10 --exp_name indobert-sentiment-codemixing-su-adv-0.6-sample10 --perturbation_technique adversarial --perturb_ratio 0.4 --perturb_lang su

python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy codemixing --finetune_epoch 5 --num_sample 10 --exp_name indobert-sentiment-codemixing-ms-adv-0.6-sample10 --perturbation_technique adversarial --perturb_ratio 0.4 --perturb_lang ms

python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy synonym_replacement  --finetune_epoch 5 --num_sample 10 --exp_name indobert-sentiment-sr-adv-0.6-sample10 --perturbation_technique adversarial --perturb_ratio 0.4