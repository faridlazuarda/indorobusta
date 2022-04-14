random_seed=26092020
cuda_id=1



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.6-valid --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset valid --seed $random_seed 


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-jw-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang jw --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang su --seed $random_seed 


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.8-valid --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset valid --seed $random_seed 


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-jw-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 
