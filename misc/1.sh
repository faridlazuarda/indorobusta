random_seed=26092020
cuda_id=1



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main-dummy.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.2-valid --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset valid --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.4-valid --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset valid --seed $random_seed 









# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang su --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 



