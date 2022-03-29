# random_seed=24032022
random_seed=26092020
# random_seed=42
cuda_id=4


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 
















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 














CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 



















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 

















############################ sentiment #####################################
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 
















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 














CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 



















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 






########################IndoBERT-Large###############################

