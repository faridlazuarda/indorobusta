random_seed=26092020
cuda_id=2



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.2-valid --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset valid --seed $random_seed








CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.4-valid --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset valid --seed $random_seed








CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.6-valid --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset valid --seed $random_seed








CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.8-valid --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset valid --seed $random_seed













###############################SENTIMENT#######################################
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.2-valid  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.2-valid --perturbation_technique codemixing --perturb_ratio 0.2 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.2-valid --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset valid --seed $random_seed















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.4-valid  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.4-valid --perturbation_technique codemixing --perturb_ratio 0.4 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.4-valid --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset valid --seed $random_seed


















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.6-valid  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.6-valid --perturbation_technique codemixing --perturb_ratio 0.6 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.6-valid --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset valid --seed $random_seed




















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.8-valid  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.8-valid --perturbation_technique codemixing --perturb_ratio 0.8 --dataset valid --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.8-valid --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset valid --seed $random_seed

