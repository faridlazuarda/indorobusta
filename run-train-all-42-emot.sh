random_seed=42
cuda_id=7


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train  --perturb_lang en --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-it-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed














CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train  --perturb_lang en --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-it-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed














CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train  --perturb_lang en --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-it-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed 



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train  --perturb_lang en --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed  

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-codemixing-it-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed 


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-emotion-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed



CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed





















###############################SENTIMENT#######################################

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-sentiment-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train  --perturb_lang en --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed 



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-sentiment-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-sentiment-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed














# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-sentiment-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train  --perturb_lang en --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed 



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-sentiment-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-sentiment-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed














# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-sentiment-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train  --perturb_lang en --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed 



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-sentiment-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-sentiment-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


















# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-sentiment-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-sentiment-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train  --perturb_lang en --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed  

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-codemixing-it-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 5 --exp_name indobertlarge-sentiment-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed 


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-sentiment-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-sentiment-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmrlarge-sentiment-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R-Large --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmrlarge-sentiment-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed



# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-it-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang it --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-sentiment-codemixing-fr-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang fr --seed $random_seed 

# CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task sentiment --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-sentiment-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed