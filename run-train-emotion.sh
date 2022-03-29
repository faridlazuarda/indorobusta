random_seed=26092020
# random_seed=24032022
# random_seed=42
cuda_id=1


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --dataset train --perturb_lang en --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --dataset train --perturb_lang en  --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.2-train  --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.2-train --perturbation_technique codemixing --perturb_ratio 0.2 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.2-train --perturbation_technique synonym_replacement --perturb_ratio 0.2 --dataset train --seed $random_seed
















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.4-train  --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.4-train --perturbation_technique codemixing --perturb_ratio 0.4 --dataset train --perturb_lang ms --seed $random_seed
# 
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.4-train --perturbation_technique synonym_replacement --perturb_ratio 0.4 --dataset train --seed $random_seed














CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.6-train  --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.6-train --perturbation_technique codemixing --perturb_ratio 0.6 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.6-train --perturbation_technique synonym_replacement --perturb_ratio 0.6 --dataset train --seed $random_seed


















CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name indobert-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target IndoBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name indobert-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name xlmr-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target XLM-R --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name xlmr-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed


CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-en-adv-0.8-train  --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang en --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-jw-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang jw --seed $random_seed 

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-su-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang su --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial --finetune_epoch 10 --exp_name mbert-emotion-codemixing-ms-adv-0.8-train --perturbation_technique codemixing --perturb_ratio 0.8 --dataset train --perturb_lang ms --seed $random_seed

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --model_target mBERT --downstream_task emotion --attack_strategy adversarial  --finetune_epoch 10 --exp_name mbert-emotion-sr-adv-0.8-train --perturbation_technique synonym_replacement --perturb_ratio 0.8 --dataset train --seed $random_seed