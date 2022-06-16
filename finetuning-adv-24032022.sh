random_seed=24032022
cuda_id=4

CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning-adv-dfcomb.py --seed $random_seed
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning-adv-perturb_data_only.py --seed $random_seed