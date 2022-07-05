random_seed=26092020
cuda_id=5

# CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning-adv-dfcomb.py --seed $random_seed
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning-adv-perturb_data_only.py --seed $random_seed