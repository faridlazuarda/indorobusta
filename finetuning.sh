# random_seed=42
cuda_id=3


CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task emotion --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task emotion --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task emotion --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task emotion --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task emotion --seed 1


CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task emotion --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task emotion --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task emotion --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task emotion --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task emotion --seed 2


CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task emotion --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task emotion --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task emotion --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task emotion --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task emotion --seed 3












CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task sentiment --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task sentiment --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task sentiment --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task sentiment --seed 1
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task sentiment --seed 1


CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task sentiment --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task sentiment --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task sentiment --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task sentiment --seed 2
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task sentiment --seed 2


CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT --downstream_task sentiment --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target IndoBERT-Large --downstream_task sentiment --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R --downstream_task sentiment --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target XLM-R-Large --downstream_task sentiment --seed 3
CUDA_VISIBLE_DEVICES=$cuda_id python3 finetuning.py --model_target mBERT --downstream_task sentiment --seed 3