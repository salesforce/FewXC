
# FT

CUDA_VISIBLE_DEVICES=0  python xlmr_massive_finetuning.py --max_seq_length 64  --training_strategy all --learning_rate 1e-5

# PT

CUDA_VISIBLE_DEVICES=0 python xlmr_massive_finetuning.py  --prefix  --learning_rate 0.01 --epochs 1000 --training_strategy 5 --max_seq_length 64

# APT
CUDA_VISIBLE_DEVICES=0  python -u xlmr_massive_finetuning.py  --prefix  --pre_seq_len 16  --learning_rate 0.002  --training_strategy all   --epochs 1000 --previous_checkpoint dstc8-schema-guided-dialogue-withmlm-t20-all-extralayer-e10-xlm-roberta-large-16-0.001-1/checkpoint-33110/ 


# FT evaluation
python -u xlmr_sgd_finetuning_eval.py   --eval_checkpoint  MODEL_checkpoints  --test_data_dir dstc8-schema-guided-dialogue/dev_multi-lingual

# PT and APT evaluation 
python -u xlmr_massive_finetuning_eval.py --language all  --prefix   --eval_checkpoint MODEL_checkpoints --pre_seq_len 32

