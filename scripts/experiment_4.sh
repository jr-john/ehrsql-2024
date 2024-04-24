#!/bin/bash

# Abstention (Multi-Task)
cd src/data_preparation
python schema_generation.py
python abstention_data_preparation.py --schema_prompt_path schema_prompt.txt --output_path abstention_train.pkl
python data_preparation.py --schema_prompt_path schema_prompt.txt --output_path prediction_train.pkl
python merge_data.py --prediction_data_path prediction_train.pkl --abstention_data_path abstention_train.pkl

cd ../finetuning
python finetuning.py --train_data_path ../data_preparation/train.pkl --num_train_epochs 1 --task EHRSQL-Multitask
last_checkpoint=$(ls -Art sqlcoder-EHRSQL-Multitask | tail -n 1)
python merge_model.py --model_path sqlcoder-EHRSQL-Multitask/$last_checkpoint --output_path sqlcoder-EHRSQL-Multitask/merged-model
python inference.py --model_path sqlcoder-EHRSQL-Multitask/merged-model --task prediction --schema_prompt_path ../data_preparation/schema_prompt.txt
python inference.py --model_path sqlcoder-EHRSQL-Multitask/merged-model --task abstention_prediction --schema_prompt_path ../data_preparation/schema_prompt.txt

# Error Filtering
cd ../post_processing
python post_processing.py --prediction_path ../finetuning/result_submission/prediction.json --abstention_path ../finetuning/result_submission/abstention_prediction.json