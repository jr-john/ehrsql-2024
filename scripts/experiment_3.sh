#!/bin/bash

# Abstention
cd src/data_preparation
python schema_generation.py
python abstention_data_preparation.py --schema_prompt_path schema_prompt.txt --output_path abstention_train.pkl

cd ../finetuning
python finetuning.py --train_data_path ../data_preparation/abstention_train.pkl --num_train_epochs 6 --task EHRSQL-Abstention
last_checkpoint=$(ls -Art sqlcoder-EHRSQL-Abstention | tail -n 1)
python merge_model.py --model_path sqlcoder-EHRSQL-Abstention/$last_checkpoint --output_path sqlcoder-EHRSQL-Abstention/merged-model
python inference.py --model_path sqlcoder-EHRSQL-Abstention/merged-model --task abstention_prediction --schema_prompt_path ../data_preparation/schema_prompt.txt

# Text-to-SQL Generation
cd ../data_preparation
python data_preparation.py --schema_prompt_path schema_prompt.txt --output_path prediction_train.pkl

cd ../finetuning
python finetuning.py --train_data_path ../data_preparation/prediction_train.pkl --num_train_epochs 2
last_checkpoint=$(ls -Art sqlcoder-EHRSQL | tail -n 1)
python merge_model.py --model_path sqlcoder-EHRSQL/$last_checkpoint --output_path sqlcoder-EHRSQL/merged-model
python inference.py --model_path sqlcoder-EHRSQL/merged-model --task prediction --schema_prompt_path ../data_preparation/schema_prompt.txt

# Error Filtering
cd ../post_processing
python post_processing.py --prediction_path ../finetuning/result_submission/prediction.json --abstention_path ../finetuning/result_submission/abstention_prediction.json