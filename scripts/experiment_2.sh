#!/bin/bash

# Text-to-SQL Generation
cd src/data_preparation
python schema_generation.py
python data_preparation.py --schema_prompt_path schema_prompt.txt  --output_path prediction_train.pkl

cd ../finetuning
python finetuning.py --train_data_path ../data_preparation/prediction_train.pkl --num_train_epochs 2
last_checkpoint=$(ls -Art sqlcoder-EHRSQL | tail -n 1)
python merge_model.py --model_path sqlcoder-EHRSQL/$last_checkpoint --output_path sqlcoder-EHRSQL/merged-model
python inference.py --model_path sqlcoder-EHRSQL/merged-model --task prediction --schema_prompt_path ../data_preparation/schema_prompt.txt

# Error Filtering
cd ../post_processing
python post_processing.py --prediction_path ../finetuning/result_submission/prediction.json