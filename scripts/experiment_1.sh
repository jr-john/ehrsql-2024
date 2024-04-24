#!/bin/bash

# Text-to-SQL Generation (Initial Prompt)
cd src/data_preparation
python data_preparation.py --schema_prompt_path ../../schema_prompt/experiment_1.txt --output_path prediction_train.pkl

cd ../finetuning
python finetuning.py --train_data_path ../data_preparation/prediction_train.pkl --num_train_epochs 2
last_checkpoint=$(ls -Art sqlcoder-EHRSQL | tail -n 1)
python merge_model.py --model_path sqlcoder-EHRSQL/$last_checkpoint --output_path sqlcoder-EHRSQL/merged-model
python inference.py --model_path sqlcoder-EHRSQL/merged-model --task prediction --schema_prompt_path ../../schema_prompt/experiment_1.txt