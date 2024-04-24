import json
import fire
import random
import pickle
from tqdm import tqdm


def main(
    schema_prompt_path,  # path to the schema prompt
    output_path="train.pkl",  # path to save the output
    ehrsql_path="../../ehrsql-2024",  # path to the EHRSQL git repository
    seed=42,  # seed value for reproducibility
):
    random.seed(seed)

    train_data_path = f"{ehrsql_path}/data/mimic_iv/train/data.json"
    train_label_path = f"{ehrsql_path}/data/mimic_iv/train/label.json"

    with open(train_data_path, "r") as file:
        train_data = json.load(file)

    with open(train_label_path, "r") as file:
        train_label = json.load(file)

    with open(schema_prompt_path, "r") as file:
        schema_prompt = file.read()

    abstention_data = list()
    gen_data = list()

    for sample in train_data["data"]:
        if train_label[sample["id"]] == "null":
            abstention_data.append(sample)
        else:
            gen_data.append(sample)

    gen_data = random.sample(gen_data, len(abstention_data))

    sqlcoder_prompt = """### Task
    Classify whether the question is answerable or unanswerable - [QUESTION]{user_question}[/QUESTION]

    ### Instructions
    - Remember that answerable question is one that can be answered with the given database
    - Remember that unanswerable question is one that cannot be answered with the given database

    ### Database Schema
    The query will run on a database with the following schema:
    {table_metadata_string}

    ### Answer
    Given the database schema, here is the class of [QUESTION]{user_question}[/QUESTION]
    [CLASS]{answer}[/CLASS]"""

    data = abstention_data + gen_data
    input_data = []

    for sample in tqdm(data):
        if train_label[sample["id"]] == "null":
            answer = "unanswerable"
        else:
            answer = "answerable"

        prompt = sqlcoder_prompt.format(
            user_question=sample["question"],
            table_metadata_string=schema_prompt,
            answer=answer,
        )

        input_data.append(prompt)

    with open(output_path, "wb") as f:
        pickle.dump(input_data, f)


if __name__ == "__main__":
    fire.Fire(main)
