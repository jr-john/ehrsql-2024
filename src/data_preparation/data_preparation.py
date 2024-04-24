import json
import fire
import pickle
from tqdm import tqdm


def main(
    schema_prompt_path,  # path to the schema prompt
    output_path="train.pkl",  # path to save the output
    ehrsql_path="../../ehrsql-2024",  # path to the EHRSQL git repository
):
    train_data_path = f"{ehrsql_path}/data/mimic_iv/train/data.json"
    train_label_path = f"{ehrsql_path}/data/mimic_iv/train/label.json"

    with open(train_data_path, "r") as file:
        train_data = json.load(file)

    with open(train_label_path, "r") as file:
        train_label = json.load(file)

    with open(schema_prompt_path, "r") as file:
        schema_prompt = file.read()

    sqlcoder_prompt = """### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{user_question}[/QUESTION]
[SQL]{answer}[/SQL]"""

    data = train_data["data"]
    input_data = []

    for sample in tqdm(data):
        if train_label[sample["id"]] == "null":
            answer = "I do not know"
        else:
            answer = train_label[sample["id"]]

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
