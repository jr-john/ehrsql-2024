import os
import re
import json
import fire
import torch
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


def post_process(answer):
    answer = answer.strip()
    if answer == "I do not know":
        answer = "null"
    else:
        answer = answer.replace("\n", " ")
        answer = re.sub("[ ]+", " ", answer)
    return answer


def write_json(path, file):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, "w+") as f:
        json.dump(file, f)


def main(
    model_path,  # path to the model
    task,  # task to perform (Options - prediction, abstention_prediction)
    schema_prompt_path,  # path to the schema prompt
    result_dir="result_submission",  # path to save the output
    ehrsql_path="../../ehrsql-2024",  # path to the EHRSQL git repository
    num_gpus=1,  # number of GPUs to use
    gpu_memory_utilization=0.9,  # GPU memory utilization
    seed=42,  # seed value for reproducibility
):
    test_data_path = f"{ehrsql_path}/data/mimic_iv/test/data.json"

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
    torch.manual_seed(seed)

    with open(test_data_path, "r") as file:
        test_data = json.load(file)

    with open(schema_prompt_path, "r") as file:
        schema_prompt = file.read()

    stopwords = ["[/SQL]"]
    if "abstain" in task.lower() or "abstention" in task.lower():
        stopwords = ["[/CLASS]"]

    class Model:
        def __init__(self):
            self.llm = LLM(
                model=model_path,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=num_gpus,
            )
            self.sampling_params = SamplingParams(
                max_tokens=4096, temperature=0, stop=stopwords, seed=seed, logprobs=1
            )

        def generate(self, input_data):
            """
            Arguments:
                input_data: list of python dictionaries containing 'id' and 'input'
            Returns:
                labels: python dictionary containing sql prediction or 'null' values associated with ids
            """

            labels = {}
            scores = {}

            inputs = [sample["input"] for sample in input_data]
            ids = [sample["id"] for sample in input_data]
            outputs = self.llm.generate(inputs, self.sampling_params)

            for idx, output in enumerate(outputs):
                labels[ids[idx]] = post_process(output.outputs[0].text)

                score = 0
                probs = output.outputs[0].logprobs
                for prob in probs:
                    score += list(prob.values())[0].logprob
                scores[ids[idx]] = score

            return labels, scores

    myModel = Model()
    data = test_data["data"]

    sqlcoder_prompt = """### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{table_metadata_string}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{user_question}[/QUESTION]
[SQL]"""

    if "abstain" in task.lower() or "abstention" in task.lower():
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
[CLASS]"""

    input_data = []

    for sample in tqdm(data):
        sample_dict = {}
        sample_dict["id"] = sample["id"]

        prompt = sqlcoder_prompt.format(
            user_question=sample["question"],
            table_metadata_string=schema_prompt,
        )
        sample_dict["input"] = prompt

        input_data.append(sample_dict)

    label_y, scores = myModel.generate(input_data)

    prediction_output_path = os.path.join(result_dir, "prediction.json")
    scores_output_path = os.path.join(result_dir, "scores.json")
    if "abstain" in task.lower() or "abstention" in task.lower():
        prediction_output_path = os.path.join(result_dir, "abstention_prediction.json")
        scores_output_path = os.path.join(result_dir, "abstention_scores.json")

    write_json(prediction_output_path, label_y)
    write_json(scores_output_path, scores)


if __name__ == "__main__":
    fire.Fire(main)
