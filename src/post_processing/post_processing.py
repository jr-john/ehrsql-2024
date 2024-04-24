import os
import json
import fire
import multiprocessing as mp
import matplotlib.pyplot as plt
from utils import post_process_sql, execute_all, execute_all_distributed


def main(
    prediction_path,  # path to the prediction file
    scores_path=None,  # path to the confidence score file
    abstention_path=None,  # path to the abstention prediction file
    threshold=-0.4,  # threshold for confidence score
    output_path="prediction.json",  # path to save the output
    ehrsql_path="../../ehrsql-2024",  # path to the EHRSQL git repository
):
    db_path = f"{ehrsql_path}/data/mimic_iv/mimic_iv.sqlite"

    with open(prediction_path) as f:
        pred_dict = json.load(f)

    if scores_path:
        with open(scores_path) as f:
            scores_dict = json.load(f)

    pred_dict = {id_: post_process_sql(pred_dict[id_]) for id_ in pred_dict}

    if not os.path.exists(db_path):
        raise Exception("File does not exist: %s" % db_path)

    num_workers = mp.cpu_count()
    if num_workers > 1:
        pred_result = execute_all_distributed(
            pred_dict, db_path, tag="pred", num_workers=num_workers
        )
    else:
        pred_result = execute_all(pred_dict, db_path, tag="pred")

    if scores_path:
        scores = list(scores_dict.values())
        scores = [score for score in scores if score > -1]

        fig = plt.figure(figsize=(15, 10))
        plt.hist(scores, bins=1000)
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.show()
        fig.savefig("Confidence Plot.pdf", bbox_inches="tight")

        scores = list(scores_dict.values())
        scores = [score for score in scores if score >= threshold]
        print(f"Number of predictions with score >= {threshold}: {len(scores)}")

    if abstention_path:
        with open(abstention_path) as f:
            abstention_dict = json.load(f)

    with open(prediction_path) as f:
        result_dict = json.load(f)

    count = 0
    for key, pred in pred_result.items():
        if pred == "error_pred":
            result_dict[key] = "null"
            count += 1
        elif abstention_path and abstention_dict[key] == "unanswerable":
            result_dict[key] = "null"
            count += 1
        elif scores_path and scores_dict[key] < threshold:
            result_dict[key] = "null"
            count += 1

    print(f"Number of edits: {count}")

    output_path = output_path.replace(".json", "")

    if scores_path:
        output_path = f"{output_path}_postproc_{-threshold}.json"
    else:
        output_path = f"{output_path}_postproc.json"

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
