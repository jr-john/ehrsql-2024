import json
import fire
import pickle
from tqdm import tqdm


def main(
    prediction_data_path,  # path to the prediction data
    abstention_data_path,  # path to the abstention data
    output_path="train.pkl",  # path to save the output
):
    with open(prediction_data_path, "rb") as f:
        prediction_data = pickle.load(f)
    with open(abstention_data_path, "rb") as f:
        abstention_data = pickle.load(f)

    input_data = prediction_data + abstention_data

    with open(output_path, "wb") as f:
        pickle.dump(input_data, f)


if __name__ == "__main__":
    fire.Fire(main)
