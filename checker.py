import pandas as pd
import numpy as np

# Function to check how many characters match in the two strings
def check(pred: str, true: str):
    correct = 0
    for a, b in zip(pred, true):
        if a == b:
            correct += 1

    # Prediction is more than 8 letters, so penalize for every extra letter.
    correct -= max(0, len(pred) - len(true))
    correct = max(0, correct)
    return correct

# Function to score the model's performance
def evaluate(encoder, decoder):

    # Train data
    print("Obtaining results for training data:")
    train_data = pd.read_csv("train_data.csv").to_numpy()
    results = {
        "pred": [],
        "true": [],
        "score": [],
    }
    correct = [0 for _ in range(9)]
    for x, y in train_data:
        pred = decoder.predict(encoder.predict(x))
        score = check(pred, y)
        results["pred"].append(pred)
        results["true"].append(y)
        results["score"].append(score)

        correct[score] += 1
    print("Train dataset results:")
    for num_chr in range(9):
        print(
            f"Number of predictions with {num_chr} correct predictions: {correct[num_chr]}"
        )
    points = sum(correct[4:6]) * 0.5 + sum(correct[6:])
    print(f"Points: {points}")
    # Save predicitons and true sentences to inspect manually if required.
    pd.DataFrame.from_dict(results).to_csv("results_train.csv", index=False)

    #----------------------------------------------------------------------------------

    print("Obtaining metrics for eval data:")
    eval_data = pd.read_csv("eval_data.csv").to_numpy()
    results = {
        "pred": [],
        "true": [],
        "score": [],
    }
    correct = [0 for _ in range(9)]
    for x, y in eval_data:
        pred = decoder.predict(encoder.predict(x))
        score = check(pred, y)
        results["pred"].append(pred)
        results["true"].append(y)
        results["score"].append(score)

        correct[score] += 1
    print("Eval dataset results:")
    for num_chr in range(9):
        print(
            f"Number of predictions with {num_chr} correct predictions: {correct[num_chr]}"
        )
    points = sum(correct[4:6]) * 0.5 + sum(correct[6:])
    marks = round(min(2, points / 1400 * 2) * 2) / 2  # Rounds to the nearest 0.5
    print(f"Points: {points}")
    print(f"Marks: {marks}")
    # Save predicitons and true sentences to inspect manually if required.
    pd.DataFrame.from_dict(results).to_csv("results_eval.csv", index=False)

