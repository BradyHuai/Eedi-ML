import item_response as ir
from utils import *

import numpy as np
import matplotlib.pyplot as plt

"""
Load data from csv files
"""
def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    return train_data, valid_data, test_data

"""
Sample a batch of data from the dataset, using different seed
"""
def sample_data(train_data, batch_size, seed):
    np.random.seed(seed)

    indices = np.random.randint(low=0, high=len(train_data["user_id"]), size=batch_size)

    sample = {"user_id": [], "question_id": [], "is_correct": []}
    for i in indices:
        sample["user_id"].append(train_data["user_id"][i])
        sample["question_id"].append(train_data["question_id"][i])
        sample["is_correct"].append(train_data["is_correct"][i])

    return sample

"""
Predict the result of the dataset
"""
def predict(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = ir.sigmoid(x)
        if p_a >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred

"""
Evaluate the ensembled model
"""
def evaluate(prediction1, prediction2, prediction3, valid_data):
    predictions = []
    for i in range(len(prediction1)):
        curr = (prediction1[i] + prediction2[i] + prediction3[i]) / 3
        predictions.append(curr >= 0.5)

    return np.sum((valid_data["is_correct"] == np.array(predictions))) / len(valid_data["is_correct"])


if __name__ == '__main__':
    SEEDS = [1004, 916, 120]

    train_data, valid_data, test_data = load_data()

    sample1 = sample_data(train_data, len(train_data["user_id"]), SEEDS[0])
    sample2 = sample_data(train_data, len(train_data["user_id"]), SEEDS[1])
    sample3 = sample_data(train_data, len(train_data["user_id"]), SEEDS[2])
    
    theta1, beta1, _, _, _ = ir.irt(sample1, valid_data, 0.005, 30)
    theta2, beta2, _, _, _ = ir.irt(sample2, valid_data, 0.01, 30)
    theta3, beta3, _, _, _ = ir.irt(sample3, valid_data, 0.015, 30)

    p1_val = predict(valid_data, theta1, beta1)
    p2_val = predict(valid_data, theta2, beta2)
    p3_val = predict(valid_data, theta3, beta3)
    print("Validation accuracy: {}".format(evaluate(p1_val, p2_val, p3_val, valid_data)))

    p1_test = predict(test_data, theta1, beta1)
    p2_test = predict(test_data, theta2, beta2)
    p3_test = predict(test_data, theta3, beta3)
    print("Testing accuracy: {}".format(evaluate(p1_test, p2_test, p3_test, test_data)))
