from utils import *

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # DONE:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    sum_0 = 0
    sum_1 = 0
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    for idx, value in enumerate(is_correct):
        i = user_id[idx]
        j = question_id[idx]
        if value == 0:
            sum_0 += np.log((1 - sigmoid(theta[i] - beta[j])))
        if value == 1:
            sum_1 += np.log(sigmoid(theta[i] - beta[j]))

    log_lklihood = sum_0 + sum_1

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # DONE:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    dl_theta = np.zeros(theta.shape)
    dl_beta = np.zeros(beta.shape)

    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    # let i represent student id
    # let j represent question id

    # update theta
    for idx, value in enumerate(is_correct):
        i = user_id[idx]
        j = question_id[idx]
        temp = sigmoid(theta[i] - beta[j])
        # 0 is incorrect
        if value == 0:
            dl_theta[i] -= temp
        # 1 is correct
        if value == 1:
            dl_theta[i] += (1 - temp)
    theta = theta + lr * dl_theta

    # update beta
    for idx, value in enumerate(is_correct):
        i = user_id[idx]
        j = question_id[idx]
        temp = sigmoid(theta[i] - beta[j])
        # 0 is incorrect
        if value == 0:
            dl_beta[j] += temp
        # 1 is correct
        if value == 1:
            dl_beta[j] -= (1 - temp)
    beta = beta + lr * dl_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)

    MODIFIED return values to help with plotting later
    """
    # DONE: Initialize theta and beta.
    n_students = len(set(data['user_id']))
    n_questions = len(set(data['question_id']))
    theta = np.empty(n_students)
    theta.fill(1)
    beta = np.empty(n_questions)
    beta.fill(0.5)
    train_llk_lst = []
    val_llk_lst = []
    val_acc_lst = []

    for _ in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)

        train_llk_lst.append(-train_neg_lld)
        val_llk_lst.append(-val_neg_lld)
        val_acc_lst.append(score)

        print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # DONE: You may change the return values to achieve what you want.
    return theta, beta, train_llk_lst, val_llk_lst, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # DONE:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 20
    theta, beta, train_llk_lst, val_llk_lst, val_acc_lst = irt(train_data, val_data, lr, iterations)

    plt.plot(val_acc_lst)
    plt.ylabel("Validation accurary")
    plt.xlabel("Iteration Number")
    plt.title("Validation Accurary as a Function of Number of Iterations")
    plt.show()

    plt.plot(train_llk_lst)
    plt.ylabel("log-likelihood")
    plt.xlabel("Iteration Number")
    plt.title("Training log-likelihood as a Function of Number of Iterations")
    plt.show()

    plt.plot(val_llk_lst)
    plt.ylabel("log-likelihood")
    plt.xlabel("Iteration Number")
    plt.title("Validation log-likelihood as a Function of Number of Iterations")
    plt.show()

    final_val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_acc = evaluate(data=test_data, theta=theta, beta=beta)

    print(f"The final validation accuracy is {final_val_acc}")
    print(f"The final testing accuracy is {final_test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # DONE:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1 = 9
    j2 = 88
    j3 = 102

    p1 = {theta_i: sigmoid(theta_i - beta[j1]) for theta_i in set(theta)}
    p2 = {theta_i: sigmoid(theta_i - beta[j2]) for theta_i in set(theta)}
    p3 = {theta_i: sigmoid(theta_i - beta[j3]) for theta_i in set(theta)}
    lst = sorted(p1.items())
    x, y = zip(*lst)
    plt.plot(x, y, label=f"Question {j1}")
    lst = sorted(p2.items())
    x, y = zip(*lst)
    plt.plot(x, y, label=f"Question {j2}")
    lst = sorted(p3.items())
    x, y = zip(*lst)
    plt.plot(x, y, label=f"Question {j3}")
    plt.ylabel("Probability of Correct Answer")
    plt.xlabel(r"$\theta$")
    plt.title(r"Probability of Correct Answer as a Function of $\theta$")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
