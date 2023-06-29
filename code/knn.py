from sklearn.impute import KNNImputer
from utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy at k = {}: {}".format(k, acc))
    return acc


def sparse_matrix_evaluate_item(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_question_id, cur_user_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_question_id, cur_user_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(np.transpose(matrix))
    acc = sparse_matrix_evaluate(valid_data, np.transpose(mat))
    print("Validation Accuracy at k = {}: {}".format(k, acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def run_and_plot(k_lst, user, sparse_matrix, data):
    accuracy_rates = []
    for k in k_lst:
        if user is True:
            acc = knn_impute_by_user(sparse_matrix, data, k)
        else:
            acc = knn_impute_by_item(sparse_matrix, data, k)
        accuracy_rates.append(acc)

    # plotting the points
    plt.plot(k_lst, accuracy_rates, '-o')

    # labelling x, y axes
    plt.xlabel('k')
    plt.ylabel('accuracy rate')

    # graph title
    if user is True:
        plt.title('k v/s accuracy rate for test data for user-based collaborative filtering')
    else:
        plt.title('k v/s accuracy rate for test data for item-based collaborative filtering')

    plt.savefig('knn.png')
    return accuracy_rates


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_lst = [1, 6, 11, 16, 21, 26]

    knn_impute_by_item(sparse_matrix, test_data, 21)

    # for user - part a
    acc = np.array(run_and_plot(k_lst, True, sparse_matrix, val_data))

    # part b
    max_acc = np.argmax(acc)
    k_star = k_lst[max_acc]
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(k_star, test_acc_user)

    # for item - part c
    acc_item = np.array(run_and_plot(k_lst, False, sparse_matrix, val_data))
    max_acc_item = np.argmax(acc_item)
    k_star_item = k_lst[max_acc_item]
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)
    print(k_star_item, test_acc_item)

    # part d
    print("Test performance for user based: ", test_acc_user, "\nTest performance for item based: ", test_acc_item)
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
