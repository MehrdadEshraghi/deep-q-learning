import numpy as np
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def calculate_f1_score(optimal_policy_to_evaluate_name, optimal_policy_iteration_name='optimal_policy.txt', ):
    optimal_policy_iteration = np.genfromtxt(optimal_policy_iteration_name).reshape((1922, 1)) - 1

    optimal_policy_saved = np.genfromtxt(optimal_policy_to_evaluate_name, delimiter=',')[:, -1].reshape((1922, 1)) - 1


    optimal_policy_saved[optimal_policy_saved == 0] = 5

    optimal_policy_saved[optimal_policy_saved == 1] = 0

    optimal_policy_saved[optimal_policy_saved == 5] = 1


    optimal_policy_iteration[optimal_policy_iteration == 0] = 5

    optimal_policy_iteration[optimal_policy_iteration == 1] = 0

    optimal_policy_iteration[optimal_policy_iteration == 5] = 1

    return f1_m(optimal_policy_iteration, optimal_policy_saved)

def save_optimal_policy(optimal_policy, filename='optimal_policy.csv'):
    print(optimal_policy.shape)
    with open(filename, 'w') as file:
        for i in range(optimal_policy.shape[2]):
            for j in range(optimal_policy.shape[0]):
                for k in range(optimal_policy.shape[1]):
                    file.write(f"{j},{k},{i + 1},{optimal_policy[j][k][i] + 1}\n")
    print(f"Optimal policy with queue saved to {filename}")

def calculate_optimal_policy(agent):
    optimal_policy = np.zeros((31, 31, 2), dtype=int)
    for i in range(31):
        for j in range(31):
            for k in range(2):
                state = np.array([[i, j, k]])
                action = agent.act(state)
                optimal_policy[i][j][k] = action
    return optimal_policy