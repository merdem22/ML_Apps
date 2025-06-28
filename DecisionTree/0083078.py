import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]



# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below
    
    # the funciton finds the best split for a given set of nodes.
    def find_split(indices):

        best_error = float('inf')
        best_feature = None 
        best_split_value = None
        
        #for each feature
        for feature in range(X_train.shape[1]):
            x_values = X_train[indices, feature]
            y_values = y_train[indices]

            unique_values = np.unique(x_values)
            if unique_values.shape[0] == 1:
                continue  # cannot split if there is only one unique value

            # Candidate split points are midpoints between successive unique values
            split_points = (unique_values[:-1] + unique_values[1:]) / 2

            #for each possible split
            for split_val in split_points:

                y_left = y_values[x_values > split_val]
                y_right = y_values[x_values <= split_val]

                mean_left = np.mean(y_left)
                mean_right = np.mean(y_right)
                error_left = np.sum((y_left - mean_left) ** 2)
                error_right = np.sum((y_right - mean_right) ** 2)
                total_error = error_left + error_right

                if total_error < best_error:
                    best_error = total_error
                    best_feature = feature
                    best_split_value = split_val

        return best_feature, best_split_value

    # initialise root node
    node_indices[1] = np.arange(X_train.shape[0])
    need_split[1] = True
    is_terminal[1] = False

    #do it for the entire tree
    while True:
        split_nodes = [id for id, predicate in need_split.items() if predicate]
        if len(split_nodes) == 0:
            break

        for id in split_nodes:
            indices = node_indices[id]
            need_split[id] = False  # mark as processed

            # pre‑pruning
            if indices.shape[0] <= P:
                is_terminal[id] = True
                node_means[id] = np.mean(y_train[indices])
                continue

            best_feature, best_split_val = find_split(indices)
            if best_feature is None:  # no valid split
                is_terminal[id] = True
                node_means[id] = np.mean(y_train[indices])
                continue

            # store split info
            node_features[id] = best_feature
            node_splits[id] = best_split_val
            is_terminal[id] = False

            left_id = 2 * id# x > val
            right_id = 2 * id + 1

            left_indices = indices[X_train[indices, best_feature] > best_split_val]
            right_indices = indices[X_train[indices, best_feature] <= best_split_val]

            node_indices[left_id] = left_indices
            node_indices[right_id] = right_indices

            is_terminal[left_id] = False
            is_terminal[right_id] = False
            need_split[left_id] = True
            need_split[right_id] = True


    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below

    #traversing the tree
    def helper(x):
        id = 1
        while not is_terminal[id]:
            feature = node_features[id]
            split_val = node_splits[id]
            if x[feature] > split_val:
                id = 2 * id#left
            else:
                id = 2 * id + 1#right
        return node_means[id]

    y_predict = np.array([helper(x) for x in X_query])
    # your implementation ends above
    return(y_predict)



# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    def helper(id, rules):
        if is_terminal[id]:
            rules_array = np.array(rules, dtype=str)
            print(f"Node {id:02d}: {rules_array} => {node_means[id]}")
            return

        feature = node_features[id]
        split_val = node_splits[id]

        str_le = f"x{feature + 1} <= {split_val:.2f}" #less than or equal to
        str_gt = f"x{feature + 1} > {split_val:.2f}" #greater than

        helper(2 * id, rules + [str_gt])
        helper(2 * id + 1, rules + [str_le])

    helper(1, []) #run on root with empty set of rules

    # your implementation ends above

    

P = 256
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)

P_set = [2, 4, 8, 16, 32, 64, 128, 256]
rmse_train = []
rmse_test = []
for P in P_set:
    is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)

    y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
    rmse_train.append(np.sqrt(np.mean((y_train - y_train_hat)**2)))

    y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
    rmse_test.append(np.sqrt(np.mean((y_test - y_test_hat)**2)))

fig = plt.figure(figsize = (8, 4))
plt.semilogx(P_set, rmse_train, "ro-", label = "train", base = 2)
plt.semilogx(P_set, rmse_test, "bo-", label = "test", base = 2)
plt.legend()
plt.xlabel("$P$")
plt.ylabel("RMSE")
plt.show()
fig.savefig("decision_tree_P_comparison.pdf", bbox_inches = "tight")
