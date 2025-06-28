import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("hw06_predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("hw06_predicted_probabilities2.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    # your implementation starts below

    #sort, because we want midpoints
    sorted_probs = np.sort(predicted_probabilities)
    N = sorted_probs.shape[0]

    thresholds = np.zeros(N + 1)
    thresholds[0] = sorted_probs[0] * 0.5

    # intermediates
    for i in range(1, N):
        thresholds[i] = (sorted_probs[i - 1] + sorted_probs[i]) / 2.0

    thresholds[N] = (sorted_probs[-1] + 1.0) / 2.0

    # your implementation ends above
    return thresholds

thresholds1 = calculate_threholds(predicted_probabilities1)
print(thresholds1)

thresholds2 = calculate_threholds(predicted_probabilities2)
print(thresholds2)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    # your implementation starts below

    # Count actual positives (label == 1) and negatives (label == -1)
    pos_cnt = np.sum(true_labels == 1)
    neg_cnt = np.sum(true_labels == -1)

    fp_rates = np.zeros(thresholds.shape[0])
    tp_rates = np.zeros(thresholds.shape[0])

    # For each cutoff, mark predictions and accumulate TP/FP counts
    for idx, midpoint in enumerate(thresholds):
        predicted_positive = predicted_probabilities >= midpoint

        tp = np.sum(predicted_positive & (true_labels == 1))
        fp = np.sum(predicted_positive & (true_labels == -1))

        # Normalize
        tp_rates[idx] = tp / pos_cnt
        fp_rates[idx] = fp / neg_cnt

    # your implementation ends above
    return fp_rates, tp_rates

fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    # your implementation starts below
    widths = fp_rates[:-1] - fp_rates[1:]
    heights = (tp_rates[:-1] + tp_rates[1:]) / 2.0
    auroc = np.sum(widths * heights)
    # your implementation ends above
    return auroc

auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))