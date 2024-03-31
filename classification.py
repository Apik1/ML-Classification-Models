import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

##########
# Part 1 #
##########

def generate_random_points(size=10, low=0, high=1):
    data = (high - low) * np.random.random_sample((size, 2)) + low
    return data

N = 20 # number of samples in each class
X1 = generate_random_points(N, 0, 1)
y1 = np.ones(N)
X2 = generate_random_points(N, 1, 2)
y2 = np.zeros(N)
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)
indices = np.arange(2*N)
np.random.shuffle(indices)
X = X[indices, :]
y = y[indices]

##########
# Part 2 #
##########

def knn(X, y, test_sample, k=3):
    distance = np.sqrt(((X - test_sample) ** 2).sum(axis=1))
    index = np.argsort(distance)[:k]
    nearest_labels = y[index]
    predicted_class = np.argmax(np.bincount(nearest_labels.astype(int)))
    return predicted_class, X[index], y[index]

# Testing sample
test_sample = np.array([1.0, 1.0])

# Predict the class and the K nearest neighbors
predicted_class, neighbors, neighbor_classes = knn(X, y, test_sample, 3)

print("Predicted class:", predicted_class)
print("K nearest neighbors:", neighbors)
print("Neighbors' classes:", neighbor_classes)

##########
# Part 3 #
##########

# Plot training data
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(test_sample[0], test_sample[1], color='green', label='Testing Sample')

# Highlight neighbors
for neighbor in neighbors:
    plt.scatter(neighbor[0], neighbor[1], facecolors='none', edgecolors='black', s=100)

plt.legend()
plt.show()

######
# P2 #
######

# Generate the dataset for Part 2
N = 20 
X1 = generate_random_points(N, 0, 1.5)
y1 = np.ones(N)
X2 = generate_random_points(N, 0.5, 2)
y2 = np.zeros(N)
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)
indices = np.arange(2*N)
np.random.shuffle(indices)
X = X[indices, :]
y = y[indices]

def manual_k_fold(X, y, num_folds=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices) 
    fold_sizes = np.full(num_folds, len(X) // num_folds, dtype=int)
    fold_sizes[:len(X) % num_folds] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        yield train_indices, test_indices
        current = stop

def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        prediction = np.argmax(np.bincount(nearest_labels.astype(int)))
        predictions.append(prediction)
    return predictions

k_values = [1, 3, 5, 7, 9]
overall_accuracies = {}

for k in k_values:
    fold_accuracies = []
    for train_indices, test_indices in manual_k_fold(X, y, num_folds=5):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        y_pred = knn_predict(X_train, y_train, X_test, k)
        accuracy = calculate_accuracy(y_test, y_pred)
        fold_accuracies.append(accuracy)
    
    overall_accuracies[k] = fold_accuracies
    print(f"K={k}, Accuracies per fold: {fold_accuracies}, Average Accuracy: {np.mean(fold_accuracies)}")

optimal_k = max(overall_accuracies, key=lambda k: np.mean(overall_accuracies[k]))
print(f"Optimal K: {optimal_k}")

######
# P3 #
######

# Part 1: Generate data for part 3
N = 20  # number of samples in each class
X1 = generate_random_points(N, 0, 1)
y1 = np.ones(N)
X2 = generate_random_points(N, 1, 2)
y2 = np.zeros(N)
X_train = np.concatenate((X1, X2), axis=0)
y_train = np.concatenate((y1, y2), axis=0)
indices = np.arange(2 * N)
np.random.shuffle(indices)
X_train = X_train[indices, :]
y_train = y_train[indices]

# Generate testing set manually
X_test = np.array([[0.5, 0.5], [1, 1], [1.5, 1.5]])

# Part 2: Implement logistic regression and make prediction
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    w = np.zeros(n + 1)
    X_bias = np.column_stack((np.ones(m), X))

    for _ in range(num_iterations):
        z = np.dot(X_bias, w)
        h = sigmoid(z)
        gradient = np.dot(X_bias.T, (h - y)) / m
        w -= learning_rate * gradient

    return w

w = logistic_regression(X_train, y_train)
y_pred_logistic = sigmoid(np.dot(np.column_stack((np.ones(len(X_test)), X_test)), w)) >= 0.5

print("Predicted classes (Logistic Regression):")
print(y_pred_logistic.astype(int))

# Part 3: implement SVM
svm_model = SVC(kernel='linear', C=1000)
svm_model.fit(X_train, y_train) 
y_pred_svm = svm_model.predict(X_test)

# Print results
print("Predicted classes (SVM):")
print(y_pred_svm)

# Part 4: Plot  
plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], color='r', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='b', label='Class 2')

x1_range = np.arange(0, 2, 0.1)
x2_range = -(w[0] + w[1] * x1_range) / w[2]
plt.plot(x1_range, x2_range, color='g', label='Logistic Regression')

x1_range = np.arange(0, 2, 0.1)
x2_range = -(svm_model.intercept_[0] + svm_model.coef_[0][0] * x1_range) / svm_model.coef_[0][1]
plt.plot(x1_range, x2_range, color='m', label='SVM')

plt.legend()
plt.title('Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

################
# Extra Credit #
################

# Set letter counts
counts_class_1 = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
counts_class_2 = {'a': 0, 'b': 0, 'c': 0, 'd': 0}

# add data
training_data = [("abc", 1), ("abbc", 1), ("aacd", 1), ("bcdd", 1), ("abc", 1),
                 ("ccd", 2), ("add", 2), ("bdd", 2), ("aac", 2), ("ad", 2)]

# Count letters in each class
for string, class_label in training_data:
    letters_present = set(string)
    for letter in letters_present:
        if class_label == 1:
            counts_class_1[letter] += 1
        else:
            counts_class_2[letter] += 1

# Calculate probabilities
probabilities_class_1 = {letter: count / 5 for letter, count in counts_class_1.items()}
probabilities_class_2 = {letter: count / 5 for letter, count in counts_class_2.items()}
prior_class_1 = 5 / 10
prior_class_2 = 5 / 10

#Print Answers
print("Probabilities Class 1:", probabilities_class_1)
print("Probabilities Class 2:", probabilities_class_2)

print("\n1. For each test string, I calculated the product of the probabilities of each letter in the string for each class.")
print("2. I multiplied this with the prior probability of the class.")
print("3. The class with the higher resulting probability is chosen for the predicted class.\n")

def classify_string(test_string, probabilities_class_1, probabilities_class_2, prior_class_1, prior_class_2):
    letters_in_test = set(test_string)
    prob_class_1 = np.log(prior_class_1)
    prob_class_2 = np.log(prior_class_2)
    
    for letter in letters_in_test:
        if letter in probabilities_class_1:
            prob_class_1 += np.log(probabilities_class_1[letter])
        if letter in probabilities_class_2:
            prob_class_2 += np.log(probabilities_class_2[letter])
            
    return 1 if prob_class_1 > prob_class_2 else 2

# Classify test strings
test_strings = ["abbd", "bbcc"]
for test in test_strings:
    class_label = classify_string(test, probabilities_class_1, probabilities_class_2, prior_class_1, prior_class_2)
    print(f"Test String: '{test}' classified as Class {class_label}")
