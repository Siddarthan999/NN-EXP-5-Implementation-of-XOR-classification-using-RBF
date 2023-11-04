# NN-EXP-5-Implementation-of-XOR-classification-using-RBF

## AIM:
To classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.
  
## EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
<img width="541" alt="image" src="https://user-images.githubusercontent.com/112920679/201299438-5d1926f9-25e9-4f20-b392-1c112880ef56.png">

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below
<img width="246" alt="image" src="https://user-images.githubusercontent.com/112920679/201299568-d9398233-71d8-41b3-8b08-a39d5b95e3f1.png">

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.


A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.

<img width="261" alt="image" src="https://user-images.githubusercontent.com/112920679/201300944-5510d7f4-ea0f-45ec-875d-87f463927e9d.png">

The RBF of hidden neuron as gaussian function 

<img width="206" alt="image" src="https://user-images.githubusercontent.com/112920679/201302321-a09f72e9-2352-4f88-838c-3324f6c5f57e.png">


## ALGORIHM:

/** Write the Algorithm in steps**/

## PROGRAM:
```
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
def generate_xor_data(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5)
    return X, y
def train_rbf_classifier(X, y, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rbf_features = np.exp(-kmeans.transform(X_scaled) ** 2)

    rbf_classifier = MLPClassifier(hidden_layer_sizes=(n_clusters,), activation='identity', max_iter=10000)
    rbf_classifier.fit(rbf_features, y)

    return kmeans, scaler, rbf_classifier
def predict_rbf_classifier(X, kmeans, scaler, rbf_classifier):
    X_scaled = scaler.transform(X)
    rbf_features = np.exp(-kmeans.transform(X_scaled) ** 2)
    return rbf_classifier.predict(rbf_features)
X_train, y_train = generate_xor_data(200)
X_test, y_test = generate_xor_data(100)

kmeans, scaler, rbf_classifier = train_rbf_classifier(X_train, y_train)
y_pred = predict_rbf_classifier(X_test, kmeans, scaler, rbf_classifier)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot XOR data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, s=100)
plt.title("XOR Data (Training Set)")

# Plot RBF activations in hidden space
X_test_scaled = scaler.transform(X_test)
rbf_features_test = np.exp(-kmeans.transform(X_test_scaled) ** 2)

plt.subplot(1, 2, 2)
plt.scatter(rbf_features_test[:, 0], rbf_features_test[:, 1], c=y_pred, cmap=plt.cm.Paired, s=100)
plt.title("RBF Activations in Hidden Space (Test Set)")

plt.show()
```
## OUTPUT :
```
Accuracy: 0.54
```
![image](https://github.com/Siddarthan999/NN-EXP-5-Implementation-of-XOR-classification-using-RBF/assets/91734840/8b7c9555-7d67-457a-bb47-4774f8fedcd8)

## RESULT:
Thus, classifying the Binary input patterns of XOR data by implementing Radial Basis Function Neural Networks has been implemented and executed successfully.
