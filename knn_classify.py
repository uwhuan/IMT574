import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

df = pd.read_csv('./data/stayzilla_combined_clean.csv')

# Define the KNeighborsClassifier which takes in the number of neighbors and returns the accuracy and f1 scores
def knn(X, y, n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    a = accuracy_score(y_test, y_pred)
    b = f1_score(y_test, y_pred.round(), zero_division=1)
    # print(confusion_matrix(y_test, y_pred))
    return a, b

# This function will iterate the knn function m times and calculate the average
def multiple_knn(m, n):
    s1 = 0
    s2 = 0
    for i in range(0, m):
        acc, f = knn(X, y, n)
        s1 = s1+acc
        s2 = s2+f
    return s1/m, s2/m

# Define the independent variables
X = df[['latitude', 'longitude','room_price', 'Newspaper', 'Parking',
       'Card Payment', 'Elevator', 'Pickup & Drop', 'Veg Only', 'Bar', 'Laundry', 'adult_occupancy',
       'child_occupancy', 'num_amenities', 'zones', 'descr_len', 'deluxe', 'num_simhotel', 'wifi', 'ac', 'breakfast',
       'response_label', 'Apartment', 'Homestay', 'Hotel', 'House', 'Lodge', 'Resort', 'Spa', 'Villa']]

y = df['service_value']

accuracy = []
f1 = []
ni = range(0, 300)

# loop over the number of neighbors to find a best fit
for i in ni:
    acc, f = multiple_knn(20, i)
    accuracy.append(acc)
    f1.append(f)

# Plot the accuracy and f1 scores
plt.figure(figsize=(10,7))
plt.plot(ni, accuracy, '-o')
plt.title("Prediction Accuracy")
plt.figure(figsize=(10,7))
plt.plot(ni, f1, '-o')
plt.title("F1 score")

# Zoom in to find the best match
p = 140
q = 160
plt.figure(figsize=(10,7))
plt.plot(ni[p:q], accuracy[p:q], '-o')
plt.title("Prediction Accuracy")
plt.figure(figsize=(10,7))
plt.plot(ni[p:q], f1[p:q], '-o')
plt.title("F1 score")


print('Best accuracy score is achieved when n = {}'.format(np.argmax(accuracy)))
print('Best f1 score is achieved when n = {}'.format(np.argmax(f1)))