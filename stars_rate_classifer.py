import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense


raw = pd.read_csv('./data/stayzilla_com-travel_sample.csv')
raw = raw[1139:]

features = [
       'hotel_star_rating', 
       'highlight_value', 
       'landmark', 
       'service_value', 
       'amenities', 
       'room_types'
    ]

raw = raw[features]
raw['highlight_value'] = raw['highlight_value'].map(lambda x: x.replace('What should you know?', '') if pd.notnull(x) else '')
raw['highlight_value'] = raw['highlight_value'].map(lambda x: x.replace('\n', '') if pd.notnull(x) else '')

# extract landmark dist
landmark_ser = raw['landmark']
landmark_feat = ['airport_dist', 'railway_dist', 'bus_stand_dist']
for i in range(len(landmark_feat)):
    raw[landmark_feat[i]] = landmark_ser.map(lambda x: int(x.split(' | ')[i].split(' Km from ')[0]) if pd.notnull(x) else 0 )

# get room_type counts 
room_type_ser = raw['room_types']
raw['room_types_count'] = room_type_ser.map(lambda x: len(x.split(' | ')))


# vectorizing service_value & amenities
def vectorizing(colname, df):
    df[colname] = df[colname].str.replace(' \|*\| ', '|', regex=True)
    service_value_df = df[colname].str.get_dummies('|') 
    df = pd.concat([df, service_value_df], axis=1)
    return df

raw = vectorizing('service_value', raw)
raw = vectorizing('amenities', raw)
del raw['Car Parking']

other_feat_arr = raw.iloc[:, len(features):].values

y = np.asarray(raw['hotel_star_rating'])
corpus = raw["highlight_value"]
corpus_ls = list(corpus)
vectorizer = CountVectorizer()
desc_arr = vectorizer.fit_transform(corpus_ls).toarray()
X = np.concatenate((desc_arr, other_feat_arr), axis=1)

# get result of classic machine learning classifiers
def get_ML_result(method, kernal):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    if method == "Gaussian":
        gnb = GaussianNB()
        gnb.fit(X_train,y_train.ravel())
        y_pred = gnb.predict(X_test)
    if method == "Multinomial":
        mnb = MultinomialNB()
        mnb.fit(X_train,y_train.ravel())
        y_pred = mnb.predict(X_test)
    if method == "KNN":
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    if method == "SVM": 
        svm = SVC(kernel= kernal)
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_test)
    if method == "randforest": 
        rfc = RandomForestClassifier(n_estimators=50)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
    return (
            accuracy_score(y_test, y_pred), 
            f1_score(y_test, y_pred, average='macro')
        )

# experiment basic structured nn result
def get_NN_result(X, y):
    onehotencoder = OneHotEncoder(categories='auto')
    y = onehotencoder.fit_transform(pd.DataFrame(y)).toarray()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    nn = Sequential()
    nn.add(Dense(1109, activation='relu'))
    nn.add(Dense(688, activation='relu'))
    nn.add(Dense(288, activation='relu'))
    nn.add(Dense(90, activation='relu'))
    nn.add(Dense(5, activation='sigmoid'))

    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train,y_train)
    y_pred = nn.predict(X_test)
    y_pred = (y_pred>0.5)
    from sklearn.metrics import accuracy_score, classification_report
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))


def multiple_runs(method, kernal='', n=50):
    print(method + ' ' + kernal)
    accuracy = f1 = 0
    for _ in range(n):
        cur_accuracy, cur_f1 = get_ML_result(method, kernal)
        accuracy += cur_accuracy
        f1 += cur_f1

    print("Result of {0}:\nAVE Accuracy: {1},\nAVE F1: {2}\n\n".format(
            method,
            accuracy/n, 
            f1/n))
    return 

if __name__ == '__main__':
    # multiple_runs("Gaussian")
    # multiple_runs("Multinomial")
    # multiple_runs("KNN")
    multiple_runs("SVM", "linear")
    multiple_runs("SVM", "poly")
    multiple_runs("SVM", "sigmoid")
    # multiple_runs("randforest")
    get_NN_result(X, y)
