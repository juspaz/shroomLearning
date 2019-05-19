import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import KFold as kfold
# from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results

def train_predict2(learner, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    learner = learner.fit(X_train, y_train)

    predictions_test = learner.predict(X_test)
    
    predictions_train = learner.predict(X_train)
    results['acc_train'] = accuracy_score(y_train, predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train, predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    print("{} trained on {} samples.".format(learner.__class__.__name__, len(X_train)))

    return results
data = pd.read_csv("agaricus-lepiota.csv")

print(len(data))

for index, row in data.iterrows():
    for name, values in data.iteritems():
        if row[name] == '?':
            data = data.drop(index)

print(len(data))

target = 'class' # The class we want to predict
labels = data[target]

features = data.drop(target, axis=1) # Remove the target class from the dataset

categorical = features.columns # Since every fearure is categorical we use features.columns

features = pd.concat([features, pd.get_dummies(features[categorical])], axis=1) # Convert every categorical feature with one hot encoding

features.drop(categorical, axis=1, inplace=True) # Drop the original feature, leave only the encoded ones

labels = pd.get_dummies(labels)['p'] # Encode the target class, 1 is deadly 0 is safe

#X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2, random_state=0)

clf_A = GaussianNB()
clf_C = RandomForestClassifier(n_estimators=3)
clf_B = KNeighborsClassifier()

kf = kfold(n_splits=10,shuffle=True)
xset = np.array(features)
yset= np.array(labels)
print ("Atskirta")
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    i=0
    for train_ind, test_ind in kf.split(features, labels):
        
        X_train, X_test = xset[train_ind], xset[test_ind]
        y_train, y_test = yset[train_ind], yset[test_ind]
        
        #print("Train")
        #print(len(X_train))
        #print("Test")
        #print(len(X_test))
        results[clf_name][i] = train_predict2(clf,X_train, y_train, X_test, y_test)
        print(results[clf_name][i])
        i=i+1
        


#training_length = len(X_train)
#samples_1 = int(training_length * 0.01)
#samples_10 = int(training_length * 0.1)
#samples_100 = int(training_length * 1)

#results = {}
#for clf in [clf_A, clf_B, clf_C]:
#    clf_name = clf.__class__.__name__
#    results[clf_name] = {}
#    for i, samples in enumerate([samples_1, samples_10, samples_100]):
#        results[clf_name][i] = \
#        train_predict(clf, samples, X_train, y_train, X_test, y_test)
#        print(results[clf_name][i])