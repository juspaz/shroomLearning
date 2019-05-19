import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_predict_userInput(clf1, clf2, clf3, X_train, y_train, categorical, categorical2, sample_size = 5644):
    results = {}    

    clf1 = clf1.fit(X_train[:sample_size], y_train[:sample_size])
    clf2 = clf2.fit(X_train[:sample_size], y_train[:sample_size])
    clf3 = clf3.fit(X_train[:sample_size], y_train[:sample_size])
    clfs = [clf1, clf2, clf3]

    while True:
        features = {}
        inpt = []
        print('\nIveskite grybo atributus atskirtus tarpo simboliu:' + ' '*15 + '[arba] exit -> baigti darba')
        inpt = input(">").split()
        if len(inpt) < 1:
            continue
        elif inpt[0] == "exit":
            break
        for idx, val in enumerate(inpt):
            temp = []
            temp.append(val)
            features[categorical[idx]] = temp #creates dict of input values as lists
        df = pd.DataFrame(data=features) #creates dF from dict
        df = pd.concat([df, pd.get_dummies(df[categorical])], axis=1) #encoding
        df.drop(categorical, axis=1, inplace=True) #cleaning non numeric values and categ. col.
        #adding the rest of columns
        for newCell in categorical2:
            if newCell in df.columns:
                continue            
            df[newCell] = 0
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(df)
        #print(df)
        df = df.reindex(columns=categorical2)
        hardVote, softVote = 0, 0.0
        for c in clfs:       
            res = c.predict_proba(df)[:,1]
            prob = res[0].item()
            print ('{:25} | Tikimybe jog grybas nuodingas: {:.6f}'.format(c.__class__.__name__, prob))
            pred = c.predict(df)
            hardVote += pred
            softVote += prob
        softVote = softVote / 3
        if hardVote>=2:
            print('Hard vote prognoze: [nuodingas]')
        else:
            print('Hard vote prognoze: [valgomas]')
        if softVote >= 0.5:
            print('Soft vote prognoze: [nuodingas]')
        else:
            print('Soft vote prognoze: [valgomas]')
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

categorical2 = features.columns
#print('{}  {}'.format(categorical2, len(categorical2)))

labels = pd.get_dummies(labels)['p'] # Encode the target class, 1 is deadly 0 is safe

clf_A = GaussianNB()
clf_B = RandomForestClassifier(n_estimators=100)
clf_C = KNeighborsClassifier()

'''
Different sets of attributes for testing -> train_predict_userInput(*)
f y g t n f c b w e b s s w w p w t p w y p   EDIBLE
x y n f n f c b w e b y y n n p w t p w y d   EDIBLE
f y c f m a c b w e c k y c c p w n n w c d   POISONOUS
'''
train_predict_userInput(clf_A, clf_B, clf_C, features, labels, categorical, categorical2)