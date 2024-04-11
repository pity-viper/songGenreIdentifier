import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from sklearn.metrics import classification_report
from IPython.display import Image
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import joblib
from scipy.stats import uniform, poisson

# Figure out what to import the csv file
df = pd.read_csv('song_features_3genre_v4.csv', index_col="file_name") #, index_col='file_name'

#features from the csv file that we need to train the ai on
#all the other data is just white noise
features = ['chroma_stft_mean', 'chroma_stft_dev', 'spectral_centroid_mean', 'spectral_centroid_dev',
            'spectral_rolloff_mean', 'spectral_rolloff_dev', 'mfcc_mean', 'mfcc_var', 'rms_mean', 'rms_dev',
            'zero_crossing_rate_mean', 'zero_crossing_rate_dev']

#the genre in the dataset is a string and the ai cant run on a string, so you change it to a number here
df['genre'] = df.genre.map({'rock': 0, 'hiphop': 1, 'pop': 2})


#create x and y
#x is a seperate file with just the train data
#y is a seperate file with the cooresponding target values
X = df.loc[:, features]
y = df['genre']

#split the dataset into testing and training data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

parameters = {
    "max_depth": poisson(mu=2, loc=2),
    "max_leaf_nodes": poisson(mu=5, loc=5),
    "min_samples_split": uniform(),
    "min_samples_leaf": uniform()
}

rsearch = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), parameters, cv=5, random_state=0, refit=True)
rsearch.fit(X_train, y_train)
"""
#create an instance of the model
clf = DecisionTreeClassifier(max_depth = 50)

#train that biotch
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
"""
cls_params = rsearch.best_params_
#print(rsearch)
print(f"best parameters: {cls_params}")
print(X_train.shape[0])
cls_params["min_samples_split"] = np.ceil(cls_params["min_samples_split"]*X_train.shape[0])
cls_params['min_samples_leaf'] = np.ceil(cls_params['min_samples_leaf']*X_train.shape[0])
print(f"best parameters: {cls_params}")
clf = rsearch.best_estimator_
plt.figure(figsize=(9, 9), dpi=300)
tree.plot_tree(clf,
               feature_names=features,
               class_names=['rock', 'hiphop', 'pop'],
               filled=True)
#plt.show()
y_pred = clf.predict(X_test)
#print(y_pred)
print(classification_report(y_test, y_pred))
print(clf.get_depth())

'''
joblib.dump(clf, 'genre_identifier_model.pkl')
clf_load = joblib.load('genre_identifier_model.pkl')
print(clf_load.score(X_test, y_test))'''
