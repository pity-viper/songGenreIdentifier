import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from IPython.display import Image

# Figure out what to import the csv file
df = pd.read_csv('song_features.csv', index_col='file_name')

#features from the csv file that we need to train the ai on
#all the other data is just white noise
features = ['chroma_stft_mean', 'chroma_stft_dev', 'spectral_centroid_mean', 'spectral_centroid_dev',
            'spectral_rolloff_mean', 'spectral_rolloff_dev', 'rms_mean', 'rms_dev', 'zero_crossing_rate_mean',
            'zero_crossing_rate_dev']

#the sex in the data set is a string and the ai cant run on a string, so you change it to a number here
df['genre'] = df.genre.map({'rock': 0, 'hiphop': 1})

''''#the dataset has some values where age is empty this line gets rid of that
df = df.loc[~df['Age'].isnull(), :]'''

#create x and y
#x is a seperate file with just the train data
#y is a seperate file with the cooresponding target values
X = df.loc[:, features]
y = df['genre']

#split the dataset into testing and training data using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

for i in range(len(y_train)):
    print(y_train[i])

#create an instince of the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 10)

#train that biotch
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

'''plt.figure(figsize=(9,9), dpi = 300)
tree.plot_tree(clf,
               feature_names = features,
               class_names=['rock', 'hiphop'],
               filled = True);
plt.show()'''


'''import joblib
joblib.dump(clf, 'titanic_model.pkl')
clf_load = joblib.load('titanic_model.pkl')
print(clf_load.score(X_test, y_test))'''