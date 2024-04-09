import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Figure out what to import the csv file
df = pd.read_csv('song_features_3genre_v4.5.csv', index_col='file_name')

#features from the csv file that we need to train the ai on
#all the other data is just white noise
features = ['spectral_centroid_mean', 'spectral_centroid_dev','rms_mean', 'rms_dev']

#the genre in the dataset is a string and the ai cant run on a string, so you change it to a number here
df['genre'] = df.genre.map({'rock': 0, 'hiphop': 1, 'pop': 2, 'country': 3})


#create x and y
#x is a seperate file with just the train data
#y is a seperate file with the cooresponding target values
X = df.loc[:, features]
y = df['genre']

#split the dataset into testing and training data using train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

test_data = []
headers = [
        "file_name",
        "genre",
        "chroma_stft_mean",
        "chroma_stft_dev",
        "spectral_centroid_mean",
        "spectral_centroid_dev",
        "spectral_rolloff_mean",
        "spectral_rolloff_dev",
        "mfcc_mean",
        "mfcc_var",
        "rms_mean",
        "rms_dev",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_dev"
]

from features_to_csv import get_feature_vectors
test_data.extend(get_feature_vectors('dataset/back in black.mp3', audio_genre='rock'))
test = pd.DataFrame.from_records(test_data, columns=headers)
X_test = test.loc[:, features]

#create an instance of the model
clf = DecisionTreeClassifier(max_depth = 50)

#train that biotch
clf.fit(X, y)   #_train, y_train)

prediction = clf.predict(X_test)
print(prediction)
prediction_mode = pd.DataFrame(prediction).mode()
print(prediction_mode.iloc[0][0])
dic = {
        0: 'rock',
        1: 'hiphop',
        2: 'pop'
}

#print(clf.score(X_test, y_test))

'''plt.figure(figsize=(9, 9), dpi=300)
tree.plot_tree(clf,
               feature_names=features,
               class_names=['rock', 'hiphop', 'pop'],
               filled=True)
#plt.show()'''
print(clf.get_depth())
'''
joblib.dump(clf, 'genre_identifier_model.pkl')
clf_load = joblib.load('genre_identifier_model.pkl')
print(clf_load.score(X_test, y_test))'''
