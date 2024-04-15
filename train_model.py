import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from features_to_csv import get_feature_vectors, load_files

# Figure out what to import the csv file
df = pd.read_csv('song_features_4genre.csv', index_col='file_name')

# features from the csv file that we need to train the ai on
# all the other data is just white noise
features = ['chroma_stft_mean', 'chroma_stft_dev', 'spectral_centroid_mean', 'spectral_centroid_dev',
            'spectral_rolloff_mean', 'spectral_rolloff_dev', 'mfcc_mean', 'mfcc_dev', 'rms_mean', 'rms_dev',
            'zero_crossing_rate_mean', 'zero_crossing_rate_dev']

# the genre in the dataset is a string and the ai cant run on a string, so you change it to a number here
df['genre'] = df.genre.map({'rock': 0, 'hiphop': 1, 'pop': 2, 'country': 3})

# create x and y
# x is a seperate file with just the train data
# y is a seperate file with the cooresponding target values
X = df.loc[:, features]
y = df['genre']

# split the dataset into testing and training data using train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

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
    "mfcc_dev",
    "rms_mean",
    "rms_dev",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_dev"
]

# create an instance of the model
clf = DecisionTreeClassifier(max_depth=50)

# train that biotch
clf.fit(X, y)  # _train, y_train)

rock_test_files = load_files("./test_songs/rock/", "mp3")
hiphop_test_files = load_files("./test_songs/hiphop/", "mp3")

genres = {
    0: 'rock',
    1: 'hiphop',
    2: 'pop',
    3: 'country'
}
results = {}
for path in rock_test_files:
    test_data = []
    test_data.extend(get_feature_vectors(path, audio_genre='rock'))
    test = pd.DataFrame.from_records(test_data, columns=headers)
    X_test = test.loc[:, features]

    prediction = clf.predict(X_test)
    counts = pd.DataFrame(prediction).value_counts()
    #print(prediction)
    prediction_mode = pd.DataFrame(prediction).mode()
    results[f"(rock) {os.path.basename(path)}"] = (genres[prediction_mode.iloc[0][0]], counts)

for path in hiphop_test_files:
    test_data = []
    test_data.extend(get_feature_vectors(path, audio_genre='hiphop'))
    test = pd.DataFrame.from_records(test_data, columns=headers)
    X_test = test.loc[:, features]

    prediction = clf.predict(X_test)
    counts = pd.DataFrame(prediction).value_counts()
    #print(prediction)
    prediction_mode = pd.DataFrame(prediction).mode()
    results[f"(hiphop) {os.path.basename(path)}"] = (genres[prediction_mode.iloc[0][0]], counts)

scores = {
    "rock": 0,
    "hiphop": 0,
}
"""
for k, v in results.items():
    print(type(v))
    print(v[0])
"""
for k, v in results.items():
    print(f"{k} : {v}")
    if "rock" in k and "rock" in v[0]:
        scores["rock"] = scores["rock"] + 1
    elif "hiphop" in k and "hiphop" in v[0]:
        scores["hiphop"] = scores["rock"] + 1

print(f"rock: {scores['rock']}/10\nhiphop: {scores['hiphop']}/10")

# print(clf.score(X_test, y_test))

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
