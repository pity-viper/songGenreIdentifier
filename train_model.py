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


# features from the csv file that we need to train the ai on
# all the other data is just white noise
features = ['chroma_stft_mean', 'chroma_stft_dev', 'spectral_centroid_mean', 'spectral_centroid_dev',
            'spectral_rolloff_mean', 'spectral_rolloff_dev', 'mfcc_mean', 'mfcc_dev', 'rms_mean', 'rms_dev',
            'zero_crossing_rate_mean', 'zero_crossing_rate_dev']


def predict_song_genre(song_file_path, model_file_path, genre="predict", testing=False):
    """
    Predict the genre of a given song, or validate that the model correctly predicts the genre of a song
    Args:
        song_file_path (str): Path to the song file on the system
        model_file_path (str): Path to the model .pkl file on the system
        genre (str): Optional parameter for the genre you expect the audio file to be.
        testing (bool): Optional parameter for if you are testing the functionality of the model

    Returns:
        tuple: (Song name and expected genre, (predicted genre, count of each genre prediction))
    """

    num_to_genre = {
        0: 'rock',
        1: 'hiphop',
        2: 'pop',
        3: 'country'
    }

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

    test_data = []
    model = joblib.load(model_file_path)
    test_data.extend(get_feature_vectors(song_file_path, audio_genre=genre))
    test = pd.DataFrame.from_records(test_data, columns=headers)
    X_test = test.loc[:, features]

    prediction = model.predict(X_test)
    counts = pd.DataFrame(prediction).value_counts()
    prediction_mode = pd.DataFrame(prediction).mode()
    if not testing:
        return f"{num_to_genre[prediction_mode.iloc[0][0]]}"
    else:
        return f"({genre}) {os.path.basename(song_file_path)}", (num_to_genre[prediction_mode.iloc[0][0]], counts)


if __name__ == "__main__":
    # Import the csv file
    df = pd.read_csv('song_features_4genre_v2.csv', index_col='file_name')

    # Map genres to numerical values
    df['genre'] = df.genre.map({'rock': 0, 'hiphop': 1, 'pop': 2, 'country': 3})

    # create X and Y
    # X is a separate file with just the train data
    # Y is a separate file with the corresponding target values
    X = df.loc[:, features]
    y = df['genre']

    # Split the dataset into testing and training data using train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # Create and train the model
    clf = DecisionTreeClassifier(max_depth=50, class_weight="balanced")
    clf.fit(X, y)  # _train, y_train)

    # Load our testing songs
    rock_test_files = load_files("./test_songs/rock/")
    hiphop_test_files = load_files("./test_songs/hiphop/")
    pop_test_files = load_files("./test_songs/pop/")
    country_test_files = load_files("./test_songs/country/")

    # Calculate predictions on test songs
    results = {}
    for path in rock_test_files:
        k, v = predict_song_genre(path, "rock")
        results[k] = v

    for path in hiphop_test_files:
        k, v = predict_song_genre(path, "hiphop")
        results[k] = v

    for path in pop_test_files:
        k, v = predict_song_genre(path, "pop")
        results[k] = v

    for path in country_test_files:
        k, v = predict_song_genre(path, "country")
        results[k] = v
    
    scores = {
        "rock": 0,
        "hiphop": 0,
        "pop": 0,
        "country": 0
    }

    # display results of predictions on test songs
    for k, v in results.items():
        print(f"{k} : {v}")
        if "rock" in k and "rock" in v[0]:
            scores["rock"] = scores["rock"] + 1
        elif "hiphop" in k and "hiphop" in v[0]:
            scores["hiphop"] = scores["hiphop"] + 1
        elif "pop" in k and "pop" in v[0]:
            scores["pop"] = scores["pop"] + 1
        elif "country" in k and "country" in v[0]:
            scores["country"] = scores["country"] + 1

    print(f"rock: {scores['rock']}/{len(rock_test_files)}\nhiphop: {scores['hiphop']}/{len(hiphop_test_files)}"
          f"\npop: {scores['pop']}/{len(pop_test_files)}\ncountry: {scores['country']}/{len(country_test_files)}")

    # print(clf.score(X_test, y_test))

    plt.figure(figsize=(9, 9), dpi=300)
    tree.plot_tree(clf,
                   feature_names=features,
                   class_names=['rock', 'hiphop', 'pop', 'country'],
                   filled=True)
    plt.show()
    print(clf.get_depth())
    print(clf.feature_importances_)
    print(sum(clf.feature_importances_))

    #joblib.dump(clf, 'TEST1.pkl')
    #clf_load = joblib.load('TEST1.pkl')
