import os.path
from glob import glob
import numpy as np
import librosa
from librosa import feature
import csv

# Load the normal audio files
norm_rock_dir = "./dataset/rock/"
norm_hiphop_dir = "./dataset/hiphop/"
norm_rock_files = glob(norm_rock_dir + "*.wav")
#norm_rock_files = list(map(os.path.basename, norm_rock_files))
norm_hiphop_files = glob(norm_hiphop_dir + "*.wav")
#norm_hiphop_files = list(map(os.path.basename, norm_hiphop_files))
rock_genre_list = ["rock" for i in range(len(norm_rock_files))]
hiphop_genre_list = ["hiphop" for i in range(len(norm_rock_files))]

# Define the functions to extract the features
fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_rolloff
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]


def get_feature_vector(ts, sr):
    feat_i = [func(y=ts, sr=sr) for func in fn_list_i]
    feat_ii = [func(y=ts) for func in fn_list_ii]
    feat_vector_i = [(np.mean(x), np.std(x)) for x in feat_i]
    feat_vector_ii = [(np.mean(x), np.std(x)) for x in feat_ii]
    feat_vector_i = [item for tup in feat_vector_i for item in tup]
    feat_vector_ii = [item for t in feat_vector_ii for item in t]
    feature_vector = feat_vector_i + feat_vector_ii
    return feature_vector

# Feature extraction
norm_rock_feat = []
norm_hiphop_feat = []
for file in norm_rock_files:
    ts, sr = librosa.load(file)
    feature_vector = get_feature_vector(ts, sr)
    norm_rock_feat.append(feature_vector)
for file in norm_hiphop_files:
    ts, sr = librosa.load(file)
    feature_vector = get_feature_vector(ts, sr)
    norm_hiphop_feat.append(feature_vector)

# Export to CSV File
outfile = "song_features.csv"
header = [
    "file_name",
    "genre",
    "chroma_stft_mean",
    "chroma_stft_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_rolloff_mean",
    "spectral_rolloff_var",
    "rms_mean",
    "rms_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var"
]
norm_rock_files = list(map(os.path.basename, norm_rock_files))
rock_genre_list = np.array(rock_genre_list)
norm_rock_files = np.array(norm_rock_files)
norm_hiphop_files = list(map(os.path.basename, norm_hiphop_files))
hiphop_genre_list = np.array(hiphop_genre_list)
norm_hiphop_files = np.array(norm_hiphop_files)
norm_rock_feat = np.hstack((rock_genre_list.reshape(-1, 1), norm_rock_feat))
norm_rock_feat = np.hstack((norm_rock_files.reshape(-1, 1), norm_rock_feat))
norm_hiphop_feat = np.hstack((hiphop_genre_list.reshape(-1, 1), norm_hiphop_feat))
norm_hiphop_feat = np.hstack((norm_hiphop_files.reshape(-1, 1), norm_hiphop_feat))
#print(norm_rock_feat)
#print(norm_hiphop_feat)

with open(outfile, "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(header)
    csv_writer.writerows(norm_rock_feat)
    csv_writer.writerows(norm_hiphop_feat)
