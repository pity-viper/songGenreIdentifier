import os.path
from glob import glob
import numpy as np
import librosa
from librosa import feature
import csv

# Load the normal audio files
norm_rock_dir = "./dataset/rock/"
norm_hiphop_dir = "./dataset/hiphop/"
norm_pop_dir = "./dataset/pop/"
norm_rock_files = glob(norm_rock_dir + "*.wav")
norm_hiphop_files = glob(norm_hiphop_dir + "*.wav")
norm_pop_files = glob(norm_pop_dir + "*.wav")
#audio_files = [*norm_rock_files, *norm_rock_files, *norm_pop_files]
#rock_genre_list = ["rock" for i in range(len(norm_rock_files))]
#hiphop_genre_list = ["hiphop" for i in range(len(norm_rock_files))]

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
    """
    Extract audio features from files, then compute their mean and standard deviation
    Args:
        ts (np.ndarray): A time series array of the audio file computed by Librosa
        sr (int): The sample rate of the audio file in Hz

    Returns:
        list: A list of the means and standard deviations for each extracted feature
    """
    # Call feature extraction functions on all the files
    # To Do call the split_audio_file function here so I know the file name in this function, then put filename at beginning of feature vector so each audio segment also has the file name
    feat_i = [func(y=ts, sr=sr) for func in fn_list_i]
    feat_ii = [func(y=ts) for func in fn_list_ii]
    # Compute the mean and standard deviation for the features of all files
    feat_vector_i = [(np.mean(x), np.std(x)) for x in feat_i]
    feat_vector_ii = [(np.mean(x), np.std(x)) for x in feat_ii]
    feat_vector_i = [item for tup in feat_vector_i for item in tup]
    feat_vector_ii = [item for t in feat_vector_ii for item in t]
    # Combine and return the two different feature sets
    feature_vector = feat_vector_i + feat_vector_ii
    return feature_vector


def split_audio_file(audio_file, sec_len):
    """
    Splits an audio file into segments of specified length
    Args:
        audio_file (str): String pathname of .wav audio file
        sec_len (int): number of seconds long the audio segment should be

    Returns:
        A list of the audio sections and the sample rate of the audio in tuple format (audio_sections, sample_rate)
    """
    time_series, sample_rate = librosa.load(audio_file)
    sec_samples = sec_len * sample_rate
    total_sections = int(np.ceil(len(time_series)/sec_samples))
    audio_sections = []
    for i in range(total_sections):
        start_sample = i * sec_samples
        end_sample = min((i + 1) * sec_samples, len(time_series))
        section = time_series[start_sample:end_sample]
        audio_sections.append(section)
    if audio_sections[-1].size < sample_rate:
        audio_sections.pop(-1)
    return audio_sections, sample_rate


# Feature extraction
norm_rock_feat = []
norm_hiphop_feat = []
norm_pop_feat = []
#song_feat = []

for file in norm_rock_files:
    #ts, sr = librosa.load(file)
    #feature_vector = get_feature_vector(ts, sr)
    #norm_rock_feat.append(feature_vector)
    audio_sections, sr = split_audio_file(file, 3)
    for ts in audio_sections:
        feature_vector = get_feature_vector(ts, sr)
        norm_rock_feat.append(feature_vector)
for file in norm_hiphop_files:
    #ts, sr = librosa.load(file)
    #feature_vector = get_feature_vector(ts, sr)
    #norm_hiphop_feat.append(feature_vector)
    audio_sections, sr = split_audio_file(file, 3)
    for ts in audio_sections:
        feature_vector = get_feature_vector(ts, sr)
        norm_hiphop_feat.append(feature_vector)
for file in norm_pop_files:
    audio_sections, sr = split_audio_file(file, 3)
    for ts in audio_sections:
        feature_vector = get_feature_vector(ts, sr)
        norm_pop_feat.append(feature_vector)
"""
for file in audio_files:
    audio_sections, sr = split_audio_file(file, 3)
    for ts in audio_sections:
        feature_vector = get_feature_vector(ts, sr)
        song_feat.append(feature_vector)
"""
# Define CSV file needed info
outfile = "song_features_more_3genre.csv"
headers = [
    "genre",
    "chroma_stft_mean",
    "chroma_stft_dev",
    "spectral_centroid_mean",
    "spectral_centroid_dev",
    "spectral_rolloff_mean",
    "spectral_rolloff_dev",
    "rms_mean",
    "rms_dev",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_dev"
]

# Format arrays in way that CSV library likes
rock_genre_list = ["rock" for i in range(np.shape(np.array(norm_rock_feat))[0])]
hiphop_genre_list = ["hiphop" for i in range(np.shape(np.array(norm_hiphop_feat))[0])]
pop_genre_list = ["pop" for i in range(np.shape(np.array(norm_pop_feat))[0])]
rock_genre_list = np.array(rock_genre_list)
hiphop_genre_list = np.array(hiphop_genre_list)
pop_genre_list = np.array(pop_genre_list)
norm_rock_feat = np.hstack((rock_genre_list.reshape(-1, 1), norm_rock_feat))
norm_hiphop_feat = np.hstack((hiphop_genre_list.reshape(-1, 1), norm_hiphop_feat))
norm_pop_feat = np.hstack((pop_genre_list.reshape(-1, 1), norm_pop_feat))
#rock_genre_list = ["rock" for i in range(len(norm_rock_files))]
#hiphop_genre_list = ["hiphop" for i in range(len(norm_hiphop_files))]
#pop_genre_list = ["pop" for i in range(len(norm_pop_files))]
#genre_list = [*rock_genre_list, *hiphop_genre_list, *pop_genre_list]
#norm_rock_files = list(map(os.path.basename, norm_rock_files))
#norm_rock_files = np.array(norm_rock_files)
#norm_hiphop_files = list(map(os.path.basename, norm_hiphop_files))
#genre_list = np.array(genre_list)
#norm_hiphop_files = np.array(norm_hiphop_files)
#norm_rock_feat = np.hstack((norm_rock_files.reshape(-1, 1), norm_rock_feat))
#norm_hiphop_feat = np.hstack((norm_hiphop_files.reshape(-1, 1), norm_hiphop_feat))
#norm_feat = np.hstack((genre_list.reshape(-1, 1), song_feat))

# Export to CSV File
with open(outfile, "w", newline="") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(headers)
    csv_writer.writerows(norm_rock_feat)
    csv_writer.writerows(norm_hiphop_feat)
    csv_writer.writerows(norm_pop_feat)
    #csv_writer.writerows(song_feat)
