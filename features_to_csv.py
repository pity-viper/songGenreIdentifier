import os.path
import numpy as np
import librosa
from librosa import feature
import csv
import re
import itertools
from pathlib import Path


def load_files(path):
    """
    Read all filenames in directory and all subdirectories of path
    Args:
        path (str): Path of directory containing the files. Absolute path or relative. Ex: ("./path/to/your/directory/")

    Returns:
        list: A list of the paths to all the files in the given directory
    """
    return list(filter(lambda x: x.suffix in file_extensions, Path(path).glob("**/*")))


# Define the functions to extract the features
fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_rolloff,
    feature.mfcc
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]

file_extensions = [
    ".wav",
    ".mp3"
]


def get_feature_vectors(audio_file, audio_genre=None):
    """
    Extract audio features from files, then compute their mean and standard deviation
    Args:
        audio_file (str): The path to the audio file
        audio_genre (str): Optional parameter for the genre of audio_file. If unknown, pass 'predict'

    Returns:
        list: The filename, genre, and the means and standard deviations for each extracted feature
    """
    file_pattern = f"({'|'.join(file_extensions)})"
    feature_vectors = []
    file_name = os.path.basename(audio_file)
    if audio_genre:
        genre = audio_genre
    else:
        genre = get_genre(file_name)
    # Split the audio file in segments
    sections, sr = split_audio_file(audio_file, 1)
    # Call feature extraction functions for each segment
    for i in range(len(sections)):
        identifiers = [re.sub(file_pattern, f".clip{i}\\1", file_name), genre]
        # Compute audio features
        feat_i = [func(y=sections[i], sr=sr) for func in fn_list_i]
        feat_ii = [func(y=sections[i]) for func in fn_list_ii]
        # Compute the mean and standard deviation of the features
        feat_vector_i = [(np.mean(x), np.std(x)) for x in feat_i]
        feat_vector_ii = [(np.mean(x), np.std(x)) for x in feat_ii]
        # Combine and return the two different feature sets
        feat_vector = list(itertools.chain(identifiers, *feat_vector_i, *feat_vector_ii))
        feature_vectors.append(feat_vector)
    return feature_vectors


def get_genre(file_name):
    """
    Get the genre of the given file name
    Args:
        file_name (str): The basename of the file

    Returns:
        str: The genre of the given file_name

    Raises:
        ValueError: If file_name does not contain 'rock', 'hiphop', 'pop', or 'country'
    """
    if "rock" in file_name:
        return "rock"
    elif "hiphop" in file_name:
        return "hiphop"
    elif "pop" in file_name:
        return "pop"
    elif "country" in file_name:
        return "country"
    else:
        raise ValueError(
            f"Expected a filename containing 'rock', 'hiphop', 'pop', or 'country'. Instead got '{file_name}'"
        )


def split_audio_file(audio_file, sec_len):
    """
    Splits an audio file into segments of specified length
    Args:
        audio_file (str): String pathname of .wav audio file
        sec_len (float): number of seconds long the audio segment should be

    Returns:
        tuple: Segments of audio_file and the sample rate of audio_file in tuple format (audio_sections, sample_rate)
    """
    time_series, sample_rate = librosa.load(audio_file)
    # Determine the number of sections in audio_file of length sec_len
    sec_samples = sec_len * sample_rate
    total_sections = int(np.ceil(len(time_series)/sec_samples))
    audio_sections = []
    # Split into sections
    for i in range(total_sections):
        start_sample = i * sec_samples
        end_sample = min((i + 1) * sec_samples, len(time_series))
        section = time_series[start_sample:end_sample]
        audio_sections.append(section)
    # If the last section is smaller than the rest, remove it
    if audio_sections[-1].size < sample_rate:
        audio_sections.pop(-1)
    return audio_sections, sample_rate


if __name__ == "__main__":
    # Load the audio files
    rock_files = load_files("./dataset/rock/")
    hiphop_files = load_files("./dataset/hiphop/")
    #country_files = load_files("./dataset/country/")
    pop_files = load_files("./dataset/pop/")

    # Extract features for all files
    song_feat = []
    for file in rock_files:
        feature_vector = get_feature_vectors(file, audio_genre="rock")
        song_feat.extend(feature_vector)

    for file in hiphop_files:
        feature_vector = get_feature_vectors(file, audio_genre="hiphop")
        song_feat.extend(feature_vector)
    '''
    for file in country_files:
        feature_vector = get_feature_vectors(file, audio_genre="country")
        song_feat.extend(feature_vector)
    '''
    for file in pop_files:
        feature_vector = get_feature_vectors(file, audio_genre="pop")
        song_feat.extend(feature_vector)

    # Define CSV file needed info
    outfile = "song_features_4genre_v4.csv"
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

    # Export to CSV File
    with open(outfile, "w", newline="") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(headers)
        csv_writer.writerows(song_feat)
