import os.path
from glob import glob
import numpy as np
import librosa
from librosa import feature
import csv
import re


def load_files(path, extension):
    """
    Read all filenames in directory at path ending with extension
    Args:
        path (str): Path of directory containing the files. Absolute path or relative. Ex: ("./path/to/your/directory/")
        extension (str): The file extension to load. Do NOT include a dot. Ex: ("wav")

    Returns:
        list: A list of the paths to all the files in the given directory
    """
    return sorted(glob(f"{path}*.{extension}"))


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
    ".mp3",
    ".flac"
]

pattern = f"({'|'.join(file_extensions)})"


def get_feature_vectors(audio_file, audio_genre=None):
    """
    Extract audio features from files, then compute their mean and standard deviation
    Args:
        audio_file (str): The path to the audio file
        audio_genre (str): Optional parameter for the genre of audio_file

    Returns:
        list: The filename, genre, and the means and standard deviations for each extracted feature
    """
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
        identifiers = [re.sub(pattern, f".clip{i}\\1", file_name), genre]
        feat_i = [func(y=sections[i], sr=sr) for func in fn_list_i]
        feat_ii = [func(y=sections[i]) for func in fn_list_ii]
        # Compute the mean and standard deviation for the features of all files
        feat_vector_i = [(np.mean(x), np.std(x)) for x in feat_i]
        feat_vector_ii = [(np.mean(x), np.std(x)) for x in feat_ii]
        feat_vector_i = [item for tup in feat_vector_i for item in tup]
        feat_vector_ii = [item for tup in feat_vector_ii for item in tup]
        # Combine and return the two different feature sets
        feat_vector = feat_vector_i + feat_vector_ii
        feat_vector = identifiers + feat_vector
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
        ValueError: If file_name does not contain 'rock', 'hiphop', or 'pop'
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
            f"Expected a filename containing 'rock', 'hiphop', or 'pop', got '{file_name}'"
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


def main():
    norm_rock_files = load_files("./dataset/rock/", "wav")
    norm_hiphop_files = load_files("./dataset/hiphop/", "wav")
    norm_pop_files = load_files("./dataset/pop/", "wav")
    norm_country_files = load_files("./dataset/country/", "wav")
    audio_files = [*norm_rock_files, *norm_hiphop_files, *norm_pop_files, *norm_country_files]
    #audio_files = [*norm_rock_files, *norm_hiphop_files]

    # Feature extraction
    song_feat = []
    for file in audio_files:
        feature_vector = get_feature_vectors(file)
        song_feat.extend(feature_vector)

    # Define CSV file needed info
    outfile = "song_features_4genre.csv"
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

    # Export to CSV File
    with open(outfile, "w", newline="") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(headers)
        csv_writer.writerows(song_feat)


if __name__ == "__main__":
    main()
