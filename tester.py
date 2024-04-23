import unittest
import glob
from pathlib import Path
from features_to_csv import load_files, get_genre, get_feature_vectors, split_audio_file
from train_model import predict_song_genre


class TestGenreIdentifier(unittest.TestCase):
    def setUp(self):
        #self.test_file = Path("./test_songs/test_case_data/rock.00000.wav").resolve().__str__()
        self.test_file = load_files("./test_songs/test_case_data/")[0].resolve().__str__()
        self.feature_vector = get_feature_vectors(self.test_file)
        self.model_file = glob.glob("TEST1.pkl")[0]

    def test_load_files(self):
        test_path = Path("./test_songs/test_case_data/rock.00000.wav").resolve().__str__()
        self.assertEqual(test_path, self.test_file, msg="load_files() is not working correctly")

    def test_get_genre(self):
        self.assertEqual(get_genre(self.test_file), "rock",
                         msg="get_genre() failed to extract genre from filename")
        with self.assertRaises(ValueError, msg="get_genre() failed to catch an incorrect filename"):
            get_genre("not_a_valid_filename")

    def test_get_feature_vectors(self):
        test_vector = get_feature_vectors(self.test_file)
        for i in range(len(test_vector)):
            mean_dif = abs(test_vector[i][2] - self.feature_vector[i][2])/test_vector[i][2]
            stddev_dif = abs(test_vector[i][3] - self.feature_vector[i][3])/test_vector[i][3]
            self.assertTrue(len(test_vector[i]) == 14, msg="get_feature_vectors() returned an incorrect number"
                                                           "of features")
            self.assertTrue(mean_dif <= (0.05 * self.feature_vector[i][2]), msg="get_feature_vectors() mean values"
                                                                                "vary by more than 5% between runs")
            self.assertTrue(stddev_dif <= (0.05 * self.feature_vector[i][3]), msg="get_feature_vectors() standard"
                                                                                  "deviation values vary by more"
                                                                                  "than 5% between runs")

    def test_split_audio_file(self):
        sections, sr = split_audio_file(self.test_file, 1)
        self.assertTrue(len(sections) == 30, msg="split_audio_file() does not generate the correct number of sections")
        self.assertTrue(sr == 22050, msg="split_audio_file() does not return the correct sample rate")

    def test_predict_song_genre(self):
        test_genre = predict_song_genre(self.test_file, self.model_file)
        self.assertEqual(test_genre, "rock", msg="predict_song_genre() cannot predict genre correctly")


if __name__ == '__main__':
    unittest.main()
