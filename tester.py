import unittest
from features_to_csv import load_files, get_genre


class TestGenreIdentifier(unittest.TestCase):
    def setUp(self):
        self.test_file = str(load_files("./test_songs/test_case_data/")[0].resolve())

    def test_load_files(self):
        self.assertIn("songGenreIdentifier/test_songs/test_case_data/rock.00000.wav", self.test_file,
                      msg="load_files() is not working correctly")

    def test_get_genre(self):
        self.assertEqual(get_genre(self.test_file), "rock",
                         msg="get_genre() failed to extract genre from filename")
        with self.assertRaises(ValueError, msg="get_genre() failed to catch an incorrect filename"):
            get_genre("not_a_valid_filename")


if __name__ == '__main__':
    unittest.main()
