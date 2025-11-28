import unittest
from scripts.encrypt_data import encrypt, KEY_FILES, get_iv
import pandas as pd
from glob import glob
from datasets import load_dataset
import os
import random

class TestMathUtils(unittest.TestCase):
    def __assert_mapping_encrypted_subset(self, encrypted_sentences, sentences, key, iv):
        for i, e in enumerate(encrypted_sentences):
            expected = encrypt(sentences[i], key=key, iv=iv)
            self.assertEqual(e, expected)

    def test_encrypt_e2e(self):
        P_TEST_SHARD = "./data/plain_text/wikipedia/shard_0.csv"
        C_TEST_SHARD = "./data/encrypted/wikipedia/128-bytes/shard_0.csv"

        p_set = pd.read_csv(P_TEST_SHARD)
        c_set = pd.read_csv(C_TEST_SHARD)

        key = KEY_FILES[0]
        iv = get_iv(key)

        sentences = p_set["text"].to_list()
        encrypted_sentences = c_set["text"].to_list()
        self.__assert_mapping_encrypted_subset(encrypted_sentences, sentences, key, iv)


    def test_loading_sets(self):
        PLAIN_TEXT_DATA_DIR = "./data/plain_text/wikipedia"
        ENCRYPTED_TEXT_DATA_DIR = "./data/encrypted/wikipedia"
        KEY_NAME = "128-bytes"
        key = KEY_FILES[0] # 128-bytes.hex
        iv = get_iv(key)

        c_set_dir = f"{ENCRYPTED_TEXT_DATA_DIR}/{KEY_NAME}"
        c_set_files = sorted(glob(os.path.join(c_set_dir, "**")))
        c_set = load_dataset(
            "csv", data_files=c_set_files, split="train",
            download_mode="force_redownload", verification_mode="no_checks"
        )

        p_set_files = sorted(glob(os.path.join(PLAIN_TEXT_DATA_DIR, "**")))
        p_set = load_dataset(
            "csv", data_files=p_set_files, split="train",
            download_mode="force_redownload", verification_mode="no_checks"
        )

        p_set_len, c_set_len = len(p_set), len(c_set)
        self.assertEqual(
            p_set_len, c_set_len, 
            "plain text set and ciphered text set have different sizes"
        )

        sample_size = 500
        random_sample = random.sample(range(p_set_len), sample_size)

        p_subset = p_set.select(random_sample) # type: ignore
        c_subset = c_set.select(random_sample) # type: ignore
        self.__assert_mapping_encrypted_subset(
            c_subset["text"], p_subset["text"], 
            key, iv
        )

if __name__ == "__main__":
    unittest.main()
