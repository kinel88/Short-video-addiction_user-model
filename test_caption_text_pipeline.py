import unittest
from pathlib import Path

from caption_text_pipeline import build_caption_text_feature_bundle, load_kuairec_caption_category_rows


CAPTION_FILE = Path(__file__).with_name("kuairec_caption_category.csv")


@unittest.skipUnless(CAPTION_FILE.exists(), "kuairec_caption_category.csv is required")
class CaptionTextPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caption_df, cls.parse_meta = load_kuairec_caption_category_rows(str(CAPTION_FILE))
        cls.text_feature, cls.segmented_df, cls.text_meta = build_caption_text_feature_bundle(
            str(CAPTION_FILE),
            text_dim=64,
        )

    def test_load_real_caption_file_recovers_all_video_ids(self):
        self.assertEqual(self.parse_meta["valid_video_rows"], 10728)
        self.assertEqual(int(self.caption_df["video_id"].nunique()), 10728)
        self.assertEqual(self.parse_meta["duplicate_video_rows"], 0)
        self.assertEqual(self.parse_meta["invalid_video_id_rows"], 4)
        self.assertEqual(
            self.parse_meta["valid_video_rows"] + self.parse_meta["invalid_video_id_rows"],
            self.parse_meta["physical_data_lines"],
        )

    def test_segmented_tokens_keep_tags_and_categories(self):
        video4 = self.segmented_df.loc[self.segmented_df["video_id"] == 4].iloc[0]
        caption_tokens = video4["segmented_caption"].split()
        topic_tokens = video4["segmented_topic_tag"].split()
        category_tokens = video4["segmented_category_tokens"].split()
        weighted_tokens = video4["segmented_text_all"].split()

        self.assertIn("\u641e\u7b11", caption_tokens)
        self.assertIn("hash_\u641e\u7b11", caption_tokens)
        self.assertIn("\u4e94\u7231\u5e02\u573a", topic_tokens)
        self.assertIn("tag_\u4e94\u7231\u5e02\u573a", topic_tokens)
        self.assertIn("\u65f6\u5c1a", category_tokens)
        self.assertIn("cat1_\u65f6\u5c1a", category_tokens)
        self.assertGreaterEqual(weighted_tokens.count("\u4e94\u7231\u5e02\u573a"), 3)
        self.assertGreaterEqual(weighted_tokens.count("\u65f6\u5c1a"), 2)

    def test_unquoted_topic_tag_and_tfidf_meta(self):
        video7 = self.segmented_df.loc[self.segmented_df["video_id"] == 7].iloc[0]
        topic_tokens = video7["segmented_topic_tag"].split()

        self.assertIn("\u7075\u9b42\u5c5e\u6027\u5927\u63ed\u79d8", topic_tokens)
        self.assertIn("tag_\u7075\u9b42\u5c5e\u6027\u5927\u63ed\u79d8", topic_tokens)
        self.assertIsNotNone(self.text_feature)
        self.assertEqual(self.text_feature.shape[1], 65)
        self.assertEqual(self.text_meta["text_feature_dim"], 64)
        self.assertEqual(self.text_meta["tfidf"]["tokenizer"], "str.split")
        self.assertIsNone(self.text_meta["tfidf"]["token_pattern"])
        self.assertFalse(self.text_meta["tfidf"]["lowercase"])


if __name__ == "__main__":
    unittest.main()
