import unittest
from similarity import tfidf_cosine_similarity

class TestCosineSimilarity(unittest.TestCase):
    def test_exact_match(self):
        """测试完全相同的文本"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好", "今天天气好"), 1.0, places=2)

    def test_partial_match(self):
        """测试部分相同的文本"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好", "今天好"), 0.58, places=2)

    def test_no_match(self):
        """测试完全不同的文本"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好", "明天有雨"), 0.0, places=2)


    def test_similar_meaning_different_words(self):
        """测试含义相似但用词不同的文本"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天是星期天，天气晴，今天晚上我要去看电影。",
                                                 "今天是周天，天气晴朗，我晚上要去看电影。"), 0.87, places=2)

    def test_different_word_orders(self):
        """测试相同词不同顺序的文本"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好，我想去公园散步。",
                                                 "我想去公园散步，今天天气好。"), 1.0, places=2)


    def test_punctuation(self):
        """测试标点符号的影响"""
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好。", "今天天气好"), 1.0, places=2)
        self.assertAlmostEqual(tfidf_cosine_similarity("今天天气好！", "今天天气好"), 1.0, places=2)

if __name__ == '__main__':
    unittest.main()
