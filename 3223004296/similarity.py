import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    """
    预处理文本：转换为小写，去除所有标点符号，并去掉首尾空白。
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # 只保留字母、数字、汉字
    return text


def tokenizer(text):
    """
    使用 jieba 分词，对预处理后的文本进行分词。
    """
    return list(jieba.cut(preprocess(text)))


def tfidf_cosine_similarity(text1, text2):
    """
    使用 TF-IDF 计算两个文本的余弦相似度，优化短文本计算，增强鲁棒性。
    """
    # 若任一文本为空，返回 0.0
    if not text1.strip() or not text2.strip():
        return 0.0

    # 预处理文本
    text1_clean = preprocess(text1)
    text2_clean = preprocess(text2)

    # 若预处理后文本完全一致（忽略顺序），认为相似度为 1.0
    if text1_clean == text2_clean:
        return 1.0

    # 初始化 TF-IDF 向量器
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        lowercase=False,
        min_df=1,
        max_df=1.0,
        norm='l2',
        smooth_idf=True,  # IDF 平滑处理，避免极端值
    )

    # 计算 TF-IDF 矩阵
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 处理短文本相似度问题
    if len(text1_clean) < 5 or len(text2_clean) < 5:
        # 额外考虑 Jaccard 相似度，适用于短文本
        set1, set2 = set(tokenizer(text1)), set(tokenizer(text2))
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
        sim = (sim + jaccard_sim) / 2  # 结合 TF-IDF 和 Jaccard

    return round(sim, 2)

