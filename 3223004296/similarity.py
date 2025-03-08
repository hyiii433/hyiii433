import jieba
import math
from collections import Counter


def cosine_similarity(text1, text2):
    words1 = Counter(jieba.lcut(text1))
    words2 = Counter(jieba.lcut(text2))

    # 去重
    all_words = set(words1.keys()).union(set(words2.keys()))

    vec1 = [words1.get(word, 0) for word in all_words]
    vec2 = [words2.get(word, 0) for word in all_words]

    # 计算点积、两向量模长
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))

    # 计算余弦相似度
    return round(dot_product / (magnitude1 * magnitude2), 2)
