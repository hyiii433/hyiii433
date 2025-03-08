import sys
from similarity import cosine_similarity


def read_file(filepath):
    # f读取文件内容并返回文本
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {filepath} 未找到")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <原文文件> <抄袭版文件> <答案文件>")
        sys.exit(1)

# 获取文件路径
    orig_file, plagiarized_file, output_file \
        = sys.argv[1], sys.argv[2], sys.argv[3]

    text1 = read_file(orig_file)
    text2 = read_file(plagiarized_file)

# 计算原文与抄袭文相似度similarity
    similarity = cosine_similarity(text1, text2)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(str(similarity))

    print(f"查重率: {similarity}")
