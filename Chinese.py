import os
import math
from collections import Counter
import jieba

# 计算字符信息熵
def calculate_char_entropy(text):
    char_count = Counter(text)
    total_chars = len(text)
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_count.values())
    return entropy

# 分块计算词信息熵
def calculate_word_entropy(text, chunk_size=1000000):  # 每次处理100万字符
    word_count = Counter()
    total_words = 0
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        words = jieba.lcut(chunk)
        word_count.update(words)
        total_words += len(words)
        print(f"已处理{min(i + chunk_size, len(text))}/{len(text)}个字符")

    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_count.values())
    return entropy

# 读取语料库文本
def read_corpus(root_dir):
    corpus = ""
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                corpus += file.read()
    return corpus

# 主程序
corpus_root = 'wiki_zh'
text = read_corpus(corpus_root)

print(f"读取到的文本总长度：{len(text)} 个字符")

# 计算字信息熵
char_entropy = calculate_char_entropy(text)
print(f"中文文本的字信息熵：{char_entropy:.4f} bits/char")

# 计算词信息熵（分块处理）
word_entropy = calculate_word_entropy(text)
print(f"中文文本的词信息熵：{word_entropy:.4f} bits/word")