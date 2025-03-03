import nltk
from nltk.corpus import gutenberg
from collections import Counter
import math

# 下载语料库（首次使用时需要）
nltk.download('gutenberg')

# 查看所有可用的文本
print("Available texts in Gutenberg Corpus:")
print(gutenberg.fileids())

# 选择一个文本（比如《Leaves of Grass》）
text = gutenberg.raw('shakespeare-hamlet.txt')

# 计算概率分布
def calculate_entropy(frequencies, total_count):
    entropy = 0
    for count in frequencies.values():
        p = count / total_count
        entropy -= p * math.log2(p)
    return entropy

# 按词计算信息熵
words = gutenberg.words('shakespeare-hamlet.txt')
word_frequencies = Counter(words)
word_entropy = calculate_entropy(word_frequencies, len(words))

# 按字母计算信息熵
letters = [char for char in text if char.isalpha()]  # 只考虑字母
letter_frequencies = Counter(letters)
letter_entropy = calculate_entropy(letter_frequencies, len(letters))

print(f"Word-level entropy: {word_entropy:.4f} bits")
print(f"Letter-level entropy: {letter_entropy:.4f} bits")