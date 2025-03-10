import nltk
from nltk.corpus import gutenberg
from collections import Counter
import matplotlib.pyplot as plt
import re
import math

# 下载语料库（首次使用时需要）
#nltk.download('gutenberg')

# 查看所有可用的文本
print("Available texts in Gutenberg Corpus:")
print(gutenberg.fileids())

# 选择一个文本（莎士比亚的《Hamlet》）
text = gutenberg.raw('shakespeare-hamlet.txt')

# 文本预处理：去除标点符号，只保留字母和空格
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

# 提取字母（全部转小写）
letters = [char.lower() for char in cleaned_text if char.isalpha()]
letter_frequencies = Counter(letters)

# 提取单词（按空格分词，并去除空白项）
words = cleaned_text.lower().split()
word_frequencies = Counter(words)

# 计算信息熵函数
def calculate_entropy(frequencies, total_count):
    entropy = 0
    for count in frequencies.values():
        p = count / total_count
        entropy -= p * math.log2(p)
    return entropy

# 计算字母信息熵
letter_entropy = calculate_entropy(letter_frequencies, len(letters))
print(f"Average Letter Entropy: {letter_entropy:.4f} bits")

# 计算单词信息熵
word_entropy = calculate_entropy(word_frequencies, len(words))
print(f"Average Word Entropy: {word_entropy:.4f} bits")

# 绘制字母的长尾分布图
sorted_letter_freq = sorted(letter_frequencies.values(), reverse=True)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(sorted_letter_freq) + 1), sorted_letter_freq, marker='o', color='skyblue')
plt.title('Long-tail Distribution of Letters in Hamlet')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.yscale('log')
plt.show()

# 绘制单词的长尾分布图
sorted_word_freq = sorted(word_frequencies.values(), reverse=True)
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(sorted_word_freq) + 1), sorted_word_freq, marker='o', color='lightcoral')
plt.title('Long-tail Distribution of Words in Hamlet')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.yscale('log')
plt.show()

# 绘制前50个单词的词频直方图
top_50_words = word_frequencies.most_common(50)
words, frequencies = zip(*top_50_words)

plt.figure(figsize=(12, 6))
plt.bar(words, frequencies, color='lightgreen')
plt.title('Top 50 Word Frequencies in Hamlet')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()