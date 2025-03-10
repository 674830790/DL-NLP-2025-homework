# -*- coding: gb2312 -*-
import os
import math
import re
import unicodedata
from collections import Counter
import jieba
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示（使用 SimHei 字体，如系统中没有请安装或修改为其他支持中文的字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_stopwords(file_path):
    """加载停用词并去除首尾空格"""
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

def read_all_corpus(root_dir):
    """
    遍历整个语料库目录，完整读取每个文件的内容并生成文本。
    对于你的目录结构（wiki_zh 下有 AA, AB, AC…AM 子文件夹，每个子文件夹内有无扩展名的文件），
    该函数都能遍历到所有文件。
    """
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    yield f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

def clean_text_for_char(text):
    """
    用于字符统计的文本清洗：
      - 删除换行符、回车符、制表符
      - 删除所有类型的空格（普通空格和全角空格）
      - 删除斜杠和引号（中英文引号）
      - 删除所有字母（a-z、A-Z）
      - 删除所有标点（利用 Unicode 分类，标点类别以 "P" 开头）
      - 删除所有数字及等号
    """
    # 删除换行、回车、制表符
    text = re.sub(r'[\n\r\t]', '', text)
    # 删除所有类型的空格（普通空格和全角空格）
    text = re.sub(r'[ \u3000]', '', text)
    # 删除斜杠和引号
    text = re.sub(r'[\/"“”]', '', text)
    # 删除所有字母
    text = re.sub(r'[A-Za-z]', '', text)
    # 删除所有标点符号
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    # 删除数字和等号
    text = re.sub(r'[0-9]', '', text)
    text = text.replace('=', '')
    return text

def prepare_text_for_seg(text):
    """
    对文本进行轻度清洗以保留标点，便于 jieba 分词：
    这里只去除换行、回车、制表符，保留其他信息帮助分词
    """
    return re.sub(r'[\n\r\t]', '', text)

def filter_stopwords_words(words, stopwords):
    """
    过滤分词结果：
      - 去除空词
      - 去除仅由标点构成的词
      - 去除停用词
      - 去除不包含任何中文字符的词（例如纯英文的 'n'）
    """
    filtered = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        # 如果词不含任何中文字符，则跳过
        if not any('\u4e00' <= ch <= '\u9fff' for ch in word):
            continue
        # 如果词完全由标点构成，则跳过
        if all(unicodedata.category(ch).startswith('P') for ch in word):
            continue
        if word in stopwords:
            continue
        filtered.append(word)
    return filtered

def filter_stopwords_chars(text, stopwords):
    """
    对字符进行停用词过滤：
    将停用词中的所有字符组成集合，然后过滤掉这些字符
    """
    stop_chars = set(''.join(stopwords))
    return ''.join(ch for ch in text if ch not in stop_chars)

def plot_long_tail(counter, title="Frequency Long Tail Distribution"):
    """绘制长尾分布图（对数坐标）"""
    sorted_freq = sorted(counter.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_freq)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Frequency (log scale)")
    plt.show()

def plot_top50(counter, title="Top 50 Frequency Distribution"):
    """绘制出现频率前50的直方图"""
    top50 = counter.most_common(50)
    labels, values = zip(*top50)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values, tick_label=labels)
    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    CORPUS_DIR = "wiki_zh"            # 语料库根目录
    STOPWORDS_FILE = "cn_stopwords.txt"  # 停用词文件

    stopwords = load_stopwords(STOPWORDS_FILE)
    
    global_char_counter = Counter()
    global_word_counter = Counter()
    total_filtered_chars = 0
    total_filtered_words = 0
    
    file_count = 0
    # 遍历整个语料库中所有文件的内容
    for text in read_all_corpus(CORPUS_DIR):
        file_count += 1
        print(f"Processing file {file_count} with {len(text)} characters")
        
        # 字符统计处理：先进行强清洗，再过滤停用词中的字符
        cleaned_text = clean_text_for_char(text)
        filtered_chars = filter_stopwords_chars(cleaned_text, stopwords)
        global_char_counter.update(filtered_chars)
        total_filtered_chars += len(filtered_chars)
        
        # 分词处理：对文本进行轻度清洗（保留标点以帮助分词），再分词和过滤停用词、非中文词
        seg_text = prepare_text_for_seg(text)
        words = jieba.lcut(seg_text)
        filtered_words = filter_stopwords_words(words, stopwords)
        global_word_counter.update(filtered_words)
        total_filtered_words += len(filtered_words)
    
    # 计算字符信息熵
    if total_filtered_chars > 0:
        char_entropy = -sum((count / total_filtered_chars) * math.log2(count / total_filtered_chars)
                            for count in global_char_counter.values())
    else:
        char_entropy = 0.0
    print(f"全局字符信息熵: {char_entropy:.4f} bits/char")
    
    # 计算词信息熵
    if total_filtered_words > 0:
        word_entropy = -sum((count / total_filtered_words) * math.log2(count / total_filtered_words)
                            for count in global_word_counter.values())
    else:
        word_entropy = 0.0
    print(f"全局词信息熵: {word_entropy:.4f} bits/word")
    
    # 绘制长尾分布图和前50频率直方图
    plot_long_tail(global_char_counter, title="字符长尾分布")
    plot_long_tail(global_word_counter, title="词长尾分布")
    
    plot_top50(global_char_counter, title="出现频率前50的字符")
    plot_top50(global_word_counter, title="出现频率前50的词")

    # 找出字频第一的字符及其频率
if global_char_counter:
    most_common_char, most_common_char_count = global_char_counter.most_common(1)[0]
    print(f"字频最高的字符是: '{most_common_char}'，出现了 {most_common_char_count} 次")
else:
    print("未统计到任何字符")