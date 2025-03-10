# -*- coding: gb2312 -*-
import os
import math
import re
import unicodedata
from collections import Counter
import jieba
import matplotlib.pyplot as plt

# ���� matplotlib ֧��������ʾ��ʹ�� SimHei ���壬��ϵͳ��û���밲װ���޸�Ϊ����֧�����ĵ����壩
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_stopwords(file_path):
    """����ͣ�ôʲ�ȥ����β�ո�"""
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

def read_all_corpus(root_dir):
    """
    �����������Ͽ�Ŀ¼��������ȡÿ���ļ������ݲ������ı���
    �������Ŀ¼�ṹ��wiki_zh ���� AA, AB, AC��AM ���ļ��У�ÿ�����ļ�����������չ�����ļ�����
    �ú������ܱ����������ļ���
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
    �����ַ�ͳ�Ƶ��ı���ϴ��
      - ɾ�����з����س������Ʊ��
      - ɾ���������͵Ŀո���ͨ�ո��ȫ�ǿո�
      - ɾ��б�ܺ����ţ���Ӣ�����ţ�
      - ɾ��������ĸ��a-z��A-Z��
      - ɾ�����б�㣨���� Unicode ���࣬�������� "P" ��ͷ��
      - ɾ���������ּ��Ⱥ�
    """
    # ɾ�����С��س����Ʊ��
    text = re.sub(r'[\n\r\t]', '', text)
    # ɾ���������͵Ŀո���ͨ�ո��ȫ�ǿո�
    text = re.sub(r'[ \u3000]', '', text)
    # ɾ��б�ܺ�����
    text = re.sub(r'[\/"����]', '', text)
    # ɾ��������ĸ
    text = re.sub(r'[A-Za-z]', '', text)
    # ɾ�����б�����
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    # ɾ�����ֺ͵Ⱥ�
    text = re.sub(r'[0-9]', '', text)
    text = text.replace('=', '')
    return text

def prepare_text_for_seg(text):
    """
    ���ı����������ϴ�Ա�����㣬���� jieba �ִʣ�
    ����ֻȥ�����С��س����Ʊ��������������Ϣ�����ִ�
    """
    return re.sub(r'[\n\r\t]', '', text)

def filter_stopwords_words(words, stopwords):
    """
    ���˷ִʽ����
      - ȥ���մ�
      - ȥ�����ɱ�㹹�ɵĴ�
      - ȥ��ͣ�ô�
      - ȥ���������κ������ַ��Ĵʣ����紿Ӣ�ĵ� 'n'��
    """
    filtered = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        # ����ʲ����κ������ַ���������
        if not any('\u4e00' <= ch <= '\u9fff' for ch in word):
            continue
        # �������ȫ�ɱ�㹹�ɣ�������
        if all(unicodedata.category(ch).startswith('P') for ch in word):
            continue
        if word in stopwords:
            continue
        filtered.append(word)
    return filtered

def filter_stopwords_chars(text, stopwords):
    """
    ���ַ�����ͣ�ôʹ��ˣ�
    ��ͣ�ô��е������ַ���ɼ��ϣ�Ȼ����˵���Щ�ַ�
    """
    stop_chars = set(''.join(stopwords))
    return ''.join(ch for ch in text if ch not in stop_chars)

def plot_long_tail(counter, title="Frequency Long Tail Distribution"):
    """���Ƴ�β�ֲ�ͼ���������꣩"""
    sorted_freq = sorted(counter.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_freq)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Frequency (log scale)")
    plt.show()

def plot_top50(counter, title="Top 50 Frequency Distribution"):
    """���Ƴ���Ƶ��ǰ50��ֱ��ͼ"""
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
    CORPUS_DIR = "wiki_zh"            # ���Ͽ��Ŀ¼
    STOPWORDS_FILE = "cn_stopwords.txt"  # ͣ�ô��ļ�

    stopwords = load_stopwords(STOPWORDS_FILE)
    
    global_char_counter = Counter()
    global_word_counter = Counter()
    total_filtered_chars = 0
    total_filtered_words = 0
    
    file_count = 0
    # �����������Ͽ��������ļ�������
    for text in read_all_corpus(CORPUS_DIR):
        file_count += 1
        print(f"Processing file {file_count} with {len(text)} characters")
        
        # �ַ�ͳ�ƴ����Ƚ���ǿ��ϴ���ٹ���ͣ�ô��е��ַ�
        cleaned_text = clean_text_for_char(text)
        filtered_chars = filter_stopwords_chars(cleaned_text, stopwords)
        global_char_counter.update(filtered_chars)
        total_filtered_chars += len(filtered_chars)
        
        # �ִʴ������ı����������ϴ����������԰����ִʣ����ٷִʺ͹���ͣ�ôʡ������Ĵ�
        seg_text = prepare_text_for_seg(text)
        words = jieba.lcut(seg_text)
        filtered_words = filter_stopwords_words(words, stopwords)
        global_word_counter.update(filtered_words)
        total_filtered_words += len(filtered_words)
    
    # �����ַ���Ϣ��
    if total_filtered_chars > 0:
        char_entropy = -sum((count / total_filtered_chars) * math.log2(count / total_filtered_chars)
                            for count in global_char_counter.values())
    else:
        char_entropy = 0.0
    print(f"ȫ���ַ���Ϣ��: {char_entropy:.4f} bits/char")
    
    # �������Ϣ��
    if total_filtered_words > 0:
        word_entropy = -sum((count / total_filtered_words) * math.log2(count / total_filtered_words)
                            for count in global_word_counter.values())
    else:
        word_entropy = 0.0
    print(f"ȫ�ִ���Ϣ��: {word_entropy:.4f} bits/word")
    
    # ���Ƴ�β�ֲ�ͼ��ǰ50Ƶ��ֱ��ͼ
    plot_long_tail(global_char_counter, title="�ַ���β�ֲ�")
    plot_long_tail(global_word_counter, title="�ʳ�β�ֲ�")
    
    plot_top50(global_char_counter, title="����Ƶ��ǰ50���ַ�")
    plot_top50(global_word_counter, title="����Ƶ��ǰ50�Ĵ�")

    # �ҳ���Ƶ��һ���ַ�����Ƶ��
if global_char_counter:
    most_common_char, most_common_char_count = global_char_counter.most_common(1)[0]
    print(f"��Ƶ��ߵ��ַ���: '{most_common_char}'�������� {most_common_char_count} ��")
else:
    print("δͳ�Ƶ��κ��ַ�")