#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-06-14 16:18
"""
# from sklearn.utils import shuffle
# import pandas as pd
#
# df = pd.read_csv("data/dev.csv", index_col=0)
# df = shuffle(df)
# df = df[:3000]
# df.to_csv("dev.csv")

import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
    #           "我 爱 北京 天安门"]  # 第四类文本的切词结果
    # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # tfidf = transformer.fit_transform(
    #     vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
    #     for j in range(len(word)):
    #         print(word[j], weight[i][j])
    # labels = []
    # for i in range(20):
    #     labels.append("DV" + str(i + 1))
    # print(labels)
    1

    # import pandas as pd
    #
    # print(" ".join(list("iooas ")))
    #
    # with open("./train") as f:
    #     lines = []
    #     words = []
    #     labels = []
    #     for line in f:
    #         contends = line.strip()
    #         word = line.strip().split(' ')[0]
    #         label = line.strip().split(' ')[-1]
    #         if contends.startswith("-DOCSTART-"):
    #             words.append('')
    #             continue
    #         # if len(contends) == 0 and words[-1] == '。':
    #         if len(contends) == 0:
    #             l = ' '.join([label for label in labels if len(label) > 0])
    #             w = ' '.join([word for word in words if len(word) > 0])
    #             lines.append([l, w])
    #             words = []
    #             labels = []
    #
    #             continue
    #         words.append(word)
    #         labels.append(label)
    # df = pd.DataFrame()
    # df["text"] = [_[1] for _ in lines]
    # df["label"] = [_[0] for _ in lines]
    # df.to_csv("seq_train.csv")

    from sklearn.externals import joblib

    tokenizer = joblib.load("./out/tokenizer.m")


    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


    def process_one_example(tokenizer, text_a, text_b=None, max_seq_len=256):
        """
            处理 单个样本
        """
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[0:(max_seq_len - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = (input_ids, input_mask, segment_ids)
        return feature


    text = "今天 天气 不错"
    print(process_one_example(tokenizer, text, text_b=None, max_seq_len=256))
