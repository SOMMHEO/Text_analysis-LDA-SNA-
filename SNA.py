import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from apyori import apriori
from konlpy.tag import Okt
import matplotlib.font_manager as fm
from matplotlib import rc


class SNA:
    def __init__(self):
        self.stopwords = []
        self.additional_stopwords = []
        self.reviews = []
        # self.stopwords_path = ""
        self.min_support = 0.01

    def preprocess_reviews(self):
        # with open(self.stopwords_path, encoding='utf-8') as file:
        #     self.stopwords = file.read().splitlines()
        # self.stopwords.extend(self.additional_stopwords)

        clean_sent = []
        for review in self.reviews:
            clean = re.sub(r'[-=+,#/\?:^$.@*"※~&%ㆍ!』\\‘·|\(\)\[\]\<\>`\'…\"\“’]', '', review)
            clean = clean.replace('\n', '').replace('\r', '')  # 줄바꿈 기호 삭제
            clean_sent.append(clean)

        okt = Okt()

        word_list = []
        for sentences in clean_sent:
            sentences = okt.normalize(sentences)
            tokens = okt.nouns(sentences)
            tokens = [token for token in tokens if token not in self.additional_stopwords]
            word_list.append(tokens)

        return word_list

    def generate_association_rules(self, word_list):
        result = list(apriori(word_list, min_support=self.min_support))
        df = pd.DataFrame(result)
        df['length'] = df['items'].apply(lambda x: len(x))
        df = df[(df['length'] == 2) & (df['support'] >= self.min_support)].sort_values(by='support', ascending=False)
        return df

    def visualize_network_graph(self, items):
        rc('font', family='Malgun Gothic')  # 전역 폰트 설정

        G = nx.Graph()
        G.add_edges_from(items)

        pr = nx.pagerank(G)
        nsize = np.array([v for v in pr.values()])
        nsize = 2000 * (nsize - min(nsize)) / (max(nsize) - min(nsize))

        pos = nx.kamada_kawai_layout(G)

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        nx.draw_networkx(G, font_family='Malgun Gothic', font_size=16,
                         node_color=list(pr.values()), node_size=nsize,
                         alpha=0.7, edge_color='.5', cmap=plt.cm.YlGn)

