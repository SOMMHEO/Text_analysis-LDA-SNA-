import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd

# LDA
import re
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 한국어
# from eunjeon import Mecab
from konlpy.tag import Okt

class LDAModel:
    def __init__(self):
        self.stopwords = []
        self.additional_stopwords = []
        self.reviews = []
        self.stopwords_path = ""
        self.limit = 10
        self.start = 2
        self.step = 1
        self.id2word = None
        self.corpus = None
        self.corpus_tfidf = None

    def preprocess_reviews(self):
        with open(self.stopwords_path, encoding='utf-8') as file:
            self.stopwords = file.read().splitlines()
        self.stopwords.extend(self.additional_stopwords)

        clean_sent = []
        for review in self.reviews:
            clean = re.sub(r'[-=+,#/\?:^$.@*"※~&%ㆍ!』\\‘·|\(\)\[\]\<\>`\'…\"\“’]', '', review)
            clean = clean.replace('\n', '').replace('\r', '') # 줄바꿈 기호 삭제
            clean_sent.append(clean)

        okt = Okt()
        # m = Mecab()

        word_list = []
        for sentences in clean_sent:
            sentences = okt.normalize(sentences)
            tokens = okt.nouns(sentences)
            tokens = [token for token in tokens if token not in self.stopwords]
            word_list.append(tokens)

        return word_list

    def compute_coherence_values(self):
        coherence_values = []
        model_list = []

        for num_topics in range(self.start, self.limit, self.step):
            model = gensim.models.LdaMulticore(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, passes=2, workers=4)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.preprocessed_reviews, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values


    # optimal LDA model(coherence socre)
    def lda_modeling(self):
        self.preprocessed_reviews = self.preprocess_reviews()

        self.id2word = corpora.Dictionary(self.preprocessed_reviews)
        self.id2word.filter_extremes(no_below=10)
        self.corpus = [self.id2word.doc2bow(text) for text in self.preprocessed_reviews]

        tf_idf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = tf_idf[self.corpus]

        model_list, coherence_values = self.compute_coherence_values()

        x = range(self.start, self.limit, self.step)
        plt.plot(x, coherence_values)
        plt.xlabel('Num Topics')
        plt.ylabel('Coherence score')
        plt.legend(('coherence_values'), loc='best')
        plt.show()

        optimal_model = model_list[coherence_values.index(max(coherence_values))]
        model_topics = optimal_model.show_topics(formatted=False)

        return optimal_model, model_topics, self.corpus, self.id2word, self.corpus_tfidf

    def document_topics_probability(self, optimal_model, corpus_tfidf, word_list):
        document_topics_df = pd.DataFrame()

        for i, row in enumerate(optimal_model[corpus_tfidf]): # optimal_model에 corpus를 넣어 각 토픽 당 확률을 알 수 있음
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:
                    omt = optimal_model.show_topic(topic_num, topn=10)
                    topic_keywords = ", ".join([word for word, prop in omt])
                    new_row = pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords], index=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
                    document_topics_df = pd.concat([document_topics_df, new_row.to_frame().T], ignore_index=True)
                else:
                    break
        document_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        return document_topics_df

    # Error 발생 시 수정
    # visualize_lda_model 메서드 호출하여 시각화
    def visualize_lda_model(self, optimal_model):
        pyLDAvis.enable_notebook()
        vis_data = gensimvis.prepare(optimal_model, self.corpus_tfidf, self.id2word)
        pyLDAvis.display(vis_data)