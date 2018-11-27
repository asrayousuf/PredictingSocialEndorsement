import pickle
import re

import numpy as np
from gensim.models.wrappers.ldamallet import LdaMallet
from gensim.corpora import Dictionary
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class Corpus():
    def __init__(self, texts):
        dictionary = Dictionary([self.text_to_tokens(text) for text in texts])
        dictionary.filter_extremes(no_below=5, no_above=0.6)
        self._dictionary = dictionary
        self._texts = texts

    def text_to_tokens(self, text):
        return [w for w in re.split('\W', text) if w]
    
    def doc2bow(self, text):
        return self._dictionary.doc2bow(self.text_to_tokens(text))

    def __iter__(self):
        for text in self._texts:
            yield self.doc2bow(text)


class TopicModelVectorizer(TransformerMixin):
    def __init__(self, model, corpus):
        self._model = model
        self._corpus = corpus
        self.scaler = StandardScaler()

    def fit(self, reviews, *args, **kwargs):
        print(f'WARNING: ignoring {args}, {kwargs}')
        self.scaler.fit(self.transform(reviews, scale=False))
        return self
    
    def topic_scores(self, reviews):
        return self._model[
            [self._corpus.doc2bow(review) for review in reviews.text.values]
        ]

    def transform(self, reviews, scale=True):
        all_reviews_topic_scores = self.topic_scores(reviews)
        topic_vectors = np.array(
            [
                [tup[1] for tup in sorted(review_topic_scores, key=lambda x: x[0])]
                for review_topic_scores in all_reviews_topic_scores
            ])
        if scale:
            return self.scaler.transform(topic_vectors)
        else:
            return topic_vectors


class PandasTopicVectorizer(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, reviews, *args, **kwargs):
        print(f'WARNING: ignoring {args}, {kwargs}')
        self.scaler.fit(self.transform(reviews, scale=False))
        return self
    
    def transform(self, reviews, scale=True):
        columns = sorted([col for col in reviews.columns if col.startswith('Topic #')])
        topic_vectors = reviews[columns].values
        if scale:
            return self.scaler.transform(topic_vectors)
        else:
            return topic_vectors


def load_topic_model(model_path):
    with open(model_path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded['corpus'], loaded['topic_model']
