import pickle
import re

import numpy as np
from gensim.models.wrappers.ldamallet import LdaMallet
from gensim.corpora import Dictionary
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class PandasBowVectorizer(TransformerMixin):
    # A bag-of-words vectorizer that works with an entire DataFrame as input.
    # This lets me create a combined feature pipeline with multiple feature vectorizers
    # where I don't have to worry about passing different columns to each.

    def __init__(self, *args, **kwargs):
        self.vectorizer = CountVectorizer(*args, **kwargs)

    def fit(self, X, *args, **kwargs):
        self.vectorizer.fit(X['text'].values)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X['text'].values)


class PandasTfidfVectxorizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.vectorizer = TfidfVectorizer(*args, **kwargs)

    def fit(self, X, *args, **kwargs):
        self.vectorizer.fit(X['text'].values)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X['text'].values)


class CustomFeatures(TransformerMixin):
    # Base class for custom features
    
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler()

    def fit(self, reviews, *args, **kwargs):
        print(f'WARNING: ignoring {args}, {kwargs}')
        self.scaler.fit(np.array([self._custom_features(r) for r in reviews.itertuples()]))
        return self
    
    def _custom_features(self, review):
        raise NotImplementedError

    def transform(self, reviews):
        return self.scaler.transform(
            np.array([self._custom_features(r) for r in reviews.itertuples()])
        )

    
class OtherReviewFeatures(CustomFeatures):
    def _custom_features(self, review):
        return [
            len(review.text),
            review.stars,
            review.readability_standard,
            review.sentiment,
        ]


class ReviewerFeatures(CustomFeatures):
    def _custom_features(self, review):
        return [
            review.fans,
            review.usr_avg_stars,
            review.usr_review_count,
            review.elite_count,
            review.usr_total_votes,
            review.avg_nw_score,
            review.max_nw_score,
            review.friend_count,
            review.posted_days
        ]
