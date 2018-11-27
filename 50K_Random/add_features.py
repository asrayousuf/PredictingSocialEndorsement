# Adds sentiment, topic and readability data to input train/test files
#
# Example usage:
# python add_features.py sampleData/train_reviews_BS.csv sampleData/test_reviews_BS.csv -m models/bs_topics_100_50000
#
# NOTE: Requires pre-trained topic model

import argparse
import pickle
import sys

import pandas as pd
import textstat
from textblob import TextBlob
from topic_utils import load_topic_model, Corpus, TopicModelVectorizer


def process_file(train_file, test_file, topic_model_file):
    dfs = {
        'train': pd.read_csv(train_file),
        'test': pd.read_csv(test_file),
    }
    # Remove outdated columns
    columns_to_remove = [
        'coleman_liau_index',
        'automated_readability_index',
        'dale_chall_readability_score',
        'linsear_write_formula',
        'gunning_fog',
        'flesch_reading_ease',
        'Unnamed: 0',
        'Unnamed: 0.1',
        'Unnamed: 0.1.1',
    ]
    for key, df in dfs.items():
        for col in columns_to_remove:
            dfs[key] = dfs[key].drop(col, axis=1)
        print('Removed old columns')
        dfs[key]['readability_standard'] = df['text'].apply(lambda r: textstat.text_standard(r, float_output=True))
        print('Added readability')
        dfs[key]['sentiment'] = df['text'].apply(lambda r: TextBlob(r).sentiment.polarity)
        print('Added sentiment')
    # Add topic scores
    corpus, topic_model = load_topic_model(topic_model_file)
    topics_vectorizer = TopicModelVectorizer(topic_model, corpus)
    topic_scores = {}
    topic_scores['train'] = topics_vectorizer.fit_transform(dfs['train'])
    topic_scores['test'] = topics_vectorizer.transform(dfs['test'])
    print('Fetched topic scores')
    for key in dfs.keys():
        scores = topic_scores[key]
        scores_df = pd.DataFrame(
            data=scores, columns=[f'Topic #{i}' for i in range(scores.shape[1])], index=dfs[key].index)
        dfs[key] = dfs[key].merge(
            scores_df,
            left_index=True, right_index=True
        )
        print('Added topic scores')
    dfs['train'].to_csv(train_file, index=False)
    dfs['test'].to_csv(test_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Train file to update')
    parser.add_argument('test_file', type=str, help='Test file to update')
    parser.add_argument('-m','--topic-model-file', type=str, help='Path to topic model to use', required=True)

    args = parser.parse_args()
    process_file(args.train_file, args.test_file, args.topic_model_file)
