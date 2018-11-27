import argparse

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier

from topic_utils import load_topic_model, TopicModelVectorizer, PandasTopicVectorizer, Corpus
from feature_utils import PandasBowVectorizer, OtherReviewFeatures, ReviewerFeatures

def run_classification(
    reviews_df, models, vectorizers, limit_inst=None,
    weight_fn=None, scaler=None, test_df=None
):
    # Create train-test split
    train_df, val_df = train_test_split(reviews_df, test_size=0.2)
    if limit_inst:
        train_df = train_df.sample(limit_inst)
        val_df = val_df.sample(limit_inst)
        if test_df is not None:
            test_df = test_df.sample(limit_inst)
    #Create structures to print results of classification nicely:
    table = PrettyTable()
    cols = ['Model', 'Vectorizer', 'Train Acc.', 'Validation Acc.']
    if test_df is not None:
        cols.append('Test Acc.')
    table.field_names = cols

    # Create training labels
    trainLabels = (train_df['total_votes'] > 0).values
    valLabels = (val_df['total_votes'] > 0).values
    if test_df is not None:
        testLabels = (test_df['total_votes'] > 0).values
    
    # This way, we can do some calculation metrics by hand:
    print(f'Train instances: {trainLabels.shape[0]}, Val instances: {valLabels.shape[0]}')
    print(np.unique(trainLabels, return_counts=True))
    print(np.unique(valLabels, return_counts=True))
    
    # Now we do the classification:
    for vectorizer in vectorizers:
        createdVectorizer = vectorizer()
        createdVectorizer.fit(train_df)
        vectorizer_name = createdVectorizer.__class__.__name__
        if vectorizer_name == 'FeatureUnion':
            vectorizer_name = ' + '.join(
                tr[1].__class__.__name__ for tr in createdVectorizer.transformer_list
            )
        print(f'Fit vectorizer {vectorizer_name}')
        train_vectors = createdVectorizer.transform(train_df)
        val_vectors = createdVectorizer.transform(val_df)
        if test_df is not None:
            test_vectors = createdVectorizer.transform(test_df)
        if scaler:
            createdScaler = scaler()
            train_vectors = createdScaler.fit_transform(train_vectors)
            val_vectors = createdScaler.transform(val_vectors)
            if test_df is not None:
                test_vectors = createdScaler.transform(test_vectors)
        print('Created vectors')
        train_accs, val_accs, test_accs = [], [], []
        for model in models:
            if (model == LinearSVC):
                createdModel = model(max_iter=5000)
            else:
                createdModel = model()

            if weight_fn:
                createdModel.fit(train_vectors, trainLabels, sample_weight=weight_fn(train_df))
            else:
                createdModel.fit(train_vectors, trainLabels)
            print(f'Fit model {model.__name__}')

            #Get error stats:
            #First the predictions:
            train_predictions = createdModel.predict(train_vectors)
            val_predictions = createdModel.predict(val_vectors)
            if test_df is not None:
                test_predictions = createdModel.predict(test_vectors)

            #Now the actual scores:
            train_accs.append(accuracy_score(trainLabels, train_predictions))
            val_accs.append(accuracy_score(valLabels, val_predictions))
            if test_df is not None:
                test_accs.append(accuracy_score(testLabels, test_predictions))
        #Now build the tables:
        max_idx = np.argmax(val_accs)
        train_acc = train_accs[max_idx]
        val_acc = val_accs[max_idx]
        model_name = models[max_idx].__name__
        row = [model_name, vectorizer_name, f'{train_acc:.3f}', f'{val_acc:.3f}']
        if test_df is not None:
            test_acc = test_accs[max_idx]
            row.append(f'{test_acc:.3f}')
        table.add_row(row)
    print(table)


def create_vectorizer_fns():
    bow_params = {
        'analyzer': 'word',
        'stop_words': 'english',
        'min_df': 50,
        'max_df': 0.7,
        'max_features': 2000,
    }
    bow_fn = lambda: PandasBowVectorizer(**bow_params)
    topics_fn = lambda: PandasTopicVectorizer()
    review_fn = lambda: OtherReviewFeatures()
    reviewer_fn = lambda: ReviewerFeatures()
    topics_review_fn = lambda: FeatureUnion([
        ('topics', topics_fn()),
        ('review', OtherReviewFeatures()),
    ])
    topics_review_reviewer_fn = lambda: FeatureUnion([
        ('topics', topics_fn()),
        ('review', OtherReviewFeatures()),
        ('reviewer', ReviewerFeatures()),
    ])
    return [bow_fn, topics_fn, review_fn, reviewer_fn, topics_review_fn, topics_review_reviewer_fn]


def main(data_file, test_file):
    models = [RandomForestClassifier, LogisticRegression, XGBClassifier, LinearSVC]
    vectorizers = create_vectorizer_fns()
    df = pd.read_csv(data_file)
    if test_file:
        test_df = pd.read_csv(test_file)
    else:
        test_df = None
    run_classification(df, models, vectorizers, limit_inst=500, test_df=test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_file', help="Which dataset to use for training and validation")
    # parser.add_argument(
    #     'topic_model_file', help="Which topic model to use")
    parser.add_argument(
        '-t', '--test-file', default=None, help="Which dataset to use for held-out testing")

    args = parser.parse_args()
    main(args.data_file, args.test_file)

