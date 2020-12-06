import sys
from sqlalchemy import create_engine
import pandas as pd
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    load data into two data frames
    :param database_filepath: Sqllite db to load
    :return:
        X: array with message values
        Y: array with label data
        category_names:  array of  categories (column names of Y)
    """
    connection_string = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(connection_string)
    df = pd.read_sql_table('disaster_message', con=engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    generate tokens: remove punctuation, stop words and perform Lemmatisation on each word
    :param text:
    :return:
    """
    text = text.strip().lower().translate(str.maketrans('', '', string.punctuation)) # remove puntuation

    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")] # remove stop words
    lemmatizer = WordNetLemmatizer()

    lemmed = [lemmatizer.lemmatize(w) for w in words] #perform Lemmatisation on each word
    return lemmed


def build_model():
    """
    :return: model based on pipeline and parameters
    """
    pipeline_RandomForest = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 100]
    }
    cv = GridSearchCV(pipeline_RandomForest, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Perform prediction and evaluate the model, based on: precision,recall,f1-score
    :param model: model build with the pipeline
    :param X_test: message data
    :param Y_test: label data (categories)
    :param category_names: categories names
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    with open('classification_report.txt', 'w') as f: # save the classification_report
        print(classification_report(Y_test, y_pred, target_names=category_names), 'classification_report.txt', file=f)


def save_model(model, model_filepath):
    """
    save model to a specific path
    :param model:  model to save
    :param model_filepath:  file location to save the model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()