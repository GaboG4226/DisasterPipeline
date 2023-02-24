import sys

# General libraries
import pandas as pd
import lazypredict
import numpy as np
import re

# Database libraries
from sqlalchemy import create_engine

# Tokenization libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

# ML libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Best model for our data analysis
from lazypredict.Supervised import LazyClassifier, LazyRegressor


def load_data(database_filepath):
    '''
    loads the data from a database 
    
    INPUT: 
        database_filepath: filepath in which the database is located
    
    OUTPUT:
        dataframe with the information read on the database
    
    '''
    
    # create connection to the sqlite database
    engine = create_engine('sqlite:///disaster_pipeline.db')
    
    # creation of dataframe with specific table from the database
    df = pd.read_sql('messages_categories', engine)
    
    return df


def tokenize(text):
    
    '''
    tokenize texts by previously lemmatizing, eliminating stop words and changing words to lower case
    
    INPUT:
        text: texts to be tokenized
        
    OUTPUT:
        text tokenized with all the intermediate steps performed
    '''
    
    # Definition of lemmatizer to be used for tokenization
    lemmatizer = WordNetLemmatizer()
    
    # Stop words definition
    stop_words = stopwords.words("english")
    
    # Normalization case and punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenization of words
    tokens = word_tokenize(text)
    
    # Stop words removal
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return tokens


def build_model():
    
    '''
    builds the model that is going to be used through a pipeline, where gridsearch is used for tunning hyperparameters
        
        
    OUTPUT:
        pipeline used as model for the training
    '''
    
    # pipeline definition
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=2000))),
    ])
    
    # parameters to be used for gridsearch
    parameters = {
        'clf__estimator__C': [0.1, 1, 10]
    }
    
    # grid search for tunning hyperparameters
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    evaluates the model by making a predictions with the X_test
    
    INPUT:
        model: model to be evaluated with the test set
        X_test: input test set that will be used to generate predictions with the model
        Y_test: results test set that will be used to evaluate the model performance
        category_names: categories to be analyzed within each column of the Y_test set
        
    OUTPUT:
        print the classification report by column name with the categories specified in the input
    '''
    # transform to dataframe to get classification_report by column
    predicted_df = pd.DataFrame(model.predict(X_test), columns=Y_test.columns)

    # loop to get report by column
    for column in Y_test.columns:
        
        # creation and printing of report
        report = classification_report(y_test[column], predicted_df[column], target_names=category_names, zero_division=0)
        print("\n Classification [Original] report for column '{}': \n {}".format(column, report))


def save_model(model, model_filepath):
    
    '''
    saves the model to a pickle file
    
    INPUT:
        model: model to be saved in pickle file
        model_filepath: path where we want our model to be saved
        
    OUTPUT:
        model saved in a .pkl file in the specified filepath
    '''
    
    pickle.dump(pipeline, open('pipeline.pkl', 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()