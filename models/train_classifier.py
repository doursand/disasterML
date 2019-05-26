# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
     '''
    Load data from database
    Input:
        database_filepath: str. Sqlite database path
    Output:
        X: Dataframe. features
        Y: Dataframe. independant variables
        categories_list: List. list of all categories   
    '''
    
    engine = create_engine('sqlite:///disasterdb.db')
    
    df = pd.read_sql_table('disaster',con=engine)
    categories_list = list(df.columns[:4])
    X = df['message']
    Y = df[categories_list]
    return X, Y, categories_list

def tokenize(text):
    '''
    Process and tokenize text
    Input:
        text: str. Sqlite database path
    
    '''
    # Normalize Text, lower case
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    '''
    Build the pipeline for the ML model. Model is optimized thru gridsearch
    
    
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multiclf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
                 
                 'vect__max_df': (0.5, 1.0),
                 'vect__max_features': (None, 500, 1000),
                 'tfidf__use_idf': (True, False),
                 'multiclf__estimator__n_estimators': [10, 20],
                 'multiclf__estimator__min_samples_split': [0.5, 1.0],
                 'multiclf__estimator__criterion': ['entropy', 'gini']
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    '''
    Save the trained model
    Input:
        model: pipeline model
        model_filepath: str. Path of the saved model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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