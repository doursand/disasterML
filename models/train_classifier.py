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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


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
    
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('disaster',con=engine)
    categories_list = ['related-1', 'request-0', 'offer-0', 'aid_related-0', 'medical_help-0', 'medical_products-0', 
        'search_and_rescue-0', 'security-0', 'military-0', 'child_alone-0', 'water-0', 'food-0', 
        'shelter-0', 'clothing-0', 'money-0', 'missing_people-0', 'refugees-0', 'death-0', 'other_aid-0', 
        'infrastructure_related-0', 'transport-0', 'buildings-0', 'electricity-0', 'tools-0', 'hospitals-0', 
        'shops-0', 'aid_centers-0', 'other_infrastructure-0', 'weather_related-0', 'floods-0', 'storm-0', 
        'fire-0', 'earthquake-0', 'cold-0', 'other_weather-0', 'direct_report-0']
        #list(df.columns[:4])
    X = df['message'].values
    Y = df[categories_list].values
    return X, Y, categories_list

def tokenize(text):
    '''
    Process and tokenize text
    Input:
        text: str. Sqlite database path
    Output:
        clean_tokens. list of cleaned tokens created from text
    '''
    # Normalize Text, lower case
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    #lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens
    


def build_model():
    '''
    Build the pipeline for the ML model. Model is optimized thru gridsearch
    Output:
       cv. model optimized thru GridSearchCV
    
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('multiclf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])
    parameters = {
        'multiclf__estimator__n_estimators': [50,100],
        'multiclf__estimator__algorithm': ['SAMME', 'SAMME.R']
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,cv=2)
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model, providing f1 score and accuracy of the model for each category
        Input:
            model: model
            X_test: X_test 
            Y_test: Y_test
            category_names: list of category from db
    '''
    
    Y_pred = model.predict(X_test)
    for category in range(len(category_names)):
        print('category: {}'.format(category_names[category]))
        print(classification_report(Y_test[:, category], Y_pred[:, category]))



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