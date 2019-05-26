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
    engine = create_engine('sqlite:///disasterdb.db')

    df = pd.read_sql_table('disaster',con=engine)
    X = df['message']
    Y = df[['related-1', 'request-0', 'offer-0', 'aid_related-0', 'medical_help-0', 'medical_products-0', 
            'search_and_rescue-0', 'security-0', 'military-0', 'child_alone-0', 'water-0', 'food-0', 
            'shelter-0', 'clothing-0', 'money-0', 'missing_people-0', 'refugees-0', 'death-0', 'other_aid-0', 
            'infrastructure_related-0', 'transport-0', 'buildings-0', 'electricity-0', 'tools-0', 'hospitals-0', 
            'shops-0', 'aid_centers-0', 'other_infrastructure-0', 'weather_related-0', 'floods-0', 'storm-0', 
            'fire-0', 'earthquake-0', 'cold-0', 'other_weather-0', 'direct_report-0']]


def tokenize(text):
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