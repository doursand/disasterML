# Disaster Response Pipeline Project

### Summary:
This project is part of the Udacity data scientist nano degree done in partnership with Figure8. 
The objective is to build a NLP machine learning model accessible thru a web app to automatically classify messages describing disaster events. 
The source of the messages used to train the machine learning model are real messages coming from disaster events.

### Requirements:

- sqlalchemy
- numpy
- pandas
- sklearn
- flask
- plotly

### How to use it ?

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database (sqlite database)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model as a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app (using flask and plotly)
    `python run.py`

3. Go to http://0.0.0.0:3001/



