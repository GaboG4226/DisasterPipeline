# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
	4. [Additional Material](#material)
3. [Authors](#authors)
4. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

The project aim is to develop a multi-output classificator for 36 natural disaster categories to be able to identify the important messages to attend through any specific set of output categories depending on the natural disaster. This project is presented through a Flask Web application that allows to classify any text for the 36 categories and displays three overview graphs of the information used by the ML pipeline for its training. It is important to mention that the graphs show information already filtered with the ETL pipeline.

The project was developed as follows:

1. ETL pipeline preparation
2. ML pipeline preparation
3. Integration of pipelines with Flask App

If wanted detail on the pipeine preparations please refer to '2.1 Jupyter Notebooks for Pïpeline' folder. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries:
    * scikit-learn==0.24.2
* Natural Language Processing Libraries:
    * nltk==3.6.5
* SQLlite Database Libraqries: 
    * SQLAlchemy==1.4.22
* Model Loading and Saving Library: 
    * pickleshare==0.7.5
* Web App and Data Visualization: 
    * Flask==2.2.2
    * plotly==5.13.0
* Data Science:
    * pandas==1.5.3
    * numpy==1.22.4

<a name="installation"></a>
### Requirements installation

The requierements for this project can be installed with the line of code provided next:

```
pip install -r requirements.txt
```

<a name="execution"></a>
### Executing Program:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or any other adress shownn in your running terminal

<a name="material"></a>
### Additional Information

The **ML Pipeline Preparation Notebook** can be used to re-train the model by changing the model and hyperparameters obtained through GridSearch.

<a name="importantfiles"></a>
### File description 
**app/templates/***: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**models/train_classifier.py**: A machine learning pipeline that loads data, trains the model and saves it as a .pkl file to be used in the 'run.py' file

**run.py**: This file can be used to launch the Flask web app used to classify disaster messages

<a name="authors"></a>
## Authors

* [Gabriel Garcia](https://github.com/GaboG4226)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model


