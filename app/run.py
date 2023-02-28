import json
import joblib
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_pipeline.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/pipeline.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract column names
    class_list = df.columns[4:]

    # list to store count of each column
    values_list = []

    # lists to get count of classifications by genre for graph 2
    direct_list = []
    news_list = []
    social_list = []

    # loop to get the count of rows of each column individually and by genre
    for column in class_list:
       values_list.append(df[df[column] == 1].count()[0])
       direct_list.append(df[(df['related'] == 1) & (df['genre'] == 'direct')].count()[0])
       news_list.append(df[(df['related'] == 1) & (df['genre'] == 'news')].count()[0])
       social_list.append(df[(df['related'] == 1) & (df['genre'] == 'social')].count()[0])

    
    # create visuals
    graphs = [
        
        # Graph 1 - Messages by gensre
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 

        # Graph 2 - Messages by classification
        {
            'data': [
                Bar(
                    x = class_list,
                    y = values_list
                )
            ],

            'layout': {
                'title':'Distribution of messages by classification',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Classification',
                    'titlefont': {
                        # padding is not working for some reason
                        'padding': '100'
                    },
                },
            'barmode':'stack'
            }
        },

        # Graph 3 - Messages of each classification by genre
        {
            'data': [
                Bar(
                    x = class_list,
                    y = direct_list,
                    name = 'Direct'
                ),
                Bar(
                    x = class_list,
                    y = news_list,
                    name = 'News'
                ),
                Bar(
                    x = class_list,
                    y = social_list,
                    name = 'Social'
                )
            ],

            'layout': {
                'title': 'Distribution of messages by classification and genre',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Classification',
                    'titlefont': {
                        # padding not working for some reason
                        'padding': '100'
                    },
                },
                'barmode': 'stack'
            }
        },

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()