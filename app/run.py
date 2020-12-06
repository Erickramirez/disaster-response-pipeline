import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
#import sklearn.externals.joblib as joblib
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = text.strip().lower().translate(str.maketrans('', '', string.punctuation))  # remove puntuation

    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]  # remove stop words
    lemmatizer = WordNetLemmatizer()

    lemmed = [lemmatizer.lemmatize(w) for w in words]  # perform Lemmatisation on each word
    return lemmed

def wordcloud_to_ploty(df):
    """
    Genrates a json with word cloud data  for rendering in plotly
    :param df: data fram with the messages
    :return: SON of plotly data
    """

    words = df.message.str.cat(sep=' ').lower().translate(str.maketrans('', '', string.punctuation)).split()
    df_words = pd.Series(words)
    full_words = ' '.join(df_words[~df_words.isin(stopwords.words("english"))].to_numpy())

    wordcloud = WordCloud(max_words=100).generate(full_words)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wordcloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get relative occurrence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 100)

    wordcloud_json = {
        'data': [
            Scatter(
                x=x,
                y=y,
                textfont=dict(size=new_freq_list,
                              color=color_list),
                hoverinfo='text',   
                hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                mode='text',
                text=word_list
            )
        ],

        'layout': {
            'title': 'Most Frequent Words Used in a Disaster Message',
            'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
        }
    }

    return wordcloud_json

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    ###
    categories_df = df.drop(['id'], axis=1)._get_numeric_data()
    top_categories_count = categories_df.sum().sort_values(ascending=False)
    top_categories_names = list(top_categories_count.index)

    JobFactors = df['message'].str.split(' ', expand=True).stack().value_counts()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_categories_names,
                    y=top_categories_count
                )
            ],

            'layout': {
                'title': 'Categories Order by Count',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                {
                    "type": "pie",
                    "hole": 0.4,
                    "name": "Genre",
                    "pull": 0,
                    "domain": {
                        "x": genre_counts,
                        "y": genre_names
                    },
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
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
        wordcloud_to_ploty(df)
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
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()