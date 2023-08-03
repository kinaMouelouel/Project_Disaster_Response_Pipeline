from flask import Flask,render_template, request 
import pandas as pd
from sqlalchemy import create_engine
import json, plotly 
from plotly.graph_objs import   Bar
import sys
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

app = Flask(__name__) 

#load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategories', engine)
sys.path.append("../models")
 
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
   genre_counts = df.groupby('genre').count()['message']
   genre_names = list(genre_counts.index)
   
   graph  = []
    
   graph.append(
        Bar(
      x = genre_names,
      y = genre_counts,
      )
    )
   layout  = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'),
                )
    
   figures = []

    
   figures.append(dict(data=graph, layout=layout))
      # Convert the plotly figures to JSON for javascript in html template
   figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
     # plot ids for the html id tag
   ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
   return render_template('index.html',figuresJSON=figuresJSON, ids=ids)


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