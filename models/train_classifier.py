import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])

import re
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer 
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import warnings
warnings.simplefilter("ignore")  

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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

def load_data(database_filepath):
    
    # create engine to connect to the database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read the table from the database into a dataframe
    df = pd.read_sql("SELECT * FROM MessagesCategories", engine).head(7000)
    X = df['message']

    # create Y values from the 36 different category columns
    Y = df.iloc[:,4:40]
    
    # take the category names
    category_names = list(Y.columns)

    return X, Y ,category_names
 
def tokenize(text):    
   detected_urls = re.findall(url_regex, text)
   for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

   tokens = word_tokenize(text)
   lemmatizer = WordNetLemmatizer()

   clean_tokens = []
   for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
   return clean_tokens




def  build_model():
    model = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', RandomForestClassifier())
    ])
    
   # define parameters to be tuned with Grid_Search_CV    
    parameters = { }
    #   # use Grid_Search_CV for tuning the parameters defined above
    model = GridSearchCV(model, parameters)
        
    return model 
 


def evaluate_model(model, X_test, Y_test, category_names):
    
    # make predictions with the model
    Y_pred = model.predict(X_test)
    
    # print metrics for each category
    for num, col in enumerate(Y_test.columns):
        print(category_names[num], classification_report(Y_test[col], Y_pred[:, num]))
       
    pass

def save_model(model, model_filepath):    
    # save model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
      
         
        print('Building model...')
        pipeline = build_model()
       
        
        print('Training model...')
        pipeline.fit(X_train, Y_train) 
                  
        print('Evaluating model...')
        evaluate_model(pipeline, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()