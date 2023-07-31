# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from custom_transformer import StartingVerbExtractor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pickle



def load_data(database_filepath):
    
    # create engine to connect to the database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read the table from the database into a dataframe
    df = pd.read_sql("SELECT * FROM MessagesCategories", engine)
    
    # create X from the "message" column
    X = df['message']

    # create Y values from the 36 different category columns
    Y = df.iloc[:,4:40]
    
    # take the category names
    category_names = list(Y.columns)

    return X, Y, category_names



def tokenize(text):

    # changing all letters to lowercase
    text = text.lower()
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # split text into tokens
    words = word_tokenize(text)
    
    # removing stopwords (for example the, an, my etc.)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # lemmatization, reduce words to their root form.
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed




def build_model():
      # text processing and model pipeline
     pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
     # define parameters for GridSearchCV
     parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
     # create gridsearch object and return as final model pipeline

     cv = GridSearchCV(pipeline, param_grid=parameters)

     return cv




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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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