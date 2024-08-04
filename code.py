#!/usr/bin/env python
# coding: utf-8

# All libraries to be imported
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
get_ipython().system(' pip install gensim')
import gensim
from gensim.utils import simple_preprocess
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
get_ipython().system(' pip install imbalanced-learn')
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Remove useless columns (to be removed when predicting for test dataset)
#train_df = train_df.drop(columns = ['teacher_id', 'project_submitted_datetime', 'project_essay_3', 'project_essay_4'])
#train_df = train_df.dropna()

# Remove useless features from train and test dataset (to be used when predicting for test dataset)

train_df = train_df.drop(columns = ['teacher_id', 'project_submitted_datetime', 'project_essay_3', 'project_essay_4'])
train_df = train_df.dropna()
test_df = test_df.drop(columns = ['teacher_id', 'project_submitted_datetime', 'project_essay_3', 'project_essay_4'])

# Splitting data into train and test data

target_feature = 'project_is_approved'
remain_feature =  train_df.columns[train_df.columns != target_feature]

train_dataset = train_df[remain_feature]
y_train = train_df[target_feature]

test_dataset = test_df

# train_dataset, test_dataset, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Concatenate all text columns

train_dataset['project_text_all'] = train_dataset['project_essay_1'].fillna('') + ' ' + train_dataset['project_essay_2'].fillna('') + ' ' + train_dataset['project_title'].fillna('') + ' ' + train_dataset['project_resource_summary'].fillna('')
test_dataset['project_text_all'] = test_dataset['project_essay_1'].fillna('') + ' ' + test_dataset['project_essay_2'].fillna('') + ' ' + test_dataset['project_title'].fillna('') + ' ' + test_dataset['project_resource_summary'].fillna('')

train_dataset = train_dataset.drop(columns = ['project_essay_1', 'project_essay_2', 'project_title', 'project_resource_summary'])
test_dataset = test_dataset.drop(columns = ['project_essay_1', 'project_essay_2', 'project_title', 'project_resource_summary'])

# Combine both test and train dataset for label encoding

combined_dataset = pd.concat([train_dataset, test_dataset], axis=0, ignore_index=True)

# Perform Label encoding on combined dataset

label_encoder_prefix = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_grade = LabelEncoder()
label_encoder_categories = LabelEncoder()
label_encoder_subcategories = LabelEncoder()

combined_dataset['teacher_prefix'] = label_encoder_prefix.fit_transform(combined_dataset['teacher_prefix'])
combined_dataset['school_state'] = label_encoder_state.fit_transform(combined_dataset['school_state'])
combined_dataset['project_grade_category'] = label_encoder_grade.fit_transform(combined_dataset['project_grade_category'])
combined_dataset['project_subject_categories'] = label_encoder_categories.fit_transform(combined_dataset['project_subject_categories'])
combined_dataset['project_subject_subcategories'] = label_encoder_subcategories.fit_transform(combined_dataset['project_subject_subcategories'])

# Split the dataset back again

train_dataset = combined_dataset.iloc[:len(train_dataset)]
test_dataset = combined_dataset.iloc[len(train_dataset):]

# Removing stop words from text fields

stopwords_list = stopwords.words("english")

def stopwords_remove(text):
  word_list=[]
  for word in gensim.utils.simple_preprocess(text):
    if word not in stopwords_list:
      word_list.append(word)
  final = ' '.join(word_list)
  return final

train_dataset['project_text_all'] = train_dataset['project_text_all'].apply(lambda x: stopwords_remove(x))
test_dataset['project_text_all'] = test_dataset['project_text_all'].apply(lambda x: stopwords_remove(x))

# Remove punctuations from text fields

def remove_punctuations(text):
  for char in text:
    if char in string.punctuation:
      text = text.replace(char,'')
  return text

train_dataset['project_text_all'] = train_dataset['project_text_all'].apply(lambda x: remove_punctuations(x))
test_dataset['project_text_all'] = test_dataset['project_text_all'].apply(lambda x: remove_punctuations(x))

# Stemitize data

def stemitize_text(text):
  tokens = word_tokenize(text)
  stemmer = PorterStemmer()
  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  return ' '.join(stemmed_tokens)

train_dataset['project_text_all'] = train_dataset['project_text_all'].apply(lambda x: stemitize_text(x))
test_dataset['project_text_all'] = test_dataset['project_text_all'].apply(lambda x: stemitize_text(x))

# Lemitize data

def lemitize_text(text):
  tokens = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
  return ' '.join(lemmatized_tokens)

train_dataset['project_text_all'] = train_dataset['project_text_all'].apply(lambda x: lemitize_text(x))
test_dataset['project_text_all'] = test_dataset['project_text_all'].apply(lambda x: lemitize_text(x))

# TDIDF Vectorization on text fields 

tfidf_vectorizer_text = TfidfVectorizer(max_features=1000)

tfidf_text = tfidf_vectorizer_text.fit_transform(train_dataset['project_text_all'])
tfidf_text = pd.DataFrame(tfidf_text.toarray(), columns=tfidf_vectorizer_text.get_feature_names_out())

train_vectorized = pd.concat([train_dataset.reset_index(drop=True), tfidf_text.reset_index(drop=True)], axis=1)

train_dataset = train_vectorized.drop(columns=['project_text_all'])

tfidf_text = tfidf_vectorizer_text.transform(test_dataset['project_text_all'])
tfidf_text = pd.DataFrame(tfidf_text.toarray(), columns=tfidf_vectorizer_text.get_feature_names_out())

test_vectorized = pd.concat([test_dataset.reset_index(drop=True), tfidf_text.reset_index(drop=True)], axis=1)

test_dataset = test_vectorized.drop(columns=['project_text_all'])

# Scaling on both datasets

scaler = StandardScaler()
train_secured_dataset  = scaler.fit_transform(train_dataset)
train_secured_dataset = pd.DataFrame(train_secured_dataset, columns=train_dataset.columns)

test_secured_dataset  = scaler.transform(test_dataset)
test_secured_dataset = pd.DataFrame(test_secured_dataset, columns=test_dataset.columns)

# Calculate class weights for training dataset

all_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight = 'balanced', classes = all_classes, y = y_train)
class_weights = dict(zip(all_classes, class_weights))

# Perform RandomForestClassifier model

rf_model = RandomForestClassifier(n_estimators = 1000, max_depth = 20, class_weight = class_weights)
rf_model.fit(train_secured_dataset, y_train)

# Predictions on test dataset

y_pred = rf_model.predict(test_secured_dataset)

predictions_df = pd.DataFrame({'predicted_label': y_pred})
predictions_df.to_csv('predict.csv', index=False)