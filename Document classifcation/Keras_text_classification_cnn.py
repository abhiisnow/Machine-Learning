# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:50:23 2019

@author: z00449fn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:09:20 2019

@author: z00449fn
"""

# Import necessary modules
import pandas as pd
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from keras.initializers import Constant
from datetime import datetime
#from gensim.corpora.dictionary import Dictionary

#tensorflow must be installed as a backend for keras

wd=os.getcwd()

data=pd.read_excel( wd + '\\input\\labeled_data_w._path-full-text-input.xlsx')


#%% preparing test and train data

data = data.rename(columns = {'doc_class':'label'})

X=data.loc[:, data.columns != 'label']
target=data['label']

#splitting data into train and test set
train, test, target_train, target_test = train_test_split(X, target, test_size=0.33, random_state=53)

# Create and fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.file_data)

#extract vocab size used as diemension for embeddings layer
vocab_size = len(tokenizer.word_index) + 1

# Prepare the data
X_train = tokenizer.texts_to_sequences(train.file_data)
X_train = pad_sequences(X_train, maxlen=5000)

X_test = tokenizer.texts_to_sequences(test.file_data)
X_test = pad_sequences(X_test, maxlen=5000)


# Get the numerical ids of column label
numerical_ids_train = target_train.astype('category').cat.codes
numerical_ids_test = target_test.astype('category').cat.codes

# One-hot encode the indexes
y_train = to_categorical(numerical_ids_train)
y_test = to_categorical(numerical_ids_test)



#%% preprocessing transfer learning with word2vec (not implemented in neural network)
#HINT: might be useful for feature extraction 
# =============================================================================
# nltk.download('wordnet')
# data_as_list=data.file_data.tolist()
# one_string=t=''.join(data_as_list)
# 
# tokens = word_tokenize(one_string)
# 
# #no stopwords
# stop_words = set(stopwords.words())
# 
# lower_tokens = [t.lower() for t in tokens]
# no_stops = [w for w in lower_tokens if not w in stop_words]
# no_stops = [w for w in no_stops if w not in  ['the', 'die', 'sie','wurde']]
# # Retain alphabetic words: alpha_only
# alpha_only = [t for t in no_stops if t.isalpha()]
# 
# # Instantiate the WordNetLemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()
# 
# # Lemmatize all tokens into a new list: lemmatized
# lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in alpha_only]
# 
# bow_simple = Counter(alpha_only)
# 
# # Print the 10 most common tokens
# most_common=(bow_simple.most_common(20))
# 
# #split list into chunks because word2vec takes a list of tokenized sentences as input)
# chunks = [alpha_only[x:x+100] for x in range(0, len(alpha_only), 100)]
# 
# w2v_model = word2vec.Word2Vec(chunks, size=300,window=5, iter=100)
# 
# #save the model so we can continue training with the loaded model
# w2v_model.save(wd +'\\models\\word2vecmodel')
# 
# =============================================================================

#%% this section is not necessary for predictions, but helpful for analysis. Shows most similar words in context of transfer learning. 
#For Embeddings initializer we used glove word vectors instead

w2v_model = word2vec.Word2Vec.load(wd +'\\models\\word2vecmodel')

# Get top 3 similar words to "..."
print(w2v_model.wv.most_similar(["bedienungsanleitung"], topn=5))
print(w2v_model.wv.most_similar(["sicherheit"], topn=5))
print(w2v_model.wv.most_similar(["profibus"], topn=5))

matrix=w2v_model.wv.vectors


#%%  
''' word embedding (transfer learning) with glove '''
# in literature glove pretrained word vectors seemed to be most used library to create embedding matrix, in context of Keras Embedding Layer
# load the whole embedding into memory (here we use vectors of dimension 200)
# for the glove vectors please refer to https://nlp.stanford.edu/projects/glove/
embeddings_index = dict()
f = open(wd + '\\input\\glove.6B\\glove.6B.200d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 200))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector




#%% keras model setup and train

# Create a model with embeddings
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim=200, trainable=False, embeddings_initializer=Constant(embedding_matrix) ,input_length=5000))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

model.summary()

# Fit the model
early_stopping_monitor = EarlyStopping(patience=2)

model.fit(X_train, y_train, validation_split=0.2, epochs=20, 
          callbacks = [early_stopping_monitor])

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %f' % (accuracy*100))

model.save(wd + '\\models\\model_file(cnn).h5') 

#%% use model
model = load_model(wd + '\\models\\model_file(cnn).h5') 

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=['certificate','manual']))

report = classification_report(y_true, y_pred,  target_names=['certificate','manual'], output_dict=True)
class_report = pd.DataFrame(report).transpose()

y_pred=np.where(y_pred==0, 'Certificate', 'Manual')

output=test
output.loc[:,'true_label']=target_test
output.loc[:,'predicted_label']=y_pred
output.loc[:,'correctly_predicted']= np.where(output.loc[:,'predicted_label']==output.loc[:,'true_label'],'y','n')



# datetime object containing current date and time
now = datetime.now()
 
# dd/mm/YY H:M:S
dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")


class_report.to_excel(wd + '\\output\\classification_report_' + dt_string + '.xlsx', index=True)
output.to_excel(wd + '\\output\\predictions_' + dt_string + '.xlsx', index=False)


