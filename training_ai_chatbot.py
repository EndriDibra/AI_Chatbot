# Author: Endri Dibra

# using the required libraries
import json
import pickle
import random
import numpy as np

# nltk library
import nltk
from nltk.stem import WordNetLemmatizer

# tensorflow - keras libraries
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# our word lemmatizer
lmtzr = WordNetLemmatizer()

# list of words from json file
words = []

# a list for intents
classes = []

# a list for intents and their patterns
documents = []

# non to be considered letters
ignore_letters = [',', '.', '!', '?']

# opening data used for our chatbot
intents_file = open('data.json').read()

# loading data
intents = json.loads(intents_file)


# traversing every intent in our json file
# and their patterns (possible inputs)

for intent in intents['intents']:

    for pattern in intent['patterns']:

        # tokenizing each word in the json file
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # adding documents in the corpus
        documents.append((word, intent['tag']))

        # adding intents (json) to our classes list
        if intent['tag'] not in classes:

            classes.append(intent['tag'])

# output: all the words from the json file
print(documents)


# lowering, lemmatizing each word and removing all duplicates
# that may exist in our json file
words = [lmtzr.lemmatize(wrd.lower()) for wrd in words if wrd not in ignore_letters]
words = sorted(list(set(words)))

# sorting classes
classes = sorted(list(set(classes)))

# documents (list): is consisted by all intents and their patterns
print(len(documents), "documents")

# classes (list): is consisted by intents
print(len(classes), "classes", classes)

# words (list): is consisted by all the words from json file
print(len(words), "unique words (lemmatized)", words)

# storing words data to words.pkl
pickle.dump(words, open('words.pkl', 'wb'))

# storing classes data to classes.pkl
pickle.dump(classes, open('classes.pkl', 'wb'))

# a list for training data
training = []

# an empty array for our output
output_arr = [0] * len(classes)


# training set, bag of words for each sentence
for document in documents:

    # initializing our bag of words list
    bag_of_words = []

    # list of tokenized words for the pattern
    pattern_words = document[0]

    # lemmatizing each word, creating base word, to represent related words
    pattern_words = [lmtzr.lemmatize(word.lower()) for word in pattern_words]

    # creating our bag of words array with 1, if word match found in current pattern
    for word in words:

        bag_of_words.append(1) if word in pattern_words else bag_of_words.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_arr)
    output_row[classes.index(document[1])] = 1

    training.append([bag_of_words, output_row])


# shuffling features and turning them into np.array
random.shuffle(training)
training = np.array(training)

# training creation and testing lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data have been created successfully")


# model creation (Sequential) of 3 layers.
# first layer has 128 neurons,
# second layer has 64 neurons and
# 3rd output layer contains number of neurons equal to
# number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# Compiling our model using Stochastic Gradient Descent (optimization algorithm)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=16, verbose=1)
model.save('model.h5', hist)

print("Model has been created successfully !!")