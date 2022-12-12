# Author: Endri Dibra

# importing required libraries
import json
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

# textblob library
from textblob import TextBlob as tb

# tensorflow - keras libraries
from tensorflow.keras.models import load_model

# nltk library
import nltk
from nltk.stem import WordNetLemmatizer

# flask library
from flask import Flask, render_template, request, jsonify

# ai voice assistant required libraries
import pyttsx3
import datetime


# AI voice assistant
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


# speaking ability for AI assistant
def speak(audio):
    engine.say(audio)
    engine.runAndWait()


# taking the current hour
# greeting the user for the first time

hour = int(datetime.datetime.now().hour)

if hour >= 0 and hour < 12:

    speak('Good Morning')

elif hour > 12 and hour < 18:

    speak('Good Afternoon')

else:

    speak('Good Evening')

speak('I am AIRUS, your artificial intelligence chatbot assistant.')


# creating the main frame for our app
app = Flask(__name__)


# html template (design) for our app
@app.get("/")
def home():

    return render_template("base.html")


@app.post("/predict")
def get_bot_response():

    userText = request.get_json().get('msg')
    response = ai_chatbot_response(userText)
    message = {"answer": response}
    return jsonify(message)


# our word lemmatizer
lmtzr = WordNetLemmatizer()

# loading our model (Sequential)
model = load_model("model.h5")

# reading and loading our dataset (json file)
intents = json.loads(open("data.json").read())

# loading words data
words = pickle.load(open("words.pkl", "rb"))

# loading classes data
classes = pickle.load(open("classes.pkl", "rb"))

# a list to store user's texts for sentiment analysis
texts = []

# declaring counters for sentiment analysis
pos= 0
neg= 0
neu= 0


# processing and organizing properly words
def cleaning_sentence(sentence):

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lmtzr.lemmatize(word.lower()) for word in sentence_words]

    # tokenizing sentences
    tokenize_sentence = sentence.split()

    # adding user's texts into the list
    texts.extend(tokenize_sentence)

    return sentence_words


# return bag of words array: 0 or 1 for words
# that exist in sentence

def bag_of_words(s, words_a, show_details=True):

    # tokenizing patterns
    sentence_words = cleaning_sentence(s)

    # initializing our bag of words list
    bag = [0] * len(words_a)

    for sentence in sentence_words:

        for i, word in enumerate(words_a):

            if word == sentence:

                # assign 1 if current word is in the vocabulary position
                bag[i] = 1

                if show_details:

                    print("found in bag: %s" % word)

    return np.array(bag)


def predict_fun(sentence, model):

    # filter out predictions below a threshold
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:

        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


# getting responses
def get_response(ints, intents_json):

    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:

        if i["tag"] == tag:

            result = random.choice(i["responses"])

            # put here your tone : playing tone.wav file

            break

    return result


# initializing chatbot responder
def ai_chatbot_response(msg):

    ints = predict_fun(msg, model)
    res = get_response(ints, intents)

    return res


# informing about the htlm link (site)
speak("Please click the link underlined in blue")


# running app
if __name__ == "__main__":

    app.run()


    # saluting user
    speak("It was great to chat with you! Hope to see you again.")


    # starting to process data (user's texts) for sentiment analysis
    speak("Sentiment analysis protocol activated. Please wait.")


    for i in texts:

        if tb(i).sentiment.polarity > 0:

            pos += 1

        elif tb(i).sentiment.polarity < 0:

            neg += 1

        else:

            neu += 1


    # percentages
    a= (pos/len(texts)) * 100
    b= (neg/len(texts)) * 100
    c= (neu/len(texts)) * 100

    # giving the final results (percentages) for sentiment analysis
    speak(f"The percentage of positive sentiment is {round(a,2)}  percent")

    speak(f"The percentage of negative sentiment is {round(b,2)} percent")

    speak(f"The percentage of neutral sentiment is {round(c,2)} percent")

    speak("Now you will see a graphic representation of the results")

    # illustrating the results
    statistics = np.array([ (pos/len(texts)) * 100, (neg/len(texts)) * 100, (neu/len(texts)) * 100 ])
    stat_labels = ["Positive", "Negative", "Neutral"]

    plt.pie(statistics, labels= stat_labels, autopct= "%1.2f%%", shadow= True, startangle= 90)
    plt.legend(title= "User's texts sentiment")
    plt.axis("equal")
    plt.show()

    # saluting
    speak("Have a nice day. Enjoy your time.")