# -*- coding: utf-8 -*-

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import time
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json',  encoding="utf-8").read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokeniser le motif - diviser les mots en tableau
    sentence_words = nltk.word_tokenize(sentence)
    # racine chaque mot - créer une forme courte pour le mot
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# retour sac de mots tableau : 0 ou 1 pour chaque mot du sac qui existe dans la phrase

def bow(sentence, words, show_details=True):
    # tokeniser le motif
    sentence_words = clean_up_sentence(sentence)
    # sac de mots - matrice de N mots, matrice de vocabulaire
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # attribuez 1 si le mot actuel est dans la position du vocabulaire
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filtrer les prédictions en dessous d'un seuil
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # trier par force de probabilité
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            result = result.replace("\n", "<br>")
            words_to_bold = ["Votre Ixina se situe au", "Téléphone ", "Mail", "Horaires d'ouverture", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            for word in words_to_bold:
                result = result.replace(word, "<b>" + word + "</b>")
            break
    return result

def chatbot_response(message):
    try:
        time.sleep(0.5)
        ints = predict_class(message, model)
        res = getResponse(ints, intents)
    except Exception:
            res = "Désolé, je n'ai pas de réponse à ce sujet"
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('message')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()