from django.shortcuts import render


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
from django.shortcuts import render

# Load and train the model once when the server starts
data = pd.read_csv("languages_datasets.csv")
x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

def predict_language(text):
    transformed_input = cv.transform([text])
    return model.predict(transformed_input)[0]

def translate_text(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# View for the home page
def home(request):
    return render(request, 'index.html')

# View for language prediction
def predict(request):
    if request.method == 'POST':
        text = request.POST['text']
        predicted_lang = predict_language(text)
        return render(request, 'predict.html', {'text': text, 'predicted_lang': predicted_lang})
    return render(request, 'predict.html')


# View for language detection and translation




def translate(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        target_lang = request.POST.get('target_lang', 'en')
        predicted_lang = predict_language(text)
        translated_text = translate_text(text, target_lang)
        return render(request, 'translate.html', {
            'text': text,
            'target_lang': target_lang,
            'predicted_lang': predicted_lang,
            'translated_text': translated_text,
        })
    return render(request, 'translate.html')
