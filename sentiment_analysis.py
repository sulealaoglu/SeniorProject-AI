# import libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import nltk #stop wordlerden kurtulmak için kullanıldı.
nltk.download('stopwords')
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import nltk
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(['seaborn-notebook'])
from sklearn.decomposition import _pca
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from spacy.lang.tr import Turkish
import jpype
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import re


def getWordWeights():
    stn = pd.read_excel("./STN.xlsx")
    stn = stn.drop_duplicates(['synonyms']).set_index('synonyms')
    for words in stn.index:
        if words in stn.index:
            if words is np.nan:
                continue
            for word in words.split(','):
                final_stn[word.strip()] = {'pos':stn.loc[words]['pos value'],'neg' : stn.loc[words]['neg value'] }

def Lemmatization(sentence):
    analysis: java.util.ArrayList = (morphology.analyzeAndDisambiguate(sentence).bestAnalysis())
    token = sentence.split() #Tokenization yapÄ±lÄ±r.
    pos=[]
    for index, i in enumerate(analysis):   
        if str(i.getLemmas()[0])=="UNK": #Kelime kÃ¶kÃ¼nÃ¼n bulamamasÄ± durumu.
            pos.append(token[index]) 
        else:
            pos.append(str(i.getLemmas()[0])) #Kelime kÃ¶kÃ¼ listeye eklenir.
        #print("lemma:")
        #print(str(i.getLemmas()[0]))  
    return pos 

def spellChecker(tokens):
    
    for index,token in enumerate(tokens):    
         
        #yazÄ±m yanlÄ±sÄ± varsa if'e girer
        if not spell_checker.check(JString(token)):
         
            if spell_checker.suggestForWord(JString(token)):
             
                  #kelimenin doÄŸru halini dÃ¶ndÃ¼rÃ¼r.
                  tokens[index] = spell_checker.suggestForWord(JString(token))[0]
                  #print((spell_checker.suggestForWord(JString(token))[0]))
     
    #Java liste yapÄ±sÄ± listeye eklenerek dÃ¼zeltilir. 
    corrected = [str(i) for i in tokens]
              
    return " ".join(corrected)

def feature_extraction(text):
    pos_val = 0
    neg_val = 0
    weight_list = []
    for token in text:
        word = token.lower()
        if word in final_stn:
            current_pos = final_stn[word]['pos']
            pos_val += current_pos
            current_neg = final_stn[word]['neg']
            neg_val +=  current_neg
            weight_list.append({'word': word, 'pos': current_pos, 'neg': current_neg})
    result = "Neutral"
    if pos_val > neg_val:
        result = "Positive"
    elif pos_val < neg_val:
        result = "Negative"
    
    response = {'words': weight_list, 'result': result}
    return response
   

def remove_stopword(tokens): 
    filtered_tokens = [token for token in tokens if token not in stop_word_list]#stop word'lerden temizlenir.  
    return filtered_tokens
    
def sentiment_analysis(sentence):
    filtered = remove_stopword(sentence.split())
    corrected = spellChecker(filtered)
    stems = Lemmatization(corrected)
    return stems

app = Flask(__name__)
CORS(app)

final_stn = {}
stop_word_list = []
stop_word_list = nltk.corpus.stopwords.words('turkish')    
stop_word_list.extend(["bir","kadar","sonra","kere","mi","ye","te","ta","nun","daki","nın","ten"])

ZEMBEREK_PATH ='./zemberek-full.jar' 
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()
TurkishSpellChecker: JClass = JClass('zemberek.normalization.TurkishSpellChecker')
spell_checker: TurkishSpellChecker  = TurkishSpellChecker(morphology)

df = pd.read_excel("TrkceTwit.xlsx")
print(df["Duygu"].value_counts())
y = df["Duygu"].values.tolist()

#etiketi 0 olan yani nötr veriler kaldırıldı.
for index,i in enumerate(y):
    if i==0:
        df.drop(index,axis=0, inplace=True)
df.reset_index(drop=True,inplace=True)

#print(df["Duygu"].value_counts())

x = df.Tweets
y = df.Duygu


X = []
for i in x:
    #text = i.lstrip(" ") #Cümle basindaki bosluklar kaldirilir.
    text = re.sub('\W+', ' ', str(i))
    text = text.replace('I','ı') #lower() yapıldığı zaman I harfi i olarak çevirildiğinden dolayı replace ile düzeltildi.
    text = text.replace('İ','i') 
    text = text.lower()
    text = " ".join([j for j in text.split() if j not in stop_word_list])
    text = " ".join([i for i in text.split() if len(i)>1])
    text = spellChecker(text.split())
    lemmaList = Lemmatization(text)
    text = " ".join(lemmaList)
    X.append(text)
    

Tfidf_Vector = TfidfVectorizer(max_features = 3000)

Tfidf_Matrix = Tfidf_Vector.fit_transform(X)

Xl = Tfidf_Matrix.toarray()

features = Tfidf_Vector.get_feature_names_out()
# Etiketleri yeniden kodla
y = y.replace({1: 0, 2: 1})


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xl, y, test_size=0.33,random_state = 42)

from xgboost import XGBClassifier

XGBModel = XGBClassifier()
XGBModel.fit(x_train, y_train)
y_pred = XGBModel.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Xgboost Classifier Accuracy: "+str(accuracy))


from sklearn.naive_bayes import MultinomialNB

multi_model = MultinomialNB()
multi_model.fit(x_train, y_train)
y_pred2 = multi_model.predict(x_test)
print("MultinomialNB Accuracy : "+ str(accuracy_score(y_test,y_pred2)))

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis_api():
    example = request.data.decode('utf-8')
    text = example.lstrip(" ")  # Remove leading spaces
    text = re.sub('\W+', ' ', str(text))
    text = text.replace('I', 'ı')  # Lowercase conversion adjustments
    text = text.replace('İ', 'i')
    text = text.lower()
    text = " ".join([j for j in text.split() if j not in stop_word_list])
    text = " ".join([i for i in text.split() if len(i) > 1])
    text = spellChecker(text.split())
    lemmaList = Lemmatization(text)
    processed_text = " ".join(lemmaList)

    transformed_text = Tfidf_Vector.transform([processed_text])
    y_pred = model.predict(transformed_text)

    response = {
        'words': example,
        'result': int(y_pred[0])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


print(feature_extraction(stems))