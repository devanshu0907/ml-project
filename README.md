import nltk
import numpy as np
import random
import string

f=open('conversationo.csv','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer # which is to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.metrics.pairwise import cosine_similarity #Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

from sklearn.naive_bayes import MultinomialNB

# ... existing code ...

# Train the Naive Bayes classifier (replace with your dataset and labels)
X_train = ['conversationo.csv']  # Training data (preprocessed text)
y_train = ['food.csv']  # Labels (corresponding categories/intents)
#clf = MultinomialNB().fit(X_train, y_train)

def respond(user_response, conversation_history):
    # ... existing code ...

    # Classify user intent using the trained model
    predicted_intent = clf.predict([preprocess_text(user_response)])[0]

    # Handle different intents based on the prediction
    if predicted_intent == "greeting":
        robo_response = greet(user_response)
    elif predicted_intent == "cafe_inquiry":
        # Retrieve relevant information from your knowledge base or dataset
        robo_response = "Here's information about the cafe item you asked about..."
    # ... add other intent handling logic ...
    else:
        # Use TF-IDF fallback for unclassified intents
        robo_response = response(user_response, conversation_history)

    return robo_response

# Optional libraries for specific algorithms (install as needed)
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes (Classification)
#from haystack.document_store import InMemoryDocumentStore
#from haystack.retriever import BM25Retriever  # BM25 (Information Retrieval)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense  # LSTM (Sequence-to-Sequence)


def preprocess_text(text):
    """Preprocesses text by converting to lowercase, removing punctuation, and lemmatizing."""
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    text = text.lower().translate(remove_punct_dict)
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in nltk.word_tokenize(text)]

def greet(user_response):
    """Handles greetings from the user."""
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad you're here!"]
    for word in user_response.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def respond(user_response, conversation_history, algorithm=None, **kwargs):
    """Generates a response using an appropriate algorithm (if specified)."""
    robo_response = ''

    # Preprocess user response and conversation history
    user_response_processed = preprocess_text(user_response)
    conversation_history_processed = []
    for sentence in conversation_history:
        conversation_history_processed.extend(preprocess_text(sentence))

    # Choose response generation method based on algorithm
    if algorithm == 'tfidf':
        # TF-IDF similarity fallback
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(conversation_history_processed + [user_response_processed])
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf == 0:
            robo_response = "I am sorry, I don't understand you."
        else:
            robo_response = conversation_history_processed[idx]

    elif algorithm == 'naive_bayes':
        # Classification with Naive Bayes (optional)
        # Requires a trained Naive Bayes model and labeled dataset (intent classification)
        if 'clf' in kwargs:
            predicted_intent = kwargs['clf'].predict([user_response_processed])[0]
            # Handle different intents based on prediction (implement logic here)
        else:
            print("Error: Naive Bayes requires a trained classifier (provide 'clf' argument)")
            return "Something went wrong. Using TF-IDF fallback..."

    elif algorithm == 'bm25':
        # Information Retrieval with BM25 (optional)
        # Requires a populated document store and BM25 retriever (knowledge base retrieval)
        if 'document_store' in kwargs and 'retriever' in kwargs:
            query = preprocess_text(user_response)
            retrieved_documents = kwargs['retriever'].retrieve(query=query, top_k=3)
            # Generate response based on retrieved documents (implement logic here)
        else:
            print('error')

flag=True
print("Cafebot: My name is Cafebot. I will answer your queries about Cafe. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Cafebot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Cafebot: "+greeting(user_response))
            else:
                print("Cafebot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")
