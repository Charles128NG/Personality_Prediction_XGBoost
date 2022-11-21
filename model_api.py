from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import data

StopWordsCache = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
Vectorizer = TfidfVectorizer(use_idf=True)
types=['infp','infj','intp','intj','entp','enfp','istp','isfp','entj','istj','enfj','isfj','estp','esfp','esfj', 'estj']

description_keywords = data.description_keywords
personality_traits={
    
    'ENFJ':"""ENFJs are idealist organizers, driven to implement their vision of what is best for humanity. They often act as catalysts for human growth because of their ability to see potential in other people and their charisma in persuading others to their ideas.""",

    'ENFP':"""ENFPs are people-centered creators with a focus on possibilities and a contagious enthusiasm for new ideas, people and activities. Energetic, warm, and passionate, ENFPs love to help other people explore their creative potential.""",

    'ENTJ':"""ENTJs are strategic leaders, motivated to organize change. They are quick to see inefficiency and conceptualize new solutions, and enjoy developing long-range plans to accomplish their vision. They excel at logical reasoning and are usually articulate and quick-witted.""",

    'ENTP':"""ENTPs are inspired innovators, motivated to find new solutions to intellectually challenging problems. They are curious and clever, and seek to comprehend the people, systems, and principles that surround them.""",

    'ESFJ':"""ESFJs are conscientious helpers, sensitive to the needs of others and energetically dedicated to their responsibilities. They are highly attuned to their emotional environment and attentive to both the feelings of others and the perception others have of them.""",

    'ESFP':"""ESFPs are vivacious entertainers who charm and engage those around them. They are spontaneous, energetic, and fun-loving, and take pleasure in the things around them: food, clothes, nature, animals, and especially people.""",

    'ESTJ':"""ESTJs are hardworking traditionalists, eager to take charge in organizing projects and people. Orderly, rule-abiding, and conscientious, ESTJs like to get things done, and tend to go about projects in a systematic, methodical way.""",

    'ESTP':"""ESTPs are energetic thrillseekers who are at their best when putting out fires, whether literal or metaphorical. They bring a sense of dynamic energy to their interactions with others and the world around them.""",

    'INFJ':"""INFJs are creative nurturers with a strong sense of personal integrity and a drive to help others realize their potential. Creative and dedicated, they have a talent for helping others with original solutions to their personal challenges.""",

    'INFP':"""INFPs are imaginative idealists, guided by their own core values and beliefs. To a Healer, possibilities are paramount; the reality of the moment is only of passing concern. They see potential for a better future, and pursue truth and meaning with their own flair.""",

    'INTJ':"""INTJs are analytical problem-solvers, eager to improve systems and processes with their innovative ideas. They have a talent for seeing possibilities for improvement, whether at work, at home, or in themselves.""",

    'INTP':"""INTPs are philosophical innovators, fascinated by logical analysis, systems, and design. They are preoccupied with theory, and search for the universal law behind everything they see. They want to understand the unifying themes of life, in all their complexity.""",

    'ISFJ':"""ISFJs are industrious caretakers, loyal to traditions and organizations. They are practical, compassionate, and caring, and are motivated to provide for others and protect them from the perils of life.""",

    'ISFP':"""SFPs are gentle caretakers who live in the present moment and enjoy their surroundings with cheerful, low-key enthusiasm. They are flexible and spontaneous, and like to go with the flow to enjoy what life has to offer.""",

    'ISTJ':"""ISTJs are responsible organizers, driven to create and enforce order within systems and institutions. They are neat and orderly, inside and out, and tend to have a procedure for everything they do.""",

    'ISTP':"""ISTPs are observant artisans with an understanding of mechanics and an interest in troubleshooting. They approach their environments with a flexible logic, looking for practical solutions to the problems at hand."""
}

file = open("trained_model_lemmatized","rb")
loaded_model = pickle.load(file)
file.close()


file = open("encodings","rb")
encodings = pickle.load(file)
file.close()
#encodings

def preprocess(dataset,lemmatize = False, stem=False, MBTI_remove=False):
    list_posts=[]
    for row in dataset.iterrows():
        posts = row[1].posts
        posts = posts.lower()
        posts = re.sub(r'https?:\/\/.*?[\s+]', ' ', posts)#remove hyperlinks
        posts = re.sub(r'http?:\/\/.*?[\s+]', ' ', posts)#remove hyperlinks
        posts = re.sub(r'[^\w\s]',' ',posts)
        posts = re.sub(r'[\d]','',posts)
        posts = re.sub(r'[_]','',posts)
        posts = re.sub(r'[\n]',' ',posts)
        posts = re.sub(r'[^a-zA-Z]',' ',posts)
        posts = re.sub(r'[rt]',' ',posts)
        if MBTI_remove:#remove mbti words
            for t in types:
                posts = posts.replace(t,"")
        if lemmatize:
            posts = " ".join([lemmatizer.lemmatize(w) for w in posts.split(' ') if w not in StopWordsCache])
        if stem:
            posts = " ".join([stemmer.stem(w) for w in posts.split(' ') if w not in StopWordsCache])
        list_posts.append(posts)
    dataset['posts'] = np.array(list_posts)    
    return dataset


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/')
def predict(data):
    #print(data)
    df = pd.DataFrame(data={'type':['INFP'],'posts':[data]})
    df = preprocess(df,lemmatize=True,MBTI_remove=True)
    ip_vect = Vectorizer.fit_transform(df['posts'])
    min_ = ip_vect.min()
    lst=ip_vect.shape[1]
    ip_vect.resize((1,89388))
    ip_v = ip_vect.toarray()
    i=lst+1
    while i<89388:
        ip_v[0][i]=min_
        i+=5
    prediction = loaded_model.predict(ip_v)
    return {"mbti_type":encodings[prediction[0]],"description":description_keywords[encodings[prediction[0]]],"traits":personality_traits[encodings[prediction[0]]]}
