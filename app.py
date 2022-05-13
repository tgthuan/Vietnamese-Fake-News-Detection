# import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import sklearn
import pickle
import copy
import re
import sys
import vncorenlp
from vncorenlp import VnCoreNLP
import nltk
from nltk import tokenize
import streamlit as st
import os



def set_page_config():
    #set page config
    st.set_page_config(
    page_title="Fake news detection",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
            'About': "# TEAM Năm chàng lính ngự lâm"
    }
    )

@st.cache
def NoiseDefuse(s):
    result = copy.copy(s)
    result = result.str.lower()
    result = result.apply(lambda x: re.sub(r'http\S+', '', x))
    result = result.apply(lambda x: x.replace('\n',' '))
    result = result.apply(lambda x: re.sub('[^aàảãáạăằẳẵắặâầẩẫấậ b c dđeèẻẽéẹêềểễếệ f g hiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstu ùủũúụưừửữứựvwxyỳỷỹýỵz +[0-9]+', '', x))
    return result

@st.cache
def reduce_dim(x):
    return x[0]

@st.cache
def TokenNize(s):
    return s.apply(st.session_state.annotator.tokenize).apply(reduce_dim)

@st.cache 
def normalized1(x):
    contractions={
        ' cđv': ' cổ động viên',
        ' thcs': ' trung học cơ sở',
        ' pgs': ' phó giáo sư ',
        ' gs': ' giáo sư ',
        ' ts': ' tiến sĩ ',
        ' gd  đt': ' giáo dục - đào tạo',
        ' gd đt': ' giáo dục - đào tạo',
        ' gdđt': ' giáo dục - đào tạo',
        ' hlv': ' huấn luyện viên',
        ' tp': ' thành phố ',
        ' hcm': ' Hồ Chí Minh ',
        ' đt': ' đội tuyển ',
        ' gd': ' giáo dục '
    }
    for k,v in contractions.items():
        x=x.replace(k,v)
    return x

@st.cache
def normalized(s):
    return s.apply(normalized1)

@st.cache
def remove_stopword(list_word):
    clean_list = []
    for i in range(len(list_word)):
        temp=list_word[i].replace('_',' ')
        if temp not in stopwords :
            clean_list.append(list_word[i])
    return clean_list

@st.cache
def Preprocess(s):
    a= TokenNize(normalized(NoiseDefuse(s)))
    a=a.apply(remove_stopword)
    return a

@st.cache
def fullPreprocess(s):
    return Preprocess(s).apply(lambda x:" ".join(x))


def choose_select():
    choose_model = st.selectbox('Choose model',('Logistic Regression','Random Forest Classifier' ,'SVC'))
                                                   
    text = st.text_input('Input news')
    choose_predict = st.button('Predict')
    return choose_model, text, choose_predict


def fake_news_dectection_page():
    left,mid,right = st.columns([55,1,44])
    with left:
        result = -1
        #button
        choose_model, text, choose_predict = choose_select()           
        if choose_predict:
            if choose_model == 'Logistic Regression':
                with open('models_saved/LogisticRegression.pkl','rb') as f:
                    if text != '':
                        text_preprocessed = fullPreprocess(pd.Series([text]))
                        loaded_model = pickle.load(f)
                        result = loaded_model.predict(text_preprocessed)
            elif choose_model == 'Random Forest Classifier':
                with open('models_saved/RandomForestClassifier.pkl','rb') as f:
                    if text != '':
                        text_preprocessed = fullPreprocess(pd.Series([text]))
                        loaded_model = pickle.load(f)
                        result = loaded_model.predict(text_preprocessed)
            elif choose_model == 'SVC':
                with open('models_saved/SVC.pkl','rb') as f:
                    if text != '':
                        text_preprocessed = fullPreprocess(pd.Series([text]))
                        loaded_model = pickle.load(f)
                        result = loaded_model.predict(text_preprocessed)
                        
            if result != -1:
                my_bar = st.progress(0)
                for p in range(100):
                    time.sleep(0.01)    
                    my_bar.progress(p+1)
                if result == 0:
                    st.success('Real news')
                elif result == 1:
                    st.error('Fake news')
            else:
                st.write('Please input news cho predidct')
                
    with right:
        with Image.open('image/app_image.png') as image_wc:
            st.image(image_wc, use_column_width='auto')

    
def home_page():
    data_member = {'ID':['19120212','19120297','19120328','19120389','19120602'],
                        'Name':['Vũ Công Duy','Đoàn Việt Nam','Võ Trọng Phú','Tô Gia Thuận','Hồ Hữu Ngọc']}
    st.subheader('ABOUT US')
    member_df = pd.DataFrame(data_member)
    st.table(member_df)


def main():
    left,mid,right = st.columns([1,1,10])
    with left:
        with Image.open('image/app_logo.jpg') as image_logo:
            st.image(image_logo, use_column_width='auto')
    with right:
        st.title('FAKE NEWS DETECTION')

    fake_news_dectection_page()
    if st.button('About Us'):
        home_page()


@st.cache(allow_output_mutation=True)
def init():
    #read lib
    #stopword
    f = open('resources/stopwords.txt', 'r', encoding='UTF-8')
    stopwords = f.read().split('\n')
    #java library
    annotator = VnCoreNLP('resources/VnCoreNLP-1.1.1.jar', annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')
    # annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
    return stopwords, annotator

set_page_config()
stopwords, st.session_state['annotator'] = init()
main() 


    