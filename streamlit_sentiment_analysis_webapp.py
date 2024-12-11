import streamlit as st
import pickle
import pandas as pd
import numpy as np
import spacy
nlp= spacy.load("en_core_web_lg")
svm_model_load= pickle.load(open("svm_model_saved.sav","rb"))
def clean_text(x):
    x=x.lower()
    doc4= nlp(x)
    clean_review= []
    for token in doc4:
        if token.is_punct or token.is_stop:
            continue
            
        clean_review.append(token.lemma_)
        
        
    return clean_review
def sentiment_pred(input_text):
    df2=pd.DataFrame([input_text],columns=['reviews'])
    df2['clean_review']=df2['reviews'].apply(lambda z: clean_text(z))
    df2['wordvec']=df2['clean_review'].apply(lambda z: nlp(str(z)).vector)
    val=df2['wordvec'].values
    val_2d=np.stack(val)

    prediction=svm_model_load.predict(val_2d)
    print(prediction)
    if prediction[0]== 1:
        return "Positive"
        
    else:
        return "Negative"
def main():
    st.title("Sentiment Analysis")
    text=st.text_input("Enter review")

    diag= ''

    if st.button('Sentiment Prediciton'):
        diag= sentiment_pred(text)

    st.success(diag)

if __name__=='__main__':
    main()    
