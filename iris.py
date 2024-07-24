import streamlit as st
import pandas as pd
from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Application pour la prévision des fleurs d'iris
Cette application prédit la catégorie des fleurs.
''')
st.image("iris.PNG",width=800)
st.sidebar.header("Les paramètres d'entrée") 

#===============================================================================================================

def user_input():
    sepal_length = st.sidebar.slider('Entrer la longueur du sepal:',4.3,10.0,0.0)
    sepal_width = st.sidebar.slider('Entrer la largeur du sepal:',2.0,10.0,0.0)
    petal_length = st.sidebar.slider('Entrer la longueur du petal:',1.0,10.0,0.0)
    petal_width = st.sidebar.slider('Entrer la largeur du petal:',0.1,10.0,0.0)
    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
    fleur_parametre = pd.DataFrame(data,index=[0])
    return fleur_parametre

#===============================================================================================================

st.subheader('On veut trouver la catégorie de cette fleur.')
df = user_input()
st.write(df)

iris = datasets.load_iris()
foret=RandomForestClassifier()  
foret.fit(iris.data, iris.target)  

#===============================================================================================================

if(st.button("▶️")):
    prediction = foret.predict(df)
    st.write("La catégorie de la fleur est:")
    st.success(iris.target_names[prediction])
    
