import streamlit as st
from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Chargement du dataset Iris et Random Forest Classifier
iris = datasets.load_iris()
x = iris.data
y = iris.target
foret=RandomForestClassifier()  
foret.fit(x,y)

# Creation de l'application de prediction titree et en tete
st.title("Application de prévision des fleurs d'iris")
st.header("Cette application predit la catégorie des fleurs d'iris")

# saisir les longueurs et largeurs de sepal et petal à laide de la fonction Slider() de streamlit
st.sidebar.header("Les parametres d'entrée des fleurs d'iris")
sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.3)
sepal_width = st.sidebar.slider('Sepal width',2.0,4.4,3.3)
petal_length = st.sidebar.slider('petal length',1.0,6.9,2.3)
petal_width = st.sidebar.slider('petal width',0.1,2.5,1.3)
st.sidebar.write('This App is creat by MP_NDIAYE')

data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
df = pd.DataFrame(data,index=[0])

# Affichage des parametres saisi par l'utilisateur dans un subheader cad paragraphe
st.subheader('On veut trouver la catégorie de cette fleur')
st.write(df)

# Definition du boutton de prediction et afichage du valeur de fleur predit
if(st.button("Submit")):
    prediction = foret.predict(df)
    st.write("la catégorie de la fleur d'iris est:",iris.target_names[prediction])