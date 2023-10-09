import streamlit as st 
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.title("🌨️Data prediction⛅")
df= pd.read_csv('weather.csv')
# Nous allons traiter les données manquents
df['Sunshine'].fillna(0, inplace=True)
df['WindGustDir'].fillna(0, inplace=True)
df['WindGustSpeed'].fillna(0, inplace=True)
df['WindSpeed9am'].fillna(0, inplace=True)
df['WindDir3pm'].fillna(0, inplace=True)
df['WindDir9am'].fillna(0, inplace=True)
# Nous allons supprimer les donnés inutiles 
df.drop('RISK_MM', inplace=True,axis=1)
df.drop('WindGustDir', inplace=True,axis=1)
df.drop('WindDir9am', inplace=True,axis=1)
df.drop('WindDir3pm', inplace=True,axis=1)
df.drop('RainToday', inplace=True,axis=1)

# separation des données 
X = df.drop('RainTomorrow',axis=1).values
y = df['RainTomorrow'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# Passons aux algorithmes prédicteurs svm

model = SVC()
model.fit(X_train, y_train)
previsor_svc = model.predict(X_test)

# les sidebar qui sont dans le Home page  
with st.sidebar:
    st.info('Data analysis and visualisation')
    st.markdown(''' 
    📧 [Météo Sénégal](<https://www.meteoart.com/africa/senegal?page=day#date=2023-07-18>)
                ''')
st.sidebar.info("SVM(Support Vector Machine) accuracy error")
st.sidebar.write('👍Accuracy:',np.round(accuracy_score(y_test,previsor_svc),2)*100,'%')
st.sidebar.write('👎Error:',100-np.round(accuracy_score(y_test,previsor_svc),2)*100,'%')
st.sidebar.header("🌐Data legend")
st.sidebar.success('Yes 🌨️🌨️🌨️')
st.sidebar.success('No ⛅⛅⛅')
st.sidebar.info('Made with 💗 by  MP')

# Fonction input user
def user_input(MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm):
    data = np.array([MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,
                     WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,
                     Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm])
    prediction_data = model.predict(data.reshape(1,-1))
    return prediction_data

# L'utilisateur saisie une valeur pour chaque caracteristique
st.info('You must be enter the values number')
MinTemp = st.number_input('MinTemp:',min_value=-1,value=50)
MaxTemp = st.number_input('MaxTemp:',value=50)
Rainfall = st.number_input('Rainfall:',value=40)
Evaporation = st.number_input('Evaporation:',value=15)
Sunshine = st.number_input('Sunshine:',value=15)
WindGustSpeed = st.number_input('WindGustSpeed:',value=1100)
WindSpeed9am = st.number_input('WindSpeed9am:',value=100)
WindSpeed3pm = st.number_input('WindSpeed3pm:',value=100)
Humidity9am = st.number_input('Humidity9am:',value=100)
Humidity3pm = st.number_input('Humidity3pm:',value=100)
Pressure9am = st.number_input('Pressure9am:',value=1500)
Pressure3pm = st.number_input('Pressure3pm:',value=1500)
Cloud9am = st.number_input('Cloud9am:',value=10)
Cloud3pm = st.number_input('Cloud3pm:',value=10)
Temp9am = st.number_input('Temp9am:',value=30)
Temp3pm = st.number_input('Temp3pm:',value=30)

# creation du bouton de prediction 
if st.button('▶️'):
    prediction = user_input(MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,
                            WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,
                            Temp9am,Temp3pm)
    st.success(prediction)
#Creation de bot
st.title('🤖Chatbot🤖')
def main():
    #st.write("Bonjour!Je suis un chatbot. Demandez-moi quoi que ce soit sur le sujet dans le fichier texte.")
    # Obtenir la question de l'utilisateur
    question = st.text_input("👨")
    reponse=""
    #Créer un bouton pour soumettre la question
    if st.button("🆗"):
        if question == 'bonjour':
             st.write("🤖",question)  
        elif question == 'donne moi la meteo':
            reponse = "renseignez les données"
            st.write("🤖",reponse)   
if __name__ == "__main__":
    main()
