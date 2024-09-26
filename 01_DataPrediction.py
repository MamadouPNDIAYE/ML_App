import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

#=========================================================================================================

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="MathsPratique",
    page_icon="image/MathsPratique.ico"
    )

#=========================================================================================================

selected = option_menu(
        menu_title = None,  # required
        options = ["Home","Data","Prediction"],   # required
        orientation ="horizontal",
        icons = ["house","book","envelope"],
        menu_icon = "cast",
        default_index = 0,
        styles = {"container":{"padding":"0!important","background-color":"gray"},
                     "icon":{"color":"white","font-size":"15px"},
                     "nav-link":{"font-size":"15px","text-align":"center","margin":"0px","--hover-color":"#eee"},
                     "nav-link-selected":{"background-color":"blue"}}
        )
    
#=========================================================================================================

 # Afficher l'image
st.sidebar.image("image/climat.webp", caption="Climat et météo")

df= pd.read_csv('dataset/weather.csv')
  # Nous allons traiter les données manquents
df['Sunshine'].fillna(0, inplace=True)
df['WindSpeed'].fillna(0, inplace=True)

encodage = LabelEncoder()
df['Rain'] = encodage.fit_transform(df['Rain'])

# separation des données 
from sklearn.model_selection import train_test_split
X = df.drop('Rain',axis=1).values
y = df['Rain'].values

# standardiser nos informations, car elles sont à des échelles très différentes
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X = minmax.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

#=========================================================================================================

if selected =="Home":
    
    if st.sidebar.toggle("A propos MathsPratique", True):
        with st.expander(":blue[Bienvenue dans MathsPratique]", True) : 
            colonne_1 , colonne_2 = st.columns([1,2])
        with colonne_1 :
            st.image("image/Logomp.jpg")
        with colonne_2 : 
            st.markdown("""
                        Nous sommes Team MP, notre expertise réside dans les domaines de la science des données et de l'intelligence artificielle.
                        Avec une approche technique et concise, nous nous engageons à fournir des solutions efficaces et précises dans nos projets.
                        """)
            
            st.markdown("""
            ## Contacts
            * 📧  [MathsPratique](<mathspratique.mp@gmail.com>)
            * 📧  +221 77 756 10 43  """)
            
    st.subheader("**:blue[Application de prévision métrologique]**",divider='rainbow')    
       
    st.markdown("""
        ### Météorologie 🌨️⛅ 
                
        La météorologie est une science qui a pour objet l'étude des phénomènes atmosphériques tels que
        les nuages, les précipitations ou le vent dans le but de comprendre comment ils se forment et évoluent en fonction
        des paramètres mesurés notamment
        **la pression**,
        **la température**,
        **l'humidité**.
      
        C'est une discipline qui traite principalement de la mécanique des fluides et de la thermodynamique 
        mais qui fait usage de différentes autres branches de la physique, de la chimie et des mathématiques.
        Purement descriptive à l'origine, la météorologie est devenue un lieu d'application de ces disciplines.
        
        ### Météorologie moderne 🌨️⛅
        La météorologie moderne permet d'établir des prévisions de l'évolution du temps
        en s'appuyant sur des modèles mathématiques à court comme à long terme qui assimilent des données 
        de nombreuses sources dont les stations, les satellites et les radars météorologiques.
        La météorologie a des applications dans des domaines très divers comme les besoins militaires,
        la production d'énergie, les transports (aériens, maritimes et terrestres), l'agriculture,  la construction, la photographie aérienne ou le cinéma.
        Elle est également appliquée pour la prévision de la qualité de l'air ou de plusieurs risques naturels d'origine atmosphérique.
            """)
    
 #=========================================================================================================
 
if selected =="Data":
    
    st.subheader("**:blue[Application de prévision métrologique]**",divider='rainbow') 
    
    st.write("Le dataset de formation du modèle : ")
    st.write(df)
    
    st.header("Collecter des données au format CSV")
    
    # Création du formulaire avec le widget Form de streamlit 
    with st.form("user_input_form"):
        st.write("Entrez vos réponses:")
        name = st.text_input("Name")
        evaporation = st.number_input("Evaporation")
        sunshine = st.number_input("Sunshine")
        windSpeed = st.number_input("WindSpeed")
        humidity = st.number_input("Humidity")
        pressure = st.number_input("Pressure")
        cloud = st.number_input("Cloud")
        temp = st.number_input("Temp")
        submit_button = st.form_submit_button("Submit")
        
    # Ici nous crée le fichier csv pour stocke les données 
    csv_file_path = "user_answers.csv"
    data = pd.DataFrame(columns=["Name","Evaporation","Sunshine","WindSpeed","Humidity","Pressure","Cloud","Temp"])
    
    # Verification du fichier csv_file_path s'il existe puis lecture avec le panda
    if os.path.exists(csv_file_path):
        data = pd.read_csv(csv_file_path)
    
    # Creation du boutton d'evoie une fois un clic sur le boutton le message suiavant 
    # s'affiche Answers submitted successfully! qui montre l'envoi dans le fichier csv a reussi 
    
    if submit_button:
        new_row = {"Name":name,"Evaporation":evaporation,"Sunshine": sunshine,"WindSpeed":windSpeed,"Humidity":humidity,"Pressure":pressure,"Cloud":cloud,"Temp":temp}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        data.to_csv(csv_file_path, index=False)
        st.success("Answers submitted successfully!")
    
    st.write("Données actuelles :")
    st.write(data)
    
    # Création du boutton telechargement une fois les données bien rmpouie le fichier peut étre telecharger 
    # qui contient les donnés enregistre 
    st.download_button(
        label="Download CSV File",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="user_answers.csv",
        mime="text/csv",)
    

    with st.sidebar:
        option_data_viz = st.selectbox("Option", ["Choisissez", "Graphe1", "Graphe2"])
        
        if option_data_viz == "Graphe1":
            st.write('Graphique de corrélation sur les données')
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(),annot=True,cbar=True)
            st.pyplot(fig)
            
        elif option_data_viz == "Graphe2":
            st.write("Graphique de corrélation sur les données")
            fig, ax =plt.subplots(1,2,figsize= (15,6))
            sns.scatterplot(df['Temp'],ax=ax[0])
            sns.countplot(df['Evaporation'],ax=ax[1])
            st.pyplot(fig)
    
#=========================================================================================================

if selected =="Prediction":
    
    with st.sidebar:
        st.markdown("""
                   
                    📧 [Openwather](<https://openweathermap.org/>)
      
                    📧 [MétéoSénégal](<https://www.meteoart.com/africa/senegal?page=day#date=2023-07-18>)
            """)
        
    st.subheader("**:blue[Application de prévision métrologique]**",divider='rainbow')
    
# Passons aux algorithmes prédicteurs svm
    model_svm = SVC()
    model_svm.fit(X_train, y_train)
    prediction_svc = model_svm.predict(X_test)
    Accuracy = (np.round(accuracy_score(y_test,prediction_svc),2)*100,'%')
 
# Passons aux algorithmes prédicteurs xgboost
    # model_xgboost = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    # model_xgboost.fit(X_train,y_train)
    # prediction_xgboost = model_xgboost.predict(X_test)
    # st.sidebar.write('Accuracy:', np.round(accuracy_score(y_test,prediction_xgboost),3)*100,'%')

    # Fonction input user
    def user_input(Evaporation,Sunshine,WindSpeed,Humidity,Pressure,Cloud,Temp):
        data = np.array([Evaporation,Sunshine,WindSpeed,Humidity,Pressure,Cloud,Temp])
        prediction_data = model_svm.predict(data.reshape(1,-1))
        return prediction_data

    # L'utilisateur saisie une valeur pour chaque caracteristique
    st.write('Entrez les données :')
  
    Evaporation = st.number_input("Evaporation")
    Sunshine = st.number_input("Sunshine")
    WindSpeed = st.number_input("WindSpeed")
    Humidity = st.number_input("Humidity")
    Pressure = st.number_input("Pressure")
    Cloud = st.number_input("Cloud")
    Temp = st.number_input("Temp")

    # creation du bouton de prediction 
    if st.button('▶️'):
        prediction = user_input(Evaporation,Sunshine,WindSpeed,Humidity,Pressure,Cloud,Temp)
        if prediction == 0:
            st.title("🤖")  
            st.write("Non ⛅⛅⛅")
        else:
            st.title("🤖")  
            st.write("Oui 🌨️🌨️🌨️")
        
    #=======================================================================================================
