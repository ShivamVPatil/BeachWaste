import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, original_encoder
from load_model import get_model

rf_model = get_model(model_path = r'Model/beach.pkl')


st.set_page_config(page_title="BeachWaste Prediction App",
                    layout="wide")


#creating option list for dropdown menu
options_season = ['Rainy','Non-Rainy']
options_festival = ['No Festival','Non-Rainy','Ganesh Chaturthi','Navratri','Diwali','Chat Pooja','Holika','Holi','Narali Pornima','Narali Pornima','Dahi Handi','New Year']





features = ['SEASON','FESTIVAL','DAY','MONTH']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        day_of_week =st.slider("Day: ", 1,7, value=0, format="%d") 
        month = st.slider("Month: ", 0, 12, value=0, format="%d")
        season = st.selectbox("Select Season: ", options=options_season)
        festival = st.selectbox("Select Festival: ", options = options_festival)
        
        
        submit = st.form_submit_button("Predict")



    if submit:
        season = original_encoder(season, options_season)
        festival = original_encoder(festival, options_festival)
        


        data = np.array([day_of_week,month,season,festival
                            ]).reshape(1,-1)
        print(data)
        pred = get_prediction(data=data, model=rf_model)

        st.write(f"The predicted weight is:  {pred[0]}")


if __name__ == '__main__':
    main()
