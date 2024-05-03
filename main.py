import os

import pandas as pd
import streamlit as st

from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Titanic App",
)

model_path = 'rf_fitted.pkl'

age = st.sidebar.number_input("How old are you?", 0, 100, 30)
child = int(age <= 16)
family_size = st.sidebar.number_input(
    "How many family members are aboard the ship (including yourself)?",
    1, 20, 1,
)
pclass_select = st.sidebar.selectbox(
    "In which passenger class are you traveling?",
    (1, 2, 3),
)

sex = st.sidebar.selectbox("Are you male or female?", ("male", "female"), index=1)

embarked = st.sidebar.selectbox(
    "Which is your port of Embarkation?",
    ("Cherbourg", "Queenstown", "Southhampton"),
)

# create input DataFrame
inputDF = pd.DataFrame(
    {
        "Age": age,
        "child": child,
        "family_size": family_size,
        "Pclass_1": pclass_select == 1,
        "Pclass_2": pclass_select == 2,
        "Pclass_3": pclass_select == 3,
        "Sex_female": sex == 'female',
        "Sex_male": sex == 'male',
        "Embarked_C": embarked == 'Cherbourg',
        "Embarked_Q": embarked == 'Queenstown',
        "Embarked_S": embarked == 'Southhampton'
    },
    index=[0],
)

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model('rf_fitted.pkl')

preds = model.predict_proba(inputDF)[0, 1]
preds = round(preds * 100, 1)


st.image("imgs/titanic_in_color.png", use_column_width=True)
st.write(f"Your Survival Probability based on the information provided is: {preds}%")
