import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

model = joblib.load("extraTrees_finetuned.pkl")
columns = joblib.load("features.pkl")
scaler = joblib.load("featuresScaleMinMax.pkl")

# Title of the app
st.title("SCHOOL DROPOUT PREDICTION")


# Input fields for each feature
school_district = st.selectbox("School Region", [0, 1, 2], format_func=lambda x: ["Central", "North", "South"][x])



# child hiehest grade 3
highest_grade = st.selectbox(
    "Learner's Highest Grade Ever Attended", [1, 2, 3,4,5,6,7,8], format_func=lambda x: [ "STD 1", "STD 2", "STD 3","STD 4","STD 5","STD 6","STD 7","STD 8"][x-1]
)



# child's age 
if highest_grade <=2:
    min_age = 5
    max_age = 17
elif highest_grade ==3:
    min_age = 6
    max_age = 17
elif highest_grade ==4:
    min_age = 7
    max_age = 17
elif highest_grade ==5:
    min_age = 8
    max_age = 17
elif highest_grade == 6:
    min_age = 9
    max_age = 17
elif highest_grade == 7:
    min_age = 10
    max_age = 17
elif highest_grade == 8:
    min_age = 10
    max_age = 17
else:
    min_age = 5
    max_age = 17  


# Child's age slider with dynamic max value
child_age = st.slider("Child's Age", min_value=min_age, max_value=max_age, step=1)

#highest grade attended in 20
highest_grade_2019_20 = st.selectbox(
    "Learner Highest Grade Attended in 2019 - 2020", [0,1,2,3,4,5,6,7,8], format_func=lambda x: [
        "Don't Know","STD 1","STD 2", "STD 3","STD 4", "STD 5",
        "STD 6", "STD 7", "STD 8", 
         ][x]
)



#parent education level
educational_level = st.selectbox("Parent's Educational Level", ["Primary", "Secondary", "Tertiary","Don't Know"])

# Title for Learner's Highest Grade Attended
st.write("Parent's highest education grade attended")

if educational_level == "Primary":
    parent_highest_grade = st.selectbox(
        "", [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: [
             "STD 1", "STD 2", "STD 3", "STD 4", "STD 5",
            "STD 6", "STD 7", "STD 8"
        ][x-1]
    )


elif educational_level == "Secondary":
    parent_highest_grade = st.selectbox(
        "", [9, 10, 11, 12], format_func=lambda x: [
            "Form 1", "Form 2", "Form 3", "Form 4"
        ][x-9]
    )
elif educational_level == "Tertiary":
    parent_highest_grade = st.selectbox(
        "", [13, 14, 15,16 ], format_func=lambda x: [
            "First Year", "Second Year", "Third Year", "Fourth Year"
        ][x-13]
    )
elif educational_level=="Don't Know":
    parent_highest_grade=0
#Household wealth index
wealth_index = st.selectbox(" Household wealth index", [0, 1, 2, 3,4], format_func=lambda x: ["Poorest", "Second", "Middle", "Fourth", "Richest"][x])








parent_teacher_progress_discussions = st.selectbox(
    "Parent Discussed with Teacher on Learner's Progress", [0, 1, 2, 3], format_func=lambda x: ["Dont Know", "No", "No Response", "Yes"][x]
)




def mock_model_predict(df):
    prediction = model.predict(df)[0]
    if prediction == 1:
        return "Learner is likely to Drop out "
    else:
        return "Learner is not likely to drop out"
    return 

if st.button("Predict"):
    features = {
        'school_district': school_district,
        'parent_teacher_progress_discussions': parent_teacher_progress_discussions,
        'parent_highest_grade': parent_highest_grade,
        'child_age': child_age,
        'highest_grade': highest_grade,
        'wealth_index': wealth_index,
        'highest_grade_2019_20': highest_grade_2019_20
    }
    
    
    data = np.array((features['school_district'],features['parent_teacher_progress_discussions'],features['parent_highest_grade'],
             features['child_age'],features['highest_grade'],features['wealth_index'],
             features['highest_grade_2019_20'])).reshape(1,-1)
    df = pd.DataFrame(scaler.transform(data), columns=columns)
    # st.write(columns)
    #st.dataframe(df)

    prediction = mock_model_predict(df)
    st.success(f"Learner Drop Out Status: {prediction}")

# To run the app, use the command: streamlit run app.py