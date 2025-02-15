import streamlit as st
import pandas as pd
import pickle
import datetime
import joblib
from sklearn.preprocessing import StandardScaler
import io

buffer = io.BytesIO()

# Load model
def load_model(model_name):
    if model_name == 'Logistic Regression':
        model = pickle.load(open('models/no_resampling_logistic_regression_model.pkl', 'rb'))
    elif model_name == 'SVM':
        model = pickle.load(open('models/no_resampling_svm_model.pkl', 'rb'))
    elif model_name == 'Decission Tree':
         model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
    elif model_name == 'Random Forest':
         model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
    elif model_name == 'XGB':
         model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
    elif model_name == 'GBM':
         model = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
    return model

# Fungsi untuk melakukan prediksi
def predict_attrition(model, data):
    predictions = model.predict(data)
    return predictions

# Fungsi untuk mewarnai prediksi
def warna(wrn):
    color = 'pink' if wrn == 'Dropout' else 'blue'
    return f'color: {color}'

def main():
    st.title('Jaya Jaya Institute Student Prediction using Machine Learning')
    st.write("B244044F")
        

    with st.sidebar.expander("How to "):
        st.sidebar.write(
            """
            Run the prediction:

                1. Upload csv template file.
                2. Choose machine learning model.
                3. Click predict button Pilih model machine learning. 
                4. Result will appear and can be 'Download (.csv)'. 

            """
        )

    # Pemilihan model ML
    model_name = st.selectbox("Pilih Model Machine Learning", ("Logistic Regression", "SVM"
                                                                        ,"Decission Tree", "Random Forest", "XGB", "GBM"
                                                                       ))

    # Upload File
    uploaded_file = st.file_uploader("Upload csv template file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        student_id = data['StudentId']        
        data = data.drop(columns=['StudentId'])

        # Load model
        model = load_model(model_name)

        # click button
        if st.button('✨ Predict'):
            
            predictions = predict_attrition(model, data)

           
            prediction_labels = ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]

            # Result
            hasil = pd.DataFrame({
                'StudentId': student_id,
                'Status Prediction': prediction_labels
            })

            
            st.write("Prediction result:")
            st.dataframe(hasil.style.applymap(warna, subset=['Status Prediction']))

            # Download result
            csv = hasil.to_csv(index=False)
            st.download_button(
                label="Download (.csv)",
                data=csv,
                file_name='hasil-prediksi-student.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()



year_now = datetime.date.today().year
year = year_now if year_now == 2024 else f'2024 - {year_now}'
name = "[B244044F]"
copyright = 'Copyright © ' + str(year) + ' ' + name
st.caption(copyright)