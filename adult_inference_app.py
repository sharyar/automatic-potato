import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load
from pathlib import Path
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping


MODEL_PATh = Path("models/adult_pipeline.joblib")
LABEL_ENCORDER_PATH = Path("models/adult_label_encoder.joblib")

st.set_page_config(layout="wide")
st.title("Inference App for Adult Income Dataset!")
st.subheader("This app predicts whether a person's income is above or below $50K based on provided inputs!")
st.subheader("Author: Sharyar Memon")


# Load the model and associated pickle objects: 
with st.spinner("Loading the model..."):
    original_dataset = pd.read_csv("data/x_adult_train.csv")
    with open('models/data_columns.pickle', 'rb') as f:
        column_mappings = pickle.load(f)
    
    pipeline_clf = load(MODEL_PATh)
    label_encoder = load(LABEL_ENCORDER_PATH)
    st.success("Models loaded successfully!")

WORKCLASS_INPUTS = ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay']
EDUCATION_INPUTS = ['Some-college', 'Prof-school', 'Bachelors', 'Assoc-voc',
       'Doctorate', 'HS-grad', 'Masters', '10th', 'Assoc-acdm', '9th',
       '5th-6th', '11th', '7th-8th', '12th', '1st-4th', 'Preschool']
MARITAL_STATUS_INPUTS = ['Never-married',
    'Married-civ-spouse',
    'Divorced',
    'Separated',
    'Widowed',
    'Married-spouse-absent',
    'Married-AF-spouse'
 ]
OCCUPATION = ['Adm-clerical',
 'Prof-specialty',
 'Protective-serv',
 'Exec-managerial',
 'Craft-repair',
 'Sales',
 'Other-service',
 'Handlers-cleaners',
 'Transport-moving',
 'Machine-op-inspct',
 'Tech-support',
 'Farming-fishing',
 'Priv-house-serv',
 'Armed-Forces']

RELATIONSHIP = ['Own-child',
 'Husband',
 'Unmarried',
 'Wife',
 'Not-in-family',
 'Other-relative']

RACE = ['White', 'Black', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo']

COUNTRY = ['Nicaragua',
 'United-States',
 'Jamaica',
 'Italy',
 'Germany',
 'Cuba',
 'Scotland',
 'Canada',
 'Philippines',
 'Mexico',
 'India',
 'Japan',
 'Cambodia',
 'Dominican-Republic',
 'Iran',
 'Ireland',
 'Vietnam',
 'El-Salvador',
 'Puerto-Rico',
 'Hungary',
 'Guatemala',
 'China',
 'Outlying-US(Guam-USVI-etc)',
 'Laos',
 'Peru',
 'France',
 'England',
 'Ecuador',
 'Columbia',
 'Thailand',
 'Poland',
 'Portugal',
 'Taiwan',
 'Greece',
 'South',
 'Haiti',
 'Honduras',
 'Yugoslavia',
 'Hong',
 'Trinadad&Tobago']

#evaluate data drift with Evidently Profile
def detect_dataset_drift(reference, production, column_mapping, get_ratio=False):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns the share of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
    """
    
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference, current_data=production, column_mapping=column_mapping)
    report = data_drift_report.as_dict()
    
    if get_ratio:
        return report["metrics"][0]["result"]["drift_share"]
    else:
        return report["metrics"][0]["result"]["dataset_drift"]

with st.form(key="input-form"):
    age = st.number_input("Age", min_value=0, max_value=100, value=20)
    workclass = st.selectbox("Workclass", WORKCLASS_INPUTS)
    education = st.selectbox("Education", EDUCATION_INPUTS)
    education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
    marital_status = st.selectbox("Marital Status", MARITAL_STATUS_INPUTS)
    occupation = st.selectbox("Occupation", OCCUPATION)
    relationship = st.selectbox("Relationship", RELATIONSHIP)
    race = st.selectbox("Race", RACE)
    sex = st.selectbox("Sex", ["Female", "Male"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=1000)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=85)
    hours = st.number_input("Hours", min_value=0, max_value=100, value=40)
    country = st.selectbox("Native Country", COUNTRY)
    st.form_submit_button(label="Predict")
    
    
    
    
st.header("You can also provide input via a csv file for batch processing!")
input_file = st.file_uploader("Upload CSV file", type=["csv"])

if input_file is not None:
    # do inference and drift checks!
    df = pd.read_csv(input_file)
    predicton = pipeline_clf.predict(df)
    prediction_decoded = label_encoder.inverse_transform(predicton)
    
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=original_dataset,
        current_data=df,
        column_mapping=column_mappings
    )
    drift_df = drift_report.as_pandas()["DataDriftTable"]
    
    st.text(f"Prediction: {prediction_decoded}")
    st.dataframe(drift_df, height=500)
    st.dataframe(drift_report.as_pandas()["DatasetDriftMetric"], height=500)
    
    drift = detect_dataset_drift(original_dataset, df, column_mappings)
    st.text(f"Data Drift: {drift}")
    

