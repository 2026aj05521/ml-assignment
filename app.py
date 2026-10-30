import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report)

st.set_page_config(page_title='Breast Cancer Classifier', page_icon='🔬', layout='wide')

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_FILES = {
    'Logistic Regression': 'Logistic_Regression.joblib',
    'Decision Tree':       'Decision_Tree.joblib',
    'KNN':                 'KNN.joblib',
    'Naive Bayes':         'Naive_Bayes.joblib',
    'SVM':                 'SVM.joblib',
}

@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))

def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'AUC':       round(roc_auc_score(y_true, y_prob), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1 Score':  round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y_true, y_pred), 4),
    }

st.title('Breast Cancer Classification - ML Assignment 2')
st.markdown('Upload **test_data.csv**, select a model, and view evaluation metrics.')
st.divider()

with st.sidebar:
    st.header('Controls')
    uploaded = st.file_uploader('Upload test_data.csv', type=['csv'])
    model_choice = st.selectbox('Select Model', list(MODEL_FILES.keys()))
    run_btn = st.button('Run Evaluation', type='primary', use_container_width=True)
    st.divider()
    st.markdown('**Dataset:** Wisconsin Breast Cancer | **Features:** 30 | **Instances:** 569')

if uploaded is None:
    st.info('Upload test_data.csv from the sidebar to get started.')
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f'Could not read file: {e}')
    st.stop()

if 'target' not in df.columns:
    st.error('CSV must contain a target column.')
    st.stop()

X = df.drop('target', axis=1).values
y = df['target']
st.success(f'Loaded {len(df)} samples with {X.shape[1]} features.')

if not run_btn:
    st.info('Click Run Evaluation in the sidebar.')
    st.stop()

with st.spinner(f'Evaluating {model_choice}...'):
    model  = load_model(model_choice)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = get_metrics(y, y_pred, y_prob)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, target_names=['Malignant (0)', 'Benign (1)'])

st.subheader(f'Evaluation Metrics - {model_choice}')
cols = st.columns(6)
for col, (name, val) in zip(cols, metrics.items()):
    col.metric(name, val)

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_choice}')
    plt.tight_layout()
    st.pyplot(fig)

with c2:
    st.subheader('Classification Report')
    st.code(cr)

st.divider()
st.subheader('All Models Comparison')
all_res = []
for mname in MODEL_FILES:
    m  = load_model(mname)
    yp = m.predict(X)
    yb = m.predict_proba(X)[:, 1]
    mt = get_metrics(y, yp, yb)
    all_res.append({'Model': mname, **mt})
res_df = pd.DataFrame(all_res).set_index('Model')
st.dataframe(res_df, use_container_width=True)
st.caption('Green = best value per metric. Evaluated on uploaded test data without retraining.')
