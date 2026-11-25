import joblib
import pandas as pd
import streamlit as st
from preprocess import FEATURES, prepare_for_model

st.set_page_config(page_title='Customer Segmentation', layout='centered')
st.title('Customer Segmentation')

bundle = joblib.load('model.pkl')
scaler, pca, model = bundle['scaler'], bundle['pca'], bundle['model']

PERSONAS = {
    0: ('Established Family Spenders',
        'Mid-high income parents, steady buyers. Offer family bundles and loyalty rewards.'),
    1: ('Budget Browsers',
        'Low income, low spend, many web visits. Offer discounts and entry-level deals.'),
    2: ('Premium Spenders',
        'High income, no kids, highest spend. Offer premium products and exclusive campaigns.'),
}


def predict(features_df):
    x = scaler.transform(features_df[FEATURES])
    x = pca.transform(x)
    return model.predict(x)


tab1, tab2 = st.tabs(['Single customer', 'File upload'])

with tab1:
    age = st.number_input('Age', 18, 100, 45)
    edu = st.selectbox('Education', ['Undergraduate', 'Graduate', 'Postgraduate'])
    income = st.number_input('Yearly income', 0, 200000, 50000, step=1000)
    recency = st.slider('Days since last purchase', 0, 100, 50)
    spent = st.number_input('Total spent (last 2 years)', 0, 3000, 500)
    purchases = st.number_input('Total purchases', 0, 50, 15)
    accepted = st.slider('Campaigns accepted', 0, 6, 0)
    children = st.slider('Children at home', 0, 4, 0)
    partner = st.radio('Living with a partner?', ['Yes', 'No'], horizontal=True)
    days = st.number_input('Days as a customer', 0, 1000, 450)
    visits = st.slider('Website visits last month', 0, 20, 5)

    if st.button('Predict segment'):
        row = pd.DataFrame([{
            'Age': age,
            'Education_Level': ['Undergraduate', 'Graduate', 'Postgraduate'].index(edu),
            'Income': income, 'Recency': recency, 'Total_Spent': spent,
            'Total_Purchases': purchases, 'Total_Accepted': accepted,
            'Children': children,
            'Family_Size': children + (2 if partner == 'Yes' else 1),
            'Is_Parent': int(children > 0), 'Customer_Days': days,
            'NumWebVisitsMonth': visits}])
        c = int(predict(row)[0])
        name, advice = PERSONAS[c]
        st.success(f'Segment {c}: {name}')
        st.write(advice)

with tab2:
    st.write('Upload a file in the same format as the original dataset (.csv or .xlsx).')
    f = st.file_uploader('Upload file', type=['csv', 'xlsx'])
    if f is not None:
        if f.name.endswith('xlsx'):
            raw = pd.read_excel(f)
        else:
            raw = pd.read_csv(f, sep=None, engine='python')
        feats = prepare_for_model(raw)
        result = feats.copy()
        result['Segment'] = predict(feats)
        result['Persona'] = result['Segment'].map(lambda c: PERSONAS[c][0])
        st.dataframe(result.head(20))
        st.bar_chart(result['Persona'].value_counts())
        st.download_button('Download results', result.to_csv(index=False), 'segments.csv')
