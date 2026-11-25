import pandas as pd

REF_YEAR = 2014
REF_DATE = pd.Timestamp('2014-10-04')

FEATURES = ['Age', 'Education_Level', 'Income', 'Recency', 'Total_Spent',
            'Total_Purchases', 'Total_Accepted', 'Children', 'Family_Size',
            'Is_Parent', 'Customer_Days', 'NumWebVisitsMonth']

MNT = ['MntWines', 'MntFruits', 'MntMeatProducts',
       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
NUM = ['NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases']
CMP = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
       'AcceptedCmp4', 'AcceptedCmp5', 'Response']


def clean(raw, training=True):
    """Cleaning. training=False keeps every row (used by the app)."""
    d = raw.copy()
    d['Income'] = d['Income'].fillna(d['Income'].median())
    d['Dt_Customer'] = pd.to_datetime(d['Dt_Customer'])
    d = d.drop(columns=['Z_CostContact', 'Z_Revenue'], errors='ignore')
    d['Marital_Status'] = d['Marital_Status'].replace({
        'Married': 'Partner', 'Together': 'Partner', 'Single': 'Single',
        'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single',
        'Absurd': 'Single', 'YOLO': 'Single'})
    if training:
        for c in ['Year_Birth', 'Income']:
            q1, q3 = d[c].quantile(0.25), d[c].quantile(0.75)
            iqr = q3 - q1
            d = d[(d[c] >= q1 - 1.5 * iqr) & (d[c] <= q3 + 1.5 * iqr)]
        d = d.drop_duplicates(subset=d.columns.drop('ID'))
    return d.reset_index(drop=True)


def build_features(d):
    """Feature engineering. Input: cleaned dataframe."""
    d = d.copy()
    d['Age'] = REF_YEAR - d['Year_Birth']
    d['Children'] = d['Kidhome'] + d['Teenhome']
    d['Family_Size'] = d['Marital_Status'].map({'Single': 1, 'Partner': 2}) + d['Children']
    d['Is_Parent'] = (d['Children'] > 0).astype(int)
    d['Total_Spent'] = d[MNT].sum(axis=1)
    d['Total_Purchases'] = d[NUM].sum(axis=1)
    d['Total_Accepted'] = d[CMP].sum(axis=1)
    d['Customer_Days'] = (REF_DATE - d['Dt_Customer']).dt.days
    d['Education'] = d['Education'].replace({
        'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'})
    d['Education_Level'] = d['Education'].map(
        {'Undergraduate': 0, 'Graduate': 1, 'Postgraduate': 2})
    return d


def prepare_for_model(raw):
    """Raw uploaded data -> model-ready feature table (used by the app)."""
    return build_features(clean(raw, training=False))[FEATURES]
