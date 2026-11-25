# Customer Segmentation

Clusters 2,240 customers of a marketing dataset into 3 segments, using
scaling, PCA and K-Means (k = 3). Built with Python, scikit-learn and Streamlit.

## Segments
- Established Family Spenders: mid-high income parents, steady buyers
- Budget Browsers: low income, low spend, many web visits
- Premium Spenders: high income, no children, highest spend

## Live app
(https://customer-segmentation-r5uecv6ypj45tajmx3bs7x.streamlit.app/)

## Files
- app.py: the Streamlit app (single customer form and file upload)
- preprocess.py: cleaning and feature code shared with the notebook
- model.pkl: saved scaler, PCA and K-Means model
- requirements.txt: libraries needed to run the app
