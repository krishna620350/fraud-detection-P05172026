# Fraud Detection UI

This project includes a small Streamlit UI to run predictions with the `model` and `model2` trained in your notebook.

Steps

1. From your notebook (after training `model` and `model2`), save the models into the `models/` folder:

```python
import joblib
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(model2, 'models/model2.pkl')
print('Saved models to models/')
```

Optional: If you used `LabelEncoder` for categorical columns (recommended), also save the encoders so the UI can apply the same encoding:

```python
from sklearn.preprocessing import LabelEncoder
enc_payment = LabelEncoder()
enc_auth = LabelEncoder()
df['payment_channel'] = enc_payment.fit_transform(df['payment_channel'].astype(str))
df['authentication_type'] = enc_auth.fit_transform(df['authentication_type'].astype(str))
joblib.dump(enc_payment, 'models/enc_payment_channel.pkl')
joblib.dump(enc_auth, 'models/enc_authentication_type.pkl')
```

2. Install dependencies (in your `myenv` or current environment):

```bash
pip install streamlit pandas scikit-learn xgboost joblib
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Notes

- The Streamlit app expects input features to be preprocessed the same way the models were trained. Use the same feature ordering and encodings.
- If a transaction row is missing `anomaly_score`, choose `model2` or the Auto fallback.
- The app accepts a JSON payload (single transaction) or a CSV file with a single row.
