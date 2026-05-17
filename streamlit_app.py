import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/model.pkl"
MODEL2_PATH = "models/model2.pkl"
CAT_COLS = ['payment_channel', 'authentication_type']

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def build_encoder_from_training_data(col):
    train_path = os.path.join('dataset', 'banking_transactions.csv')
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path, usecols=[col])
            if col in train_df.columns:
                le = LabelEncoder()
                le.fit(train_df[col].astype(str))
                os.makedirs('models', exist_ok=True)
                enc_path = os.path.join('models', f'enc_{col}.pkl')
                joblib.dump(le, enc_path)
                return le
        except Exception:
            return None
    return None


def load_encoder(col):
    enc_path = os.path.join('models', f'enc_{col}.pkl')
    if os.path.exists(enc_path):
        try:
            return joblib.load(enc_path)
        except Exception:
            return None
    return build_encoder_from_training_data(col)


def get_expected_features(m):
    if m is None:
        return None
    # sklearn-compatible attribute
    try:
        if hasattr(m, 'feature_names_in_'):
            return list(m.feature_names_in_)
    except Exception:
        pass
    # XGBoost booster feature names
    try:
        booster = m.get_booster()
        names = getattr(booster, 'feature_names', None)
        if names:
            return list(names)
    except Exception:
        pass
    try:
        names = getattr(m, 'feature_names', None)
        if names:
            return list(names)
    except Exception:
        pass
    return None

model = load_model(MODEL_PATH)
model2 = load_model(MODEL2_PATH)

# Preprocessor placeholder (can upload a preprocessing Pipeline or encoder bundle)
preprocessor = None

st.title("Fraud Detection UI")

if model is None and model2 is None:
    st.error("No models found in `models/`. Run the notebook snippet in README to save them.")
    st.stop()

input_method = st.radio("Input method", ["Paste JSON (single transaction)", "Upload CSV (single row)"])

# Allow uploading a preprocessing pipeline (joblib / pickle). If provided, the app will apply it before prediction.
uploaded_prep = st.file_uploader("Upload preprocessing pipeline (.pkl/.joblib)", type=["pkl", "joblib"]) 
if uploaded_prep is not None:
    os.makedirs('models', exist_ok=True)
    prep_path = os.path.join('models', 'preprocessor.pkl')
    with open(prep_path, 'wb') as fh:
        fh.write(uploaded_prep.getbuffer())
    try:
        loaded_obj = joblib.load(prep_path)
        # ensure it's a transformer with transform()
        if hasattr(loaded_obj, 'transform'):
            preprocessor = loaded_obj
            st.success('Loaded preprocessor (transformer)')
        else:
            # save as a model file instead to avoid overwriting existing model names
            alt_path = os.path.join('models', 'uploaded_object.pkl')
            joblib.dump(loaded_obj, alt_path)
            preprocessor = None
            if hasattr(loaded_obj, 'predict'):
                st.warning('Uploaded file looks like a model (has predict). Saved as `models/uploaded_object.pkl`. If you intended a preprocessor, upload a Pipeline object with `transform`.')
            else:
                st.error('Uploaded file is not a transformer. Please upload a preprocessing Pipeline with a `transform` method.')
    except Exception as e:
        st.error(f'Failed to load preprocessor: {e}')

# Allow uploading individual saved encoders for categorical columns
enc1 = st.file_uploader('Upload encoder for `payment_channel` (enc_payment_channel.pkl)', type=['pkl','joblib'])
if enc1 is not None:
    os.makedirs('models', exist_ok=True)
    p = os.path.join('models', 'enc_payment_channel.pkl')
    with open(p, 'wb') as fh:
        fh.write(enc1.getbuffer())
    st.success('Saved encoder to models/enc_payment_channel.pkl')

enc2 = st.file_uploader('Upload encoder for `authentication_type` (enc_authentication_type.pkl)', type=['pkl','joblib'])
if enc2 is not None:
    os.makedirs('models', exist_ok=True)
    p2 = os.path.join('models', 'enc_authentication_type.pkl')
    with open(p2, 'wb') as fh:
        fh.write(enc2.getbuffer())
    st.success('Saved encoder to models/enc_authentication_type.pkl')

def parse_input_json(txt):
    try:
        d = json.loads(txt)
        return pd.DataFrame([d])
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return None

uploaded_df = None
if input_method.startswith("Paste"):
    txt = st.text_area("Transaction JSON", height=200)
    if st.button("Load JSON"):
        df = parse_input_json(txt)
        uploaded_df = df
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            uploaded_df = df
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if uploaded_df is not None:
    st.write("Input preview:")
    st.dataframe(uploaded_df)

    if 'anomaly_score' in uploaded_df.columns:
        chosen = st.radio("Choose model", ("model", "model2", "Auto (use model, fallback to model2)"))
    else:
        chosen = st.radio("Choose model", ("model2",))

    if st.button("Run Prediction"):
        try:
            df = uploaded_df.copy()
            # Drop obvious non-feature columns if present early
            for drop_col in ['transaction_id', 'fraud_flag']:
                if drop_col in df.columns:
                    df = df.drop(columns=[drop_col])

            # If a preprocessor was uploaded, try to apply it and use the transformed output directly
            if preprocessor is not None:
                try:
                    X_trans = preprocessor.transform(df)
                    # Try to get feature names
                    try:
                        feat_names = preprocessor.get_feature_names_out(df.columns)
                    except Exception:
                        feat_names = None

                    # If transformer returned 1d for a single sample, ensure 2D
                    if X_trans.ndim == 1:
                        X_trans = X_trans.reshape(1, -1)

                    if feat_names is not None and len(feat_names) == X_trans.shape[1]:
                        X_df = pd.DataFrame(X_trans, columns=feat_names)
                    else:
                        # Fall back to expected model feature names if available
                        used_model = model if (chosen == 'model') else model2 if (chosen == 'model2') else (model if (model is not None and 'anomaly_score' in df.columns) else model2)
                        expected = get_expected_features(used_model)
                        if expected is not None and len(expected) == X_trans.shape[1]:
                            X_df = pd.DataFrame(X_trans, columns=expected)
                        else:
                            X_df = pd.DataFrame(X_trans)

                    st.write('Transformed features (from uploaded preprocessor):')
                    st.dataframe(X_df)
                    df_for_model = X_df
                except Exception as e:
                    st.error(f'Preprocessor transform failed: {e}. Falling back to in-app conversion.')
                    preprocessor = None

            # Convert any non-numeric columns to numeric where possible; use saved encoders for known categorical columns
            if preprocessor is None:
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                        continue

                    # Handle known categorical columns with saved encoders first
                    if col in CAT_COLS:
                        enc = load_encoder(col)
                        if enc is not None:
                            try:
                                df[col] = enc.transform(df[col].astype(str))
                                df[col] = pd.to_numeric(df[col])
                                continue
                            except Exception as e:
                                st.warning(f"Saved encoder for {col} failed to transform values: {e}; will try fallback encoding.")

                    # Try to coerce to numeric (for numeric strings)
                    coerced = pd.to_numeric(df[col], errors='coerce')
                    if coerced.notna().all():
                        df[col] = coerced
                        continue

                    # Fallback: use LabelEncoder (on-the-fly). Warn user this may not match training encoding.
                    try:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        st.warning(f"Column '{col}' converted using on-the-fly LabelEncoder. This may not match training encoding.")
                    except Exception as e:
                        st.error(f"Failed to convert column '{col}' to numeric: {e}")
                        raise
            # Prepare processed DataFrame for model
            if 'df_for_model' in locals():
                processed = df_for_model.reset_index(drop=True)
            else:
                processed = df.reset_index(drop=True)

            # Keep original inputs for output
            orig_inputs = uploaded_df.reset_index(drop=True)

            # Drop obvious non-feature columns if present
            for drop_col in ['transaction_id', 'fraud_flag']:
                if drop_col in processed.columns:
                    processed = processed.drop(columns=[drop_col])

            # Choose model object for feature introspection
            if chosen == "model":
                used_model = model
            elif chosen == "model2":
                used_model = model2
            else:
                used_model = model if (model is not None) else model2

            expected = get_expected_features(used_model)
            if expected is not None:
                missing = [f for f in expected if f not in processed.columns]
                extra = [c for c in processed.columns if c not in expected]
                if extra:
                    st.warning(f"Dropping unexpected columns: {extra}")
                    processed = processed.drop(columns=extra)
                if missing:
                    st.warning(f"Adding missing features with default 0: {missing}")
                    for mcol in missing:
                        processed[mcol] = 0
                # Reorder to match training
                processed = processed.reindex(columns=expected)

            # Perform predictions for all rows
            results = []
            if chosen in ("model", "model2"):
                m = model if chosen == "model" else model2
                if m is None:
                    st.error(f"`{chosen}` not found in models/")
                    raise Exception("Model missing")
                preds = m.predict(processed)
                probs = m.predict_proba(processed)[:, 1] if hasattr(m, 'predict_proba') else [None] * len(preds)
                for p, prob in zip(preds, probs):
                    risk = "🔴 HIGH" if (prob is not None and prob > 0.7) else "🟡 MEDIUM" if (prob is not None and prob > 0.3) else "🟢 LOW"
                    results.append({"fraud": bool(p), "confidence": f"{prob*100:.1f}%" if prob is not None else None, "risk_level": risk})
            else:
                # Auto: choose model per-row based on presence of anomaly_score and availability
                for i in range(len(processed)):
                    row = processed.iloc[[i]]
                    use_model = None
                    if model is not None and 'anomaly_score' in row.columns and not pd.isna(row['anomaly_score'].iloc[0]):
                        use_model = model
                    elif model2 is not None:
                        use_model = model2
                    elif model is not None:
                        use_model = model
                    else:
                        raise Exception('No available model')
                    p = use_model.predict(row)[0]
                    prob = use_model.predict_proba(row)[0][1] if hasattr(use_model, 'predict_proba') else None
                    risk = "🔴 HIGH" if (prob is not None and prob > 0.7) else "🟡 MEDIUM" if (prob is not None and prob > 0.3) else "🟢 LOW"
                    results.append({"fraud": bool(p), "confidence": f"{prob*100:.1f}%" if prob is not None else None, "risk_level": risk})

            res_df = pd.DataFrame(results)
            out_df = pd.concat([orig_inputs.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
            st.write("Predictions:")
            st.dataframe(out_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
