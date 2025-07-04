import pandas as pd
import joblib

def predict(filepath="data/new_data.xlsx"):
    model = joblib.load("models/age_model.pkl")
    encoder = joblib.load("models/encoder.pkl")

    df = pd.read_excel(filepath)
    df["Email"] = df["Email"].fillna("").apply(lambda x: x.replace("_at_", "@"))
    df["Phone"] = df["Phone"].fillna("").apply(lambda x: ''.join(filter(str.isdigit, str(x)))[:8])

    X = df[["Name", "Email", "Phone"]]
    X_enc = encoder.transform(X)

    predictions = model.predict(X_enc)
    df["Predicted_Age"] = predictions.round(0)

    df.to_excel("data/output_with_predictions.xlsx", index=False)
    print("Predictions saved to Excel.")

if __name__ == "__main__":
    predict()
