import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

def train():
    df = pd.read_csv("data/cleaned_data.csv")

    X = df[["Name", "Email_Corrected", "Phone_Corrected"]]
    y = df["Age"]

    # One-hot encode text columns
    enc = OneHotEncoder(handle_unknown='ignore')
    X_enc = enc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model trained. MSE: {mse:.2f}")
    joblib.dump(model, "models/age_model.pkl")
    joblib.dump(enc, "models/encoder.pkl")

if __name__ == "__main__":
    train()
