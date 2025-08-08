import pandas as pd
import re

def clean_email(email):
    if "_at_" in email:
        return email.replace("_at_", "@")
    if not "@" in email:
        return email + "@gmail.com"
    return email

def clean_phone(phone):
    return re.sub(r'\D', '', str(phone))[:8]  # remove non-digit

def preprocess(path="data/generated_data.xlsx"):
    df = pd.read_excel(path)

    # Apply corrections
    df["Email_Corrected"] = df["Email"].apply(clean_email)
    df["Phone_Corrected"] = df["Phone"].apply(clean_phone)

    # Drop rows where Age is missing
    df = df[df["Age"].notnull()]

    df.to_csv("data/cleaned_data.csv", index=False)
    print("Preprocessed and saved cleaned data.")

if __name__ == "__main__":
    preprocess()
