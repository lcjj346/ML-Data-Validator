# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# df['phone_clean'] = df['phone_number'].apply(clean_phone_number)
# df['phone_len'] = df['phone_clean'].apply(len)

# X = df[['phone_len']]
# y = df['is_valid_phone']  # Make sure this is in the CSV

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# model = RandomForestClassifier().fit(X_train, y_train)

# # Save model
# import joblib
# joblib.dump(model, 'ml/model_phone.pkl')
