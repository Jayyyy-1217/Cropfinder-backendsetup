import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv('Crop_data.csv')
label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])

X = data.drop('Crop', axis=1)
y = data['Crop']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(rf_model, 'mlmodel/crop_rf_model.joblib')
joblib.dump(label_encoder, 'mlmodel/crop_label_encoder.joblib')
joblib.dump(scaler, 'mlmodel/feature_scaler.joblib')
