import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.DataFrame({
    'size': [30000, 28000, 26000, 20000],
    'weight': [400, 380, 360, 300],
    'quality': ['High', 'High', 'Medium', 'Low'],
    'price': [120, 110, 90, 70]
})

le = LabelEncoder()
data['quality_encoded'] = le.fit_transform(data['quality'])

X = data[['size', 'weight', 'quality_encoded']]
y = data['price']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'pomegranate_price_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model and encoder saved!")
