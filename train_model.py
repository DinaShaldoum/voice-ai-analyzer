import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# بيانات تجريبية (تقدر تطورها بعدين)
X = np.array([
    [20, 0.02, 0.1, 0.01, 3],   # طبيعي
    [80, 0.05, 0.03, 0.07, 1],  # اضطراب
    [60, 0.08, 0.2, 0.06, 2],   # تعاطي
    [25, 0.03, 0.12, 0.02, 3],
    [70, 0.06, 0.04, 0.08, 1],
    [55, 0.07, 0.18, 0.05, 2],
])

y = np.array([0, 1, 2, 0, 1, 2])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "voice_model.pkl")

print("✅ تم إنشاء الموديل بنجاح")