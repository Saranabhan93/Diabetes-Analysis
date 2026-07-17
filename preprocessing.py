import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("✅ تم استيراد المكتبات بنجاح!")

#%% [الجزء 1: تحميل البيانات]
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
df = pd.read_csv(url, names=names)
print("✅ تم تحميل البيانات بنجاح!")

#%% [الجزء 2: معالجة القيم المفقودة المخفية]
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)
print("✅ تم استبدال الأصفار بالـ Median بنجاح!")

#%% [الجزء 3: تقسيم البيانات]
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("✅ تم تقسيم البيانات إلى Train و Test!")

#%% [الجزء 4: توحيد المقاييس]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ تم توحيد مقاييس البيانات (Scaling) بنجاح!")
