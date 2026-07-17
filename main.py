import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("✅ تم استيراد المكتبات بنجاح!")

#%% [الجزء 1: تحميل البيانات]
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
]
df = pd.read_csv(url, names=column_names)
print("✅ تم تحميل البيانات بنجاح!")

#%% [الجزء 2: استكشاف حجم البيانات ونوعها]
print("\n--- 📊 نظرة عامة على أول 5 أسطر ---")
print(df.head())

print("\n--- 📏 حجم البيانات (الأسطر والأعمدة) ---")
print(f"عدد الأسطر (Rows): {df.shape[0]}")
print(f"عدد الأعمدة (Columns): {df.shape[1]}")

print("\n--- 💾 أنواع البيانات (Data Types) ---")
print(df.dtypes)

#%% [الجزء 3: تحليل المتغير المستهدف وتوزيع الفئات]
target = "Outcome"
class_counts = df[target].value_counts()
class_percentages = df[target].value_counts(normalize=True) * 100

print("\n--- 🎯 توزيع الفئات المباشر ---")
for cls, count in class_counts.items():
    print(f"الفئة {cls}: {count} سجل، بنسبة {class_percentages[cls]:.2f}%")

#%% [الجزء 4: فحص القيم المفقودة]
print("\n--- 🔍 فحص القيم المفقودة التقليدية ---")
print(df.isnull().sum())

#%% [الجزء 5: توليد الرسومات البيانية]
print("\n--- 📉 جاري توليد الرسومات البيانية الآن... ---")
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x="Outcome", hue="Outcome", data=df, palette="Set2", ax=axes[0], legend=False)
axes[0].set_title("Class Distribution (Outcome)")
axes[0].set_xlabel("Diabetes Outcome (0 = No, 1 = Yes)")
axes[0].set_ylabel("Count")

sns.histplot(data=df, x="Age", hue="Outcome", kde=True, multiple="stack", palette="Set1", ax=axes[1])
axes[1].set_title("Age Distribution by Diabetes Outcome")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show(block=True)
