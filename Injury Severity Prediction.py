import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ===================== 1. Load Data =====================
df = pd.read_excel("Injury Severity.xlsx")
df = df.rename(columns=lambda x: x.strip()) 

assert "degree_of_inj_x" in df.columns, "目标列 'degree_of_inj_x' 不存在"
df = df.dropna(subset=["degree_of_inj_x"])  # 丢弃无标签行

# ===================== 2. Prepare Features and Target =====================
y = df["degree_of_inj_x"]
X = df.drop(columns=["degree_of_inj_x"])

# 转换所有列名为字符串（保险起见）
X.columns = X.columns.astype(str)

# 文本列和数值列拆分
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype(str)

# ===================== 3. Preprocessing Pipelines =====================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ===================== 4. Train/Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================== 5. Models =====================
models = {
    "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='saga', max_iter=5000),
    "Logistic Regression (L2)": LogisticRegression(penalty='l2', solver='saga', max_iter=5000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "MLP": MLPClassifier(max_iter=500)
}

results = []

# ===================== 6. Training and Evaluation =====================
for name, model in models.items():
    print(f"\nTraining: {name}")
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

# ===================== 7. Results Table =====================
results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
print("\n=== Summary Table ===")
print(results_df)

# ===================== 8. Visualization =====================
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"]),
            x="value", y="Model", hue="variable")
plt.title("Model Comparison on Injury Severity Classification")
plt.xlabel("Score")
plt.ylabel("Model")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()
