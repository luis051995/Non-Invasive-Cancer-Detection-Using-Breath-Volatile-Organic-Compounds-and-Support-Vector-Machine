import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.decomposition import PCA

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


df = pd.read_csv("cancer_vocs_dataset.csv")


X = df.drop("Cancer", axis=1)
y = df["Cancer"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Entrenamiento del SVM
# -----------------------------
svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)

# -----------------------------
# 6. Predicciones
# -----------------------------
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 7. Métricas
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - SVM Cancer Detection")
plt.colorbar()
plt.xticks([0, 1], ["Healthy", "Cancer"])
plt.yticks([0, 1], ["Healthy", "Cancer"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=300)
plt.close()

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix)
plt.colorbar()
plt.title("Correlation Matrix of VOC Features")
plt.xticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)),
           correlation_matrix.columns)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_matrix.png", dpi=300)
plt.close()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.transform(X))

svm_pca = SVC(kernel="rbf", C=10, gamma="scale")
svm_pca.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Healthy", alpha=0.7)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Cancer", alpha=0.7)
plt.title("SVM Class Separation using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/svm_class_separation_pca.png", dpi=300)
plt.close()


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Cancer Detection using VOCs")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=300)
plt.close()

print("✅ Todas las gráficas fueron guardadas en la carpeta 'figures/'")
