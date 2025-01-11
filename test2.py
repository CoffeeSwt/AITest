import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
# 1. 数据加载
data = pd.read_csv('Titanic_data/train.csv')

# 2. 数据预处理
# 填补缺失值
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 将类别变量转换为数字
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# 选择特征变量和目标变量
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# 3. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建模型：决策树与随机森林
# 决策树模型
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 模型预测
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# 6. 评估模型
print("决策树模型准确率:", accuracy_score(y_test, dt_predictions))
print("决策树模型分类报告:\n", classification_report(y_test, dt_predictions))
print("随机森林模型准确率:", accuracy_score(y_test, rf_predictions))
print("随机森林模型分类报告:\n", classification_report(y_test, rf_predictions))

# 7. 数据可视化：展示数据的基本特征
# 1. 生还与否分布
sns.countplot(x='Survived', data=data)
plt.title('Survived vs Not Survived')
plt.show()

# 2. 性别分布：使用 'Sex_male' 列
sns.countplot(x='Sex_male', hue='Survived', data=data)
plt.title('Survival by Gender')
plt.show()

# 3. 年龄分布
sns.histplot(data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# 4. 舱等级与生还的关系
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Pclass')
plt.show()

# 5. 船票价格与生还的关系
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Fare Distribution by Survival')
plt.show()

# 8. 模型性能可视化：混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, dt_predictions)  # 使用决策树预测结果
plot_confusion_matrix(y_test, rf_predictions)  # 使用随机森林预测结果

# 9. 随机森林特征重要性可视化
importances = rf_model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title('Feature Importance from Random Forest')
plt.show()

# 10. ROC 曲线
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 11. Precision-Recall 曲线
precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

plt.plot(recall, precision, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
