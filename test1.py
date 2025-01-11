import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 切换到 TkAgg 后端

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'

# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 检查数据结构
print(df.head())

# 构建多元线性回归模型
# 定义自变量 X 和因变量 y
X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
y = df['y']

# 添加常数项 (截距)
X = sm.add_constant(X)

# 训练模型
model = sm.OLS(y, X).fit()

# 打印回归方程
coefficients = model.params
equation = "y = {:.4f}".format(coefficients[0])  # 截距项
for i in range(1, len(coefficients)):
    equation += " + {:.4f} * {}".format(coefficients[i], X.columns[i])
print("回归方程: " + equation)

# 预测结果
predictions = model.predict(X)

# 可视化：真实值与预测值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 完美预测线
plt.xlabel("真实值 y")
plt.ylabel("预测值 y")
plt.title("真实值 vs 预测值")
plt.show()

# 残差图：真实值与残差的对比
residuals = y - predictions
plt.figure(figsize=(10, 6))
plt.scatter(y, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # 零残差线
plt.xlabel("真实值 y")
plt.ylabel("残差")
plt.title("真实值 vs 残差")
plt.show()
