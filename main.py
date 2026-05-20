from sklearn.preprocessing import StandardScaler

# 准备特征和目标变量
X_train = train_data[feature_columns]
y_train = train_data['SPC_Value']
X_test = test_data[feature_columns]
y_test = test_data['SPC_Value']

# ============ 【新增】特征标准化 ============
scaler = StandardScaler()

# 核心：只能用训练集来 fit（计算均值和方差），然后 transform 训练集和测试集
# 注意：transform 会返回 numpy 矩阵，为了保留列名（方便后续画特征重要性图），需重新包回 DataFrame
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns, 
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=X_test.columns, 
    index=X_test.index
)

# 覆盖原始变量
X_train = X_train_scaled
X_test = X_test_scaled
# ==========================================

# ============ 第七/八步：后续 XGBoost 建模代码完全不用动...