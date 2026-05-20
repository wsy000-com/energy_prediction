# ====== 在遍历画散点图的循环里，加一个 site == 1 的过滤 ======
for i, col in enumerate(feature_columns):
    # 【修改】：只取 site == 1 的数据看特征和Y的关系
    df_site1 = df_model[df_model['site'] == 1]
    temp = df_site1[[col, 'SPC_Value']].dropna(subset=['SPC_Value'])

    x = temp[col]
    y = temp['SPC_Value']
    # ... 后面的填充和画图代码完全保持不变 ...

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============ 第七/八步合并：分 Site 独立训练与评估 ============
print("\n" + "=" * 60)
print("开始按 Site 独立训练与评估模型...")
print("=" * 60)

# 定义你之前找到的最佳参数（如果你想重新网格搜索，把搜索代码放进循环里即可）
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'n_estimators': 100,
    'subsample': 0.8,
    'random_state': 42
}

sites = sorted(df_model['site'].unique())
site_models = {}  # 字典，用来保存 5 个训练好的模型

for s in sites:
    print(f"\n\n{'#' * 20} 正在处理 Site {s} {'#' * 20}")

    # 1. 提取当前 site 的数据
    train_s = train_data[train_data['site'] == s]
    test_s = test_data[test_data['site'] == s]

    if len(train_s) == 0 or len(test_s) == 0:
        print(f"Site {s} 数据不足，跳过！")
        continue

    X_train_s = train_s[feature_columns]
    y_train_s = train_s['SPC_Value']
    X_test_s = test_s[feature_columns]
    y_test_s = test_s['SPC_Value']

    # 2. 【新增】特征标准化 (每个 site 独立做标准化，最安全)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_s), columns=X_train_s.columns, index=X_train_s.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_s), columns=X_test_s.columns, index=X_test_s.index)

    # 3. 训练模型
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train_s)
    site_models[s] = model  # 存入字典

    # 4. 预测与指标计算
    y_pred_s = model.predict(X_test_scaled)
    r2 = r2_score(y_test_s, y_pred_s)
    mae = mean_absolute_error(y_test_s, y_pred_s)
    rmse = np.sqrt(mean_squared_error(y_test_s, y_pred_s))

    print(f"[{'Site ' + str(s)}] 测试集 R²：{r2:.4f} | MAE：{mae:.4f} | RMSE：{rmse:.4f}")

    # 5. 打印 Top 5 特征重要性（不同 site 可能会有差异哦！）
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"Top 5 特征：{feature_importance.head(5)['feature'].tolist()}")

    # 6. 画图 (将你原来的画图代码搬进循环，标题加上 Site 标识)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # Plot 1: 时序对比图
    axes[0].plot(y_test_s.index, y_test_s.values, label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(y_test_s.index, y_pred_s, label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('X_TIME')
    axes[0].set_ylabel('SPC_Value')
    axes[0].set_title(f'Site {s}: Actual vs Predicted Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: 预测 vs 实际散点图
    axes[1].scatter(y_test_s, y_pred_s, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    min_val = min(y_test_s.min(), y_pred_s.min())
    max_val = max(y_test_s.max(), y_pred_s.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title(f'Site {s}: Predicted vs Actual (R2 = {r2:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\n所有 Site 训练完毕！")