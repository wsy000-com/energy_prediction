# ============ 准备工作：定义分组键和检查范围 ============
group_keys = ['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']
site_cols = ['SITE_1', 'SITE_2', 'SITE_3', 'SITE_4', 'SITE_5']

print("\n" + "=" * 60)
print("开始排查 Wafer 内部存在波动的列 (非唯一列)...")
print("=" * 60)

# ============ 第一步：诊断非唯一列 ============
varying_cols_summary = {}
# 为了排查时间，我们把 index (X_TIME) 也临时重置为列一起检查
df_check = df.reset_index()
check_cols = columns_to_plot + ['X_TIME'] if 'X_TIME' in df_check.columns else columns_to_plot

# 按 Wafer 分组，找出组内唯一值数量 > 1 的列
nunique_per_group = df_check.groupby(group_keys)[check_cols].nunique(dropna=True)
varying_cols = nunique_per_group.columns[nunique_per_group.max() > 1].tolist()

if varying_cols:
    print(f"⚠️ 发现 {len(varying_cols)} 个列在同一个 Wafer 内部的值不唯一：")

    # 提取几个具体波动的例子展示给用户
    for col in varying_cols:
        print(f"\n  => 波动列名: {col}")
        # 找到哪些 Wafer 在这个列上发生了波动
        fluctuating_groups = nunique_per_group[nunique_per_group[col] > 1].index

        # 打印前 3 个示例
        for i, group_idx in enumerate(fluctuating_groups[:3], 1):
            # 获取这个特定 Wafer 的数据并查看不同值
            if isinstance(group_idx, tuple):
                # 构建查询条件
                mask = (df_check[group_keys[0]] == group_idx[0]) & \
                       (df_check[group_keys[1]] == group_idx[1]) & \
                       (df_check[group_keys[2]] == group_idx[2])
                vals = df_check.loc[mask, col].dropna().unique()
            else:
                vals = df_check.loc[df_check[group_keys[0]] == group_idx, col].dropna().unique()

            # 如果是时间列，稍微格式化一下方便阅读
            if col == 'X_TIME':
                vals_str = [pd.to_datetime(v).strftime('%H:%M:%S') for v in vals]
                print(f"       示例 Wafer {i} 的时间跨度: {vals_str}")
            else:
                print(f"       示例 Wafer {i} 的变化值: {tuple(vals)}")

        if len(fluctuating_groups) > 3:
            print(f"       ... (该列在总计 {len(fluctuating_groups)} 个 Wafer 中存在波动)")
else:
    print("✅ 未发现波动列，所有特征在一个 Wafer 内部都是绝对一致的。")

print(
    "\n(请根据上述打印结果评估：如果是时间波动或轻微的数值波动，采用均值聚合是合理的；如果分类列出现波动，建议排查原始数据。)")

# ============ 第二步：按 Wafer 去重并转化为长表 ============
print("\n" + "=" * 60)
print("开始进行 Wafer 级去重与 Site 级长表展开...")
print("=" * 60)

# 构建聚合字典
agg_funcs = {}
for col in columns_to_plot:
    if col in varying_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_funcs[col] = 'mean'  # 如果是数值型波动，取均值平滑
        else:
            agg_funcs[col] = 'first'  # 如果是分类型波动，强制取第一条
    else:
        agg_funcs[col] = 'first'  # 完全不波动的列，取 first 性能最高

# 目标变量列（Site测量值通常在 Wafer 级别是不变的，如果是通过 join 带上来的）
for col in site_cols:
    if col in df.columns:
        agg_funcs[col] = 'first'

# 执行去重：每个 Wafer 只保留 1 行物理意义上的特征
df_wafer = df.groupby(group_keys, dropna=False).agg(agg_funcs).reset_index()

# 转换为长表 (Melt)
df_ml = pd.melt(
    df_wafer,
    id_vars=group_keys + columns_to_plot,
    value_vars=site_cols,
    var_name='Site_Position',
    value_name='y'
)

# 清洗空值
df_ml = df_ml.dropna(subset=['y']).copy()

# 将 Site_Position 转为 One-Hot 编码 (转为 0 和 1)
df_ml = pd.get_dummies(df_ml, columns=['Site_Position'], prefix='pos')
pos_cols = [col for col in df_ml.columns if col.startswith('pos_SITE_')]
df_ml[pos_cols] = df_ml[pos_cols].astype(int)

print(f"✅ 转换完成！")
print(f"  - 去重后的独立 Wafer 总数: {len(df_wafer)}")
print(f"  - 展开后的 Site 级样本总数 (供 XGBoost 训练): {len(df_ml)}")