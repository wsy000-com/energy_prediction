import pandas as pd

# ── 读取数据 ─────────────────────────────────────────────────
df = pd.read_csv(r'your_path.csv', encoding='utf-8')
df['X_TIME'] = pd.to_datetime(df['X_TIME'])
df = df.sort_values('X_TIME')
df = df[df['X_TIME'].dt.month == 1].reset_index(drop=True)
# 注意：X_TIME 保持为普通列，不设为索引，避免后续 feature_cols 收集时遗漏

# ── 配置 ──────────────────────────────────────────────────────
SITE_COLS  = ['SITE_1', 'SITE_2', 'SITE_3', 'SITE_4', 'SITE_5']
GROUP_KEYS = ['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']
DROP_COLS  = {
    'SAMPLE_COUNT', 'SPC_Time', 'X_TIME_SUMMARY',
    'START_DTTS_SUMMARY', 'END_DTTS_SUMMARY',
    'START_DTTS', 'END_DTTS', 'RAWID', 'PRODUCT_ID',
    'PROC_RECIPE_ID_x', 'EQP_MODULE_ID', 'OPERATION_ID',
    'MAIN_MODULE_ID', 'ROUTE_ID', 'EQUIP_MODEL',
    'MEAS_RECIPE_ID', 'PROC_RECIPE_ID_y', 'RAW_ID',
    'MEAS_EQUIP_ID', 'LASER_TIME_TO_MAINT', 'LOT_TYPE',
    'SPC_Value',  # 直接删除，y 值用 SITE_1~5 代替
}

# ── Step 1：删除指定列 ────────────────────────────────────────
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ── Step 2：确定特征列，删除高缺失列和常量列 ─────────────────
# 特征列 = 除 X_TIME、GROUP_KEYS、SITE_COLS 之外的所有列
# 此处 X_TIME 是普通列，需显式排除，确保它不会被误删
non_feature = set(['X_TIME'] + GROUP_KEYS + SITE_COLS)
feature_cols = [c for c in df.columns if c not in non_feature]

# 删除缺失率 >= 70% 的特征列
missing_rate = df[feature_cols].isna().mean()
high_missing = missing_rate[missing_rate >= 0.7].index.tolist()
df = df.drop(columns=high_missing)
print(f"删除高缺失列 {len(high_missing)} 个: {high_missing}")

# 删除常量列（唯一值 <= 1）
feature_cols = [c for c in df.columns if c not in non_feature]
constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
df = df.drop(columns=constant_cols)
print(f"删除常量列 {len(constant_cols)} 个: {constant_cols}")

# 最终特征列（X_TIME 单独保留，不在 feature_cols 里）
feature_cols = [c for c in df.columns if c not in non_feature]
print(f"保留特征列 {len(feature_cols)} 个")

# ── Step 3：删除 SITE_1~5 全为空的行 ─────────────────────────
df = df.dropna(subset=SITE_COLS, how='all').reset_index(drop=True)

# ── Step 4：去重（X_TIME 排除在外，其余所有列完全相同才算重复）
dedup_cols = GROUP_KEYS + feature_cols + SITE_COLS
df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
print(f"去重后行数: {len(df)}")

# ── Step 5：组内均值填充空值（仅数值型特征列，按 GROUP_KEYS 分组）
# 注意：此步骤必须在 Step 4 去重之后执行
num_features = df[feature_cols].select_dtypes(include='number').columns.tolist()
df[num_features] = df.groupby(GROUP_KEYS)[num_features].transform(
    lambda x: x.fillna(x.mean())
)

# ── Step 6：melt 展开为 site 级 ───────────────────────────────
# id_vars 必须包含 X_TIME，否则时间列会被 melt 丢弃
df_ml = pd.melt(
    df,
    id_vars=['X_TIME'] + GROUP_KEYS + feature_cols,
    value_vars=SITE_COLS,
    var_name='site_col',
    value_name='SPC_Value'
)

# 'SITE_1' → 1, 'SITE_2' → 2, ...（用 str.replace 避免 expand 歧义）
df_ml['site'] = df_ml['site_col'].str.replace('SITE_', '').astype(int)
df_ml = df_ml.drop(columns=['site_col'])

# 删除 SPC_Value 为空的行
df_ml = df_ml.dropna(subset=['SPC_Value']).reset_index(drop=True)

print(f"最终 site 级样本数: {len(df_ml)}")
print(df_ml.head(10))