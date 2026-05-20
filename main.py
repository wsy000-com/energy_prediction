import pandas as pd

# ── 读取数据 ─────────────────────────────────────────────────
df = pd.read_csv(r'your_path.csv', encoding='utf-8')
df['X_TIME'] = pd.to_datetime(df['X_TIME'])
df = df.set_index('X_TIME').sort_index()
df = df[df.index.month == 1]

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

# ── Step 2：确定特征列（排除 GROUP_KEYS 和 SITE_COLS）────────
feature_cols = [c for c in df.columns if c not in GROUP_KEYS + SITE_COLS]

# 删除缺失率 > 70% 的特征列
missing_rate = df[feature_cols].isna().mean()
high_missing = missing_rate[missing_rate >= 0.7].index.tolist()
df = df.drop(columns=high_missing)
print(f"删除高缺失列 {len(high_missing)} 个: {high_missing}")

# 删除常量列（唯一值 <= 1）
feature_cols = [c for c in df.columns if c not in GROUP_KEYS + SITE_COLS]
constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
df = df.drop(columns=constant_cols)
print(f"删除常量列 {len(constant_cols)} 个: {constant_cols}")

# 更新特征列
feature_cols = [c for c in df.columns if c not in GROUP_KEYS + SITE_COLS]

# ── Step 3：删除 SITE_1~5 全为空的行 ─────────────────────────
df = df.dropna(subset=SITE_COLS, how='all')

# ── Step 4：去重（忽略日期索引，完全相同的行只保留一行）──────
df = df.reset_index()  # 把 X_TIME 变回列，参与去重判断
df = df.drop_duplicates(subset=GROUP_KEYS + feature_cols + SITE_COLS)
print(f"去重后行数: {len(df)}")

# ── Step 5：组内均值填充空值 ──────────────────────────────────
num_features = df[feature_cols].select_dtypes(include='number').columns.tolist()
df[num_features] = df.groupby(GROUP_KEYS)[num_features].transform(
    lambda x: x.fillna(x.mean())
)

# ── Step 6：melt 展开为 site 级 ───────────────────────────────
df_ml = pd.melt(
    df,
    id_vars=GROUP_KEYS + feature_cols,
    value_vars=SITE_COLS,
    var_name='site_col',
    value_name='y'
)

# 'SITE_1' → 1, 'SITE_2' → 2, ...
df_ml['site'] = df_ml['site_col'].str.extract(r'(\d+)').astype(int)
df_ml = df_ml.drop(columns=['site_col'])

# 删除 y 为空的行
df_ml = df_ml.dropna(subset=['y']).reset_index(drop=True)

print(f"最终 site 级样本数: {len(df_ml)}")
print(df_ml.head(10))