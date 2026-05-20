# ============ 把 wafer 级 SPC_Value 拆成 site 级 ============
# 只保留有 SPC 测量的行
df = df.dropna(subset=['SPC_Value']).copy()

# 同一个 wafer 内按 X_TIME 排序,编号 1-5
df = df.sort_index()
df['site_order'] = (
    df.groupby(['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']).cumcount() + 1
)
df = df[df['site_order'].between(1, 5)].copy()

# 用 site_order 选对应的 SITE_x 值,覆盖原来的 SPC_Value(均值)
site_cols = ['SITE_1', 'SITE_2', 'SITE_3', 'SITE_4', 'SITE_5']
site_arr = df[site_cols].values
df['SPC_Value'] = site_arr[
    np.arange(len(df)), df['site_order'].values - 1
]

# site_order one-hot 编码,用 pos_ 前缀避免和 SITE_x 混淆
df = pd.get_dummies(df, columns=['site_order'], prefix='pos')
# 确保 pos_1 ~ pos_5 五列都存在(防止某个位置在数据里没出现)
for i in range(1, 6):
    col = f'pos_{i}'
    if col not in df.columns:
        df[col] = 0
pos_cols = [f'pos_{i}' for i in range(1, 6)]
df[pos_cols] = df[pos_cols].astype(int)

# 检查分布
print("各位置行数分布:")
print(df[pos_cols].sum())
print(f"总行数: {len(df)}")


columns_to_plot = ['SPC_Value']

fig, axes = plt.subplots(1, 1, figsize=(20, 5), sharex=True)
axes = [axes]   # 统一成 list,后面循环不用改