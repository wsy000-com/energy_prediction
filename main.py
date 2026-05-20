# ============ 把 wafer 级 SPC_Value 拆成 site 级 ============
EXPECTED_SITES = 5   # 一个 wafer 期望测多少 site,以后换数据集改这个就行

# 1. 只保留有 SPC 测量的行
df = df.dropna(subset=['SPC_Value']).copy()
df = df.sort_index()

# 2. 诊断:每个 wafer 几行 + 相邻时间间隔(用来确认场景,不写死秒数)
group_keys = ['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']
wafer_size = df.groupby(group_keys).size()
print("每个 wafer 的 SPC 行数分布(SPC_Value 非空):")
print(wafer_size.value_counts().sort_index())
print(f"总 wafer 数: {len(wafer_size)}\n")

gaps_by_n = {}
for _, g in df.groupby(group_keys):
    if len(g) < 2:
        continue
    gaps = g.index.to_series().diff().dt.total_seconds().dropna().tolist()
    gaps_by_n.setdefault(len(g), []).extend(gaps)

print("各行数 wafer 的相邻时间间隔统计:")
for n in sorted(gaps_by_n):
    s = pd.Series(gaps_by_n[n])
    print(f"  {n} 行 wafer:gap 中位数 {s.median():.1f}s,均值 {s.mean():.1f}s,std {s.std():.2f}s,max {s.max():.1f}s")

# 自动估算基准间隔 T:优先用满测 wafer 的 gap 中位数,不写死
if EXPECTED_SITES in gaps_by_n:
    T = float(pd.Series(gaps_by_n[EXPECTED_SITES]).median())
else:
    T = float(pd.Series([g for gs in gaps_by_n.values() for g in gs]).median())
print(f"\n基准间隔 T = {T:.1f}s(自动估算)")

# 警告:如果某类 wafer 的 max gap 明显超过 1.5T,说明可能是中间漏测(场景 B),需要另写逻辑
for n, gs in gaps_by_n.items():
    if max(gs) > 1.5 * T:
        print(f"⚠️  {n} 行 wafer 出现 gap > 1.5T 的情况,可能是中间漏测,需要人工确认")

# 3. 给每行分配 site_order
df['_n'] = df.groupby(group_keys)['SPC_Value'].transform('size')
df['_cc'] = df.groupby(group_keys).cumcount() + 1

# ===== 方案 A(默认启用):不完整 wafer 缺的是末尾(缺 site_5,或 site_4-5)=====
df['site_order'] = df['_cc']

# ===== 方案 B(暂时注释):不完整 wafer 缺的是开头(缺 site_1,或 site_1-2)=====
# df['site_order'] = df['_cc'] + (EXPECTED_SITES - df['_n'])

df = df.drop(columns=['_n', '_cc'])
df = df[df['site_order'].between(1, EXPECTED_SITES)].copy()

# 4. 用 site_order 取对应的 SITE_x 值,覆盖原 SPC_Value
site_cols = [f'SITE_{i}' for i in range(1, EXPECTED_SITES + 1)]
site_arr = df[site_cols].values
df['SPC_Value'] = site_arr[np.arange(len(df)), df['site_order'].values - 1]

# 5. site_order one-hot
df = pd.get_dummies(df, columns=['site_order'], prefix='pos')
for i in range(1, EXPECTED_SITES + 1):
    col = f'pos_{i}'
    if col not in df.columns:
        df[col] = 0
pos_cols = [f'pos_{i}' for i in range(1, EXPECTED_SITES + 1)]
df[pos_cols] = df[pos_cols].astype(int)

print("\n拆分后各位置行数分布:")
print(df[pos_cols].sum())
print(f"总行数: {len(df)}")