# ====== 诊断:wafer 行数分布 + 时间间隔 ======
wafer_counts = df.groupby(['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']).size()
print("每个 wafer 的行数分布:")
print(wafer_counts.value_counts().sort_index())
print(f"总 wafer 数: {len(wafer_counts)}\n")

# 时间间隔分析
def get_intervals(group):
    times = group.index.sort_values()
    return times.to_series().diff().dt.total_seconds().dropna().tolist()

five_gaps, four_gaps = [], []
for _, g in df.groupby(['PROC_EQUIP_ID', 'LOT_ID', 'WAFER_ID']):
    intervals = get_intervals(g)
    if len(intervals) == 4:
        five_gaps.append(intervals)
    elif len(intervals) == 3:
        four_gaps.append(intervals)

if five_gaps:
    print("5 行 wafer 的相邻时间间隔(秒)统计:")
    print(pd.DataFrame(five_gaps, columns=['g1→2','g2→3','g3→4','g4→5']).describe().round(1))
if four_gaps:
    print("\n4 行 wafer 的相邻时间间隔(秒)统计:")
    print(pd.DataFrame(four_gaps, columns=['g1→2','g2→3','g3→4']).describe().round(1))