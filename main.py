columns_to_plot = feature_cols
n_cols = len(columns_to_plot)

if n_cols == 0:
    print("没有可画的列！")
else:
    n_rows = n_cols
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, n_rows * 3.5), sharex=True)

    if n_rows == 1:
        axes = [axes]

    # 【新增逻辑】：固定只看 site == 1 的数据，避免 X 被重复绘制 5 次
    # 注意：这里假设你传进来的 df 是 melt 之后的 df_ml，如果里面有 site 列就过滤
    if 'site' in df.columns:
        df_x_plot = df[df['site'] == 1].copy()
    else:
        df_x_plot = df.copy()

    groups = df_x_plot['PROC_EQUIP_ID'].unique()

    for i, col in enumerate(columns_to_plot):
        for equip in groups:
            # 【修改逻辑】：从过滤后的 df_x_plot 中取数据
            mask = df_x_plot['PROC_EQUIP_ID'] == equip
            series = df_x_plot.loc[mask, col].dropna()

            if len(series) > 0:
                axes[i].plot(series.index, series.values, linewidth=0.5,
                            label=str(equip), alpha=0.7)