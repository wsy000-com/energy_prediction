import matplotlib.pyplot as plt

# 注意：这里的 df 应该使用你上一步 melt 之后得到的数据框（也就是 df_ml）
# df = df_ml.copy()

# 获取所有的 site (1 到 5) 和设备
sites = sorted(df['site'].unique())
groups = df['PROC_EQUIP_ID'].unique()

# 动态生成子图：有几个 site 就画几行，每行高度设为 3
fig, axes = plt.subplots(len(sites), 1, figsize=(20, 3 * len(sites)), sharex=True)

# 确保 axes 是个列表（应对极端情况只有一个 site 时报错）
if len(sites) == 1:
    axes = [axes]

for i, site_val in enumerate(sites):
    # 筛选当前 site 的数据
    df_site = df[df['site'] == site_val]

    for equip in groups:
        # 筛选该设备的数据
        mask = df_site['PROC_EQUIP_ID'] == equip
        # 提取时间(X)和值(Y)，同时剔除空值
        df_plot = df_site[mask][['X_TIME', 'SPC_Value']].dropna()

        if len(df_plot) > 0:
            axes[i].plot(df_plot['X_TIME'], df_plot['SPC_Value'],
                         linewidth=0.8, label=str(equip), alpha=0.7)

    axes[i].set_ylabel(f'Site {site_val}\nSPC_Value', fontsize=12)
    axes[i].tick_params(labelsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(loc='upper right', fontsize=8, ncol=3)

axes[-1].set_xlabel('X_TIME', fontsize=13)
fig.suptitle('SPC_Value Trend by Site and Equipment', fontsize=16, y=1.02)

plt.tight_layout()
plt.show()