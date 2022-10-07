## 说明

2hours_price_setting.csv是对all_info_qugang_zhida_TF_2019.csv以两小时为间隔划分后的数据

### 列名含义说明

LDD_SVVD		航次代号

POL_CDE		出发港口缩写

POL_NME		出发港口中文

POD_CDE		到达港口缩写

POD_NME		到达港口中文

WBL_AUD_DTDL	时间段（指该时间点前的两个小时）

MAIN_SVVD_SAILING_DATE	发船时间

NUM			销量（两小时总和）

PRICE			价格（两小时最终价格）

### 其他说明

- 价格是OFT（海运费）、BUC（燃油附加费）、SLF（铅封费）、CRM（箱体附加费）和THC（码头装卸费）的总和
- 仅考虑了all_info_qugang_zhida_TF_2019.csv中箱型为20GP的箱子，剔除了其他箱型
- 每个航次的时间段内销量价格分开统计