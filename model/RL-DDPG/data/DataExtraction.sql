-- 目标：提取 IC29 航线上的 20GP 集装箱的数据，用以核查中转业务逻辑
-- 作者：胡永祎

-- 总思路：中远的业务逻辑是按集装箱收费，货运单 --> 集装箱 --> 运费
-- 1. 从货运单表格（WBL_WAYBILL_DATA）中提取满足条件的货运单
-- 2. 从箱路径表格（WBL_CNTR_LEG）中提取相关订单所涉及的集装箱
-- 3. 从运费表格（WBL_FRT_SUM）中提取每个集装箱的收费

-- 词汇表：
-- waybill n. 货运单
-- container n. 集装箱
-- freight n. 运费
-- port n. 港口
-- SVVD abbr. 船名航次
-- TF abbr. 即期市场
-- 注：想要了解更多数据库结构和表格的相关信息，请参考笔者的其他两个文档：“Route Model - 对航运过程建模” 和 “中远数据抽取代码注释版”，以及中远提供的数据库模型文件 "DTS.pdm"

-- 步骤 0: 删除创建的表格
-- DROP TABLE IC29_DATA_CHECK_WBL;
-- DROP TABLE IC29_DATA_CHECK_CNTR_LEG;
-- DROP TABLE IC29_DATA_CHECK_CNTR_FRT;

-- 步骤 1: 提取相关的货运单
-- 其中 WBL_TYPE, WBL_STATUS, BKG_MTHD 三个条件不用变动，旨在保证提取出来的是即期市场的正常订单
-- WBL_AUD_DT 是对提取订单的时间提出限制，MAIN_SVVD 是对船名航次提出要求（这里用来提取 IC29 航线的数据）
CREATE TABLE IC29_DATA_CHECK_WBL
AS
SELECT A.WBL_NUM, A.F_POL_CDE, A.L_POD_CDE, A.MAIN_SVVD, A.WBL_SOURCE, A.CGO_TYPE_NME, A.CGO_SUB_TYPE_NME, A.BKG_MTHD, A.SOC, A.WBL_AUD_DT
FROM WBL_WAYBILL_DATA A
WHERE A.WBL_AUD_DT>= TIMESTAMP '2019-10-1 00:00:00'
AND A.MAIN_SVVD LIKE 'IC29%S'
AND A.WBL_TYPE = 'NORMAL'
AND A.WBL_STATUS IN ('CONFIRM', 'BL_READY')
AND A.BKG_MTHD = 'TF'
ORDER BY WBL_NUM;

-- 显示提取的数据表格
SELECT * FROM IC29_DATA_CHECK_WBL;

-- 步骤 2: 提取货运单所对应的集装箱路径
-- 以运货单单号（WBL_NUM）为键，将步骤 1 中提取的运货单表格和集装箱表格（WBL_CNTR）合并，找到订单对应的集装箱
-- 以集装箱 UUID (WBL_CNTR_UUID) 为键再与集装箱箱路径表格（WBL_CNTR_LEG）合并，提取相关集装箱的运输路径的信息
-- CNTR_TYPE 条件是对集装箱的种类做出限制
-- 先后按照订单号（WBL_NUM），集装箱编号（CNTR_NUM）和中转次数（SEQ_NUM）排序，可以将同一订单号的集装箱数据聚集，同时看到每一个集装箱的中转，方便数据处理
CREATE TABLE IC29_DATA_CHECK_CNTR_LEG
AS
SELECT A.WBL_NUM, C.WBL_CNTR_UUID, C.SEQ_NUM, C.POL_CDE, C.POL_FCIL_CDE, C.POD_CDE, C.POD_FCIL_CDE
FROM IC29_DATA_CHECK_WBL A JOIN WBL_CNTR B
ON A.WBL_NUM = B.WBL_NUM
JOIN WBL_CNTR_LEG C
ON B.WBL_CNTR_UUID = C.WBL_CNTR_UUID
WHERE B.CNTR_TYPE = '20GP'
ORDER BY C.WBL_NUM, C.CNTR_NUM, C.SEQ_NUM;

-- 显示提取的数据表格
SELECT * FROM IC29_DATA_CHECK_CNTR_LEG;

-- 步骤 3: 提取相关集装箱的收费情况
-- 建立临时表格 TA 从步骤 1 中提取的运费单中收集所有相关的集装箱编号
-- 以集装箱编号为键，从运费单表格中（WBL_FRT_SUM）提取与收费相关的信息
CREATE TABLE IC29_DATA_CHECK_CNTR_FRT
AS
WITH TA AS(
SELECT WBL_NUM, WBL_CNTR_UUID
FROM IC29_DATA_CHECK_CNTR_LEG
GROUP BY WBL_NUM, WBL_CNTR_UUID
)
SELECT B.WBL_NUM, B.WBL_CNTR_UUID, B.CHRG_CDE, B.CHRG_NME, B.TTL_AMT
FROM TA
JOIN WBL_FRT_SUM B
ON TA.WBL_NUM = B.WBL_NUM
AND TA.WBL_CNTR_UUID = B.WBL_CNTR_UUID
ORDER BY B.WBL_NUM, B.WBL_CNTR_UUID;

SELECT * FROM IC29_DATA_CHECK_CNTR_FRT;
