# 压缩包内容说明：embryo_all_data.tgz

**文件**：`embryo_all_data.tgz`
**大小**：8.7 GB（压缩后）
**总条目**：36397 个文件/目录

> 包内路径以 `home/...` 开头（tar 打包时去掉了开头的 `/`），在新设备用 `sudo tar xzf embryo_all_data.tgz -C /` 解包即可还原为 `/home/...`。

---

## 一、表格（2 个文件）

### 1.1 临床主表

- **包内路径**：`home/data/cy0626/embryo_new/processed_0507_clinical/clinical_cleaned_full.csv`
- **大小**：1.1 MB
- **内容**：清洗后的 IVF 临床特征表。包含 `patient_embryo_key`（胚胎唯一键）、10 个数值特征（年龄、BMI、激素水平等）及 FH/LB 标签
- **用途**：被 `common.py` 的 `load_cohort()` 读取，供 step02、step03、step09 使用，是整个项目的核心输入

### 1.2 标签冲突表

- **包内路径**：`home/data/cy0626/embryo_new/processed_0507_clinical/label_conflict_report.csv`
- **大小**：2 KB
- **内容**：FH/LB 标签冲突样本的报告
- **用途**：step11 复制的 VaTEP 训练脚本直接读取（`CONFLICT_CSV` 常量），用于剔除标签冲突的样本

---

## 二、帧图像（1 个目录）

### 2.1 FEMI 处理后视频目录

- **包内路径**：`home/data/cy0626/embryo_new/Timelapse_femi_processed/`
- **大小**：143 MB（共 35059 帧）
- **内容**：经 FEMI 前处理后的逐帧图像。1151 枚胚胎各对应一个子目录，路径格式如 `140592/D2023.03.23_S01323_I4177_P-3/F0/`
- **用途**：`data/unique_video_matches.csv` 的 `embryo_dir` 列指向这里；step06 从此处扫描 `.jpg`/`.png` 帧图像，step07 用这些帧做 FEMI 特征提取

---

## 三、VaTEP 权重与脚本（3 个文件 + 1 个目录）

### 3.1 VaTEP 预训练编码器

- **包内路径**：`home/zcy/embryo_live2/experiments_0507_best_table/pretrain_pseudo/encoder.pth`
- **大小**：25 MB
- **内容**：VaTEP 预训练编码器权重
- **用途**：step10 检查是否存在，step11 作为 `--pretrained` 参数传入 VaTEP 训练脚本

### 3.2 LB 单任务训练脚本

- **包内路径**：`home/zcy/embryo_live2/experiments_0507_best_table/tune_vatep_lb_adapted.py`
- **大小**：~20 KB
- **内容**：VaTEP 训练脚本（LB 变体），仅预测活产（LB）标签
- **用途**：step10 检查存在性，step11 复制到输出目录并修改 `OUTPUT_DIR` 后启动

### 3.3 FH+LB 联合训练脚本

- **包内路径**：`home/zcy/embryo_live2/experiments_0507_best_table/tune_vatep_paper_adapted.py`
- **大小**：~20 KB
- **内容**：VaTEP 训练脚本（paper 变体），同时预测胎心（FH）和活产（LB）
- **用途**：同上，step11 复制并修改 `OUTPUT_DIR` 后启动

### 3.4 视频缓存目录

- **包内路径**：`home/zcy/embryo_live2/experiments_0507_best_table/video_cache_paper_12x48/`
- **大小**：原始 19.2 GB / 压缩后约 8.6 GB
- **内容**：1332 个 `.npy` 预处理视频张量（12×48 帧）
- **用途**：step11 作为 `--cache-dir` 传入 VaTEP 训练脚本，避免每次训练重复解码视频。这是包内最大头

---

## 四、包内没有的内容

| 内容 | 原因 |
|---|---|
| `Timelapse_1246/`（原始延时视频） | 无脚本读取，仅作身份记录 |
| `Timelapse_1246_roi_perframe/`（ROI 逐帧） | step10 只检查目录存在性，新设备建空目录即可 |
| FEMI 模型 `ihlab/FEMI` | step07 自动从 HuggingFace 下载 |
| `data/unique_video_matches.csv` 等 | 属于项目目录，随代码直接复制 |
