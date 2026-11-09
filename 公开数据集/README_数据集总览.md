# 项目公开地震数据集

整理日期：2026年11月3日。依据上级目录《公开地震数据与算法复现.md》的第二节收集五类数据，并保留第四节明确引用的 FaultSegV1 工程对照数据。

**全部资料已于2026年11月3日14:42完成下载并通过最终校验。** 完成状态见同目录《下载进度.md》及 `00_下载记录/下载任务状态.json`，核验结果见 `00_下载记录/核验结果.json`。后续实验使用同级的“算法复现工作区”，原始资料作为数据来源保留。

| 目录 | 内容 | 官方入口与版本 |
|---|---|---|
| `01_FaultSeg3D` | 原始轴顺序的训练/验证地震与断层标签、预测示例、作者代码与预训练模型 | [原作者 faultSeg](https://github.com/xinwucwp/faultSeg)、[训练验证目录](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)、[模型目录](https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd) |
| `02_Thebe` | 配套 NPZ 地震体及断层标签，涵盖 train/val/test；原始解释、许可、处理说明和作者代码 | [Harvard Dataverse，10.7910/DVN/YBYGBK](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK) |
| `03_CRACKS_F3_v2` | `images.zip`、`Fault segmentations.zip`、作者代码与解压内容 | [Zenodo v2，13926822](https://zenodo.org/records/13926822) |
| `04_HorizonTracker` | 小型地震及瞬时相位数组、教程 Notebook、作者代码 | [HorizonTracker](https://github.com/ajbugge/HorizonTracker) |
| `05_DeepRGT` | 地震/RGT 原始分卷、断层标签、v1.0.0 作者代码 | [原始数据 4536561](https://zenodo.org/records/4536561)、[代码归档 5044922](https://zenodo.org/records/5044922) |
| `06_FaultSegV3_CIG_Bench` | `FaultSegV1.zip`：200 训练对、20 验证对；FaultSegV3 作者工程 | [作者指定的 CIG-Bench 数据仓库](https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset)、[FaultSegV3](https://github.com/douyimin/FaultSegV3) |
| `07_统一比较框架` | large-bench-geo 作者代码，供后续复现参考 | [large-bench-geo](https://github.com/olivesgatech/large-bench-geo) |
| `00_下载记录` | 来源页面元数据、逐文件清单、校验值、自动下载/重试/核验记录 | 以 `文件清单.csv`、`下载清单.json`、`核验结果.json` 为准 |

## 文件命名与保存原则

一级目录采用“编号_数据集名称”。下载的原始文件保留作者名称；作者仓库压缩包附带 Git 提交号，便于固定版本。`已解压` 为从原包提取的可用内容；原始压缩包同时保留。没有训练模型或生成预测结果。

本目录独立保存公开资料；上级说明文档和其他项目资料均未改动。Thebe 的 TEST 分区原样保留，收集与核验不代表将 TEST 用于训练或模型选择。

## 已核实的格式与使用注意

### FaultSeg3D

- 训练地震与断层标签各 200 个，验证各 20 个；每个 `.dat` 含 `128³` 个小端 float32，共 8,388,608 字节。
- 200个训练地震文件、训练断层原始压缩包、F3预测示例和5个模型/结构文件均从原作者 Google Drive 的公开下载入口取得；40个验证地震/标签文件从原作者仓库提取。
- CIG-Bench `FaultSegV1.zip` 镜像已通过官方 SHA-256 校验；镜像 `seis`、`fault` 都是 float32，尺寸 `128×128×128`。镜像 `(t,h,w)` 数组按 `transpose(1,2,0)` 转回 `.dat` 原始序列后，**200个训练地震体、200个训练断层标签全部与原作者数据逐字节一致**。比较数量和逐文件来源见 `00_下载记录/FaultSeg3D镜像恢复记录.json`。
- **镜像的验证编号采用不同顺序**（例如镜像验证编号2对应原作者编号10）。`01_FaultSeg3D` 的40个验证地震/标签文件直接从原作者仓库提取，保留作者原编号；`06_FaultSegV3_CIG_Bench` 的镜像原包保持原样。编号对应关系也记录在镜像恢复记录中。
- `train/fault.zip` 保留原作者训练标签压缩包，`train/fault` 提供已恢复且核对一致的200个标签文件。
- `06_FaultSegV3_CIG_Bench` 保留镜像原包，方便 PyTorch 工程使用；与 `01_FaultSeg3D` 是同一来源的数据，不计算成第二个独立数据集。

### Thebe

- 仅下载成套 `.npz` 数据，避免重复下载同内容的 `.npy` / `.zip` 版本。
- 原存档提供 9 个训练分块、2 个验证分块和7个测试分块，每种分块均有地震和断层标签；通过 NPZ 内的数组头核对地震—标签尺寸。
- `seistrain2.npz` 与下载页另列的 `seistrain2_updated.zip` 不混用。本轮使用 NPZ 配套版本，原存档完整文件列表保存在 `Thebe_Dataverse.json`，后续如需切换版本需独立记录。
- 不用普通文件拼接恢复 `.npy` / `.npz`。恢复完整体时应按作者 Notebook 的分块轴处理。

### CRACKS/F3

- 保留全部解释者、专家和置信颜色标注；初期优先使用 `expert`。
- 蓝色确定断层、橙色确定非断层；绿色不确定、白色未标注不能直接作负样本。
- 这些是剖面显示图像；没有另行下载用于地震相分类的 Microsoft F3 标签。

### HorizonTracker

- 作者仓库内含 `seismic_volume.npy`、`instPhase_volume.npy`，以及 `HorizonTracker-tutorial.ipynb`。数据和代码均在 `作者示例与代码/已解压` 内。
- 这是算法演示数据，不是完整人工层位面的监督训练集。

### DeepRGT

- 保留 Zenodo 指定版本的 `seis.zip00`—`seis.zip28`（29 卷）、`rgt.zip00`—`rgt.zip26`（27 卷）以及 `fault.zip`。原始归档总量约 29.6 GB（十进制）。
- **分卷的真实外层格式是 gzip/tar，内层才是 `seis.zip` 或 `rgt.zip`。** `rgt.zip00` 的文件头已经核实为 gzip，解压后的 tar 首项为 `rgt.zip`。不能仅凭 `.zipNN` 后缀把每一卷当作普通 ZIP 解压。完整恢复需依编号顺序读取全部分卷，再解开 gzip/tar，最后解开内层 ZIP。
- 三类资料已核验为500组配对体，和说明文档中的“400组”有差异；具体编号及完整性检查见 `RGT完整性核验.json`。
- 检查过的 ModelScope DeepRGT 镜像使用 float16；本轮没有用它替代 Zenodo 原始数据，也没有保留该镜像大文件。
- 原始分卷只保存一份；核验已流式读取外层归档，临时恢复内层 ZIP 并通过 CRC 检查，核验临时 ZIP 已自动清理，没有长期保存几十 GB 的重复合并文件。若完整展开为训练数据，需另留充足空间。

## 校验和下载状态

- Zenodo、Dataverse、ModelScope 的文件大小与官方 MD5 / SHA-256 逐一核对。仓库和 Google Drive 文件保存本地 SHA-256，供以后复核。
- 小型 ZIP 已解压并核验 CRC。Thebe 配对数组尺寸已经核对；RGT 外层 gzip、内层 ZIP 和三类体编号已经核验。
- `下载进度.md` 保存最终进度；`下载任务状态.json` 为 `complete`，`核验结果.json` 为 `verified`，下载任务已经结束。
- 下载脚本、来源与核验记录保存在 `00_下载记录`。探测片段、测试副本、零散传输日志和工具缓存已集中隔离至其下的“待清理临时文件”；这些文件没有永久删除。若需要重新使用Google Drive下载工具，可从该目录恢复“下载工具依赖”。

使用、再分发和论文引用遵循各原作者的许可证与引用要求。许可和来源元数据随数据保留。
