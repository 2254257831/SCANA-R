# SCANA-R 中文说明

本仓库对应当前修订论文《成功示范驱动的动作噪声校准与仿真恢复学习》。
现在是本地开源准备版，已整理代码、复现实验入口、证据和项目主页，尚未上传 GitHub。

## 当前方法

对每个任务的 40 条训练源回合做五折交叉拟合，得到该回合未参与训练的策略预测误差。
从同任务、相邻阶段的误差库抽取 8 步扰动，保护夹爪通道并限制关节幅度；仿真执行扰动后
继续执行参考动作，以末步奖励等于 4 验收。失败时将同一扰动按 1、0.5、0.25、0 回退。
只提取扰动结束后的实际观测与实际执行动作作为恢复训练窗口。

这改变了旧 SCANA 的“固定观测，只修改标签”机制。`legacy/` 中的历史代码不得当作
当前算法入口；原论文历史离线表格需要原数据，不能声称本仓库凭空重建这些数据。

## 最快使用

1. 使用 Python 3.10，按照英文 [README](README.md) 安装依赖。
2. `python -m unittest discover -s tests -v`：验证算法契约和证据一致性。
3. `python scripts/summarize.py`：直接从仓库的 3,200 条记录重算成功率和配对区间。
4. `python scripts/import_artifacts.py ../SCANA-R-artifacts-v1.zip`：导入仓库外的独立证据包。
5. `python scripts/run_scana_r.py replay-check`：用记录权重重新运行一条轨迹，逐元素比对。
6. `python scripts/run_scana_r.py audit`：检查恢复窗口与实际观测、执行动作、末态奖励一致。

从零采集和训练四个当前方法的完整命令见英文 README。所有命令均在仓库根目录执行。
大文件导入 `artifacts/`，新结果写入 `outputs/`；两个目录都被 Git 忽略。

## 主页与论文

双击 [docs/index.html](docs/index.html) 即可本地预览，也可运行：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

主页采用顶会项目页常见的题名、方法总览、摘要、结果、复现资源结构，四幅图沿用已认可的
PPT 视觉。项目不是已接收的 CVPR 论文，页面没有虚构作者、会议标识或正式代码地址。
中文版论文摘要末尾暂用本地 `docs/index.html` 超链接；发布后替换为正式 Pages 地址。

本仓库使用 Apache-2.0；ACT 源码和资产沿用其原 MIT 许可。GitHub 注册邮箱只用于
本地提交身份，不作为主页公开联系方式。GitHub 用户名、作者署名和仓库地址仍待补充。
