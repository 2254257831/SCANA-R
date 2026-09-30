# SCANA-R 中文说明

本仓库对应当前修订论文《成功示范驱动的动作噪声校准与仿真恢复学习》。
现在是本地开源准备版，已整理代码、复现实验入口、证据和项目主页，尚未上传 GitHub。

## Independent-library and cost study (v46)

The current manuscript adds **11,040 evaluation episodes**. Five independently rebuilt
libraries per bimanual task yield **89.7% / 49.7% SCANA-R**, **81.7% / 38.7% Gaussian
recovery**, and **92.7% / 54.7% Gaussian at matched total active CPU cost**.
Three-library repeats on six Meta-World tasks yield macro success **89.9% SCANA-R,
89.5% Gaussian, and 93.1% original repetition**. Recovery collection helps the tested
bimanual policies, but calibration is not established as necessary or superior at
matched compute. Intervals resample libraries, within-library policies and shared
layouts; original source demonstrations remain fixed.

Channel interventions locate the transfer noise failure at a narrowly represented
joint input. A fixed temporal ensemble recovers much of the success lost under
single-step replanning. These deployment diagnostics use frozen original libraries.

See [rerun instructions](experiments/independent_libraries_v46/README.md),
[recorded results](results/independent_libraries_v46), and the new figures on the
[local project page](docs/index.html#independent-libraries). The studies below remain
historical evidence and are not pooled with this repetition.

## 新增 MuJoCo 八任务扩展

本次完成 11,600 次执行：冻结 ACT 策略的 5,600 次部署测试，以及六项 Meta-World 操作的 6,000 次测试。覆盖到达、推移、抓放、开门、开抽屉、按按钮和原双臂交接、插入。六任务常规宏平均为 SCANA-R 89.5%、原始重复 91.8%，未证明整体提升；双臂新布局仍有收益，但末段扰动和观测噪声下有明显失败。

论文新增图5—7及表10—12，原四张PPT方法图保留。查看[视频与结果](docs/mujoco-gallery.html)、[完整协议](docs/mujoco-extension.md)。运行 `python scripts/summarize_extensions.py` 可仅凭仓库CSV重算全部扩展统计。Meta-World须使用独立Python环境与 `requirements-metaworld.txt`，不能覆盖ACT的MuJoCo版本。所有新素材都是仿真，动作重规划不等于语言任务规划。

## 当前方法

对每个任务的 40 条训练源回合做五折交叉拟合，得到该回合未参与训练的策略预测误差。
从同任务、相邻阶段的误差库抽取 8 步扰动，保护夹爪通道并限制关节幅度；仿真执行扰动后
继续执行参考动作，以末态官方任务成功验收（ACT奖励4，Meta-World success字段）。失败时将同一扰动按 1、0.5、0.25、0 回退。
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

主页采用经典 CVPR 项目页常见的白底、居中题名、资源按钮、大幅视频、摘要、方法与结果结构。
首屏直接展示双任务仿真视频；下方两段同步双画面对比同一种子与布局的 Clean repeat 和 SCANA-R。
四段原始回放均逐步核验状态与奖励，状态最大误差为零。视频为特意选取的示例，不替代汇总成功率。
视频含本地 MP4、真实帧封面、原生播放控件和减少动态效果支持，无外部视频服务依赖。
渲染脚本、依赖与选例规则见 [视频说明](docs/videos.md)。四幅方法图沿用已认可的 PPT 视觉。
项目不是已接收的 CVPR 论文，页面没有虚构作者、会议标识或正式代码地址。
中文版论文摘要末尾暂用本地 `docs/index.html` 超链接；发布后替换为正式 Pages 地址。

本仓库使用 Apache-2.0；ACT 源码和资产沿用其原 MIT 许可。GitHub 注册邮箱只用于
本地提交身份，不作为主页公开联系方式。GitHub 用户名、作者署名和仓库地址仍待补充。
