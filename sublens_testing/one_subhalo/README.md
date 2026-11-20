# 单子晕基础测试

只注入一个子晕，位置和强度不传入提名或精修。使用原始观测、RMS 和 PSF，保持与旧结果相同的有效像素。

改进：亮度加权 centroidal Delaunay 布点；留出预测选择节点数、随机种子和正则；触边时扩展候选搜索范围；多起点联合精修；每次重新求源；外层更新网格。两种质量假设在每一轮使用相同的网格、正则和 macro 自由度。

旧 macro+halo χ²/Npix=1.4612；新 macro+halo χ²/Npix=1.0556。

位置误差=5.23 mas；强度相对误差=-3.93%；halo 收敛=True；macro 收敛=True。

χ²/Npix 按相同的 3744 个有效像素计算，没有扣除有效自由度；它不是 reduced χ²。源布点和正则改变后 macro-only χ²/Npix 也会改变，不能用 Δχ² 的增大直接宣称探测显著性提高。

源验证指标是在当前拟合透镜几何下的条件预测，不能视作整套发现流程的独立盲测成绩。macro 的归一化和两个剪切分量可调整，形状、斜率和中心固定在模拟输入值。新流程的子晕存在阈值尚未经过大样本零注入校准。

运行：`.venv/bin/python sublens_testing/one_subhalo/run.py --case base`；独立噪声复测：`--case repeat`；零注入检查：`--case null`。缓存保存于各结果目录，使用 `--force` 可重新计算。子晕和源的真值只用于生成与结果评估。每个候选分别比较单子晕假设，5 个提名不代表同时拟合或注入 5 个子晕。强度是截断子晕模型的角尺度参数，单位 arcsec，不是物理质量。

独立噪声复测（seed=314159）：χ²/Npix=0.9377，位置误差=4.10 mas，强度误差=-3.65%。

零注入检查：macro χ²/Npix=0.9452，加扰动后=0.9430，最佳强度=0.001629 arcsec。该弱扰动不能自动解释为检出。

文件：data/ 保存 image、RMS、PSF、mask、输入 macro 的 FITS，以及独立的真值；improved/、repeat/、null/ 各保存结果 FITS/JSON、macro 和 macro+halo 残差图、源重构图、候选图及优化日志；reference/ 是改进前的模型和残差；development/ 仅保存探索过程，不作为最终结果。根目录 before_after_residuals.png 用相同 ±4 RMS 色标比较旧、新残差。FITS 的 MODEL_MACRO/RESID_MACRO 和 MODEL/RESIDUAL 分别是两种假设的图像及 observed−model；SOURCE_NODES 包含源节点和系数，CANDIDATES/METADATA 包含拟合参数和配置。

基础调用（已读入 observation）：

```python
result = observation.search_pinn(mesh_strategy='centroidal', source_nodes=300, regularization=20.)
result = result.refine_joint(n_starts=2, retry_failed=True, maxiter=120)
result = result.refine_adaptive(source_nodes=(200, 350, 500), regularizations=(2., 20., 200.),
                                seeds=(11, 22, 33, 44), n_starts=3, maxiter=120)
result.plot_residuals()
```
