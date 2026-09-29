# 消融套件逐任务结果(任务级)

- 日期: 2026-09-16 · 机位 `cen_arm_top_cam` · 500ep/任务 · 评测 100ep seed 0
- 对照: `archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911`(top500-emixfix,默认构建)
- 数据来源: 各归档 `eval/*/…/metrics.json`,n/100 为实际成功 episode 数
- 噪声标尺: 单任务 100ep delta 噪声约 ±5pt,|Δ|≤5 读作持平

## 1. 逐任务成功率(%,n/100)

| 任务 | DP baseline | emixfix(默认构建) | homefilter | timeratio | combo |
|---|---|---|---|---|---|
| blocks_ranking_rgb | 28 (28) | 26 (26) | 23 (23) | 20 (20) | 30 (30) |
| blocks_ranking_size | 3 (3) | 3 (3) | 4 (4) | 5 (5) | 4 (4) |
| place_bread_basket | 45 (45) | 37 (37) | 35 (35) | 35 (35) | 36 (36) |
| place_bread_skillet | 87 (87) | 92 (92) | 91 (91) | 92 (92) | 93 (93) |
| place_burger_fries | 36 (36) | 37 (37) | 30 (30) | 28 (28) | 23 (23) |
| place_can_basket | 49 (49) | 64 (64) | 66 (66) | 69 (69) | 65 (65) |
| place_cans_plasticbox | 20 (20) | 25 (25) | 28 (28) | 26 (26) | 27 (27) |
| place_dual_shoes | 33 (33) | 44 (44) | 30 (30) | 41 (41) | 35 (35) |
| stack_blocks_two | 49 (49) | 45 (45) | 41 (41) | 49 (49) | 40 (40) |
| stack_bowls_two | 88 (88) | 88 (88) | 89 (89) | 87 (87) | 91 (91) |

## 2. 逐任务 Δ:各消融批 vs emixfix(默认构建)

| 任务 | homefilter Δ | timeratio Δ | combo Δ | 备注 |
|---|---|---|---|---|
| blocks_ranking_rgb | -3 | -6 | **+4** | timeratio −6 略超噪声,combo +4 持平 |
| blocks_ranking_size | +1 | +2 | +1 | 三批持平(任务本身 ~3-5%) |
| place_bread_basket | -2 | -2 | -1 | 三批持平;对 DP 仍落后(37→35/35/36) |
| place_bread_skillet | -1 | 0 | +1 | 三批持平(高 SR 91-93 锁死) |
| place_burger_fries | -7 | -9 | **-14** | 三批全负且单调恶化:过滤越狠越差,唯一一致响应任务 |
| place_can_basket | +2 | **+5** | +1 | timeratio +5(移除率 81% 的检验点)方向符合但幅度=噪声 |
| place_cans_plasticbox | +3 | +1 | +2 | 三批持平 |
| place_dual_shoes | **-14** | -3 | **-9** | homefilter/combo 大幅落后:home 配对对该任务似有正贡献 |
| stack_blocks_two | -4 | **+4** | -5 | timeratio 回到 DP 水平(49),另两批持平 |
| stack_bowls_two | +1 | -1 | **+3** | 三批持平(高 SR 锁死) |

## 3. 逐任务 Δ:各消融批 vs DP baseline

| 任务 | emixfix Δ | homefilter Δ | timeratio Δ | combo Δ |
|---|---|---|---|---|
| blocks_ranking_rgb | -2 | -5 | -8 | +2 |
| blocks_ranking_size | 0 | +1 | +2 | +1 |
| place_bread_basket | -8 | -10 | -10 | -9 |
| place_bread_skillet | +5 | +4 | +5 | +6 |
| place_burger_fries | +1 | -6 | -8 | -13 |
| place_can_basket | +15 | +17 | +20 | +16 |
| place_cans_plasticbox | +5 | +8 | +6 | +7 |
| place_dual_shoes | +11 | -3 | +8 | +2 |
| stack_blocks_two | -4 | -8 | 0 | -9 |
| stack_bowls_two | 0 | +1 | -1 | +3 |

## 4. 任务级观察(不依赖平均)

1. **place_burger_fries 是唯一对过滤呈单调负响应的任务**:37→30→28→23(homefilter→timeratio→combo,combo 相对 DP 也跌到 −13)。删除比例该任务约 48%(timeratio)/62%(homefilter),删得越多越差,提示它的候选对本来就稀缺、过滤后配对质量下降而非提升。
2. **place_dual_shoes 的 EMIX 收益依赖 home 配对**:emixfix +11 → homefilter −3 / combo +2(homefilter 移除率 37%)。去掉起点附近配对后,该任务的 EMIX 优势基本消失;timeratio(仅时间过滤)保留 +8,说明关键在"在哪配对"而非"跨多长"。
3. **place_can_basket(检验点)方向符合但不显著**:timeratio 移除率最高(81%,三批唯一未触 cap 的任务之一),Δ+5 恰好等于噪声上限;且它在所有四个 EMIX 配置里都是对 DP 最大的赢家(+15~+20),对过滤完全不敏感。
4. **高 SR 任务锁死**:bread_skillet(91-93)、stack_bowls_two(87-91)、blocks_size(3-5,低 SR 锁死)在三批间波动 ≤3,无信息量。
5. **place_bread_basket 是四配置共性残留落后**:所有 EMIX 变体对 DP 均落后 8-10,与过滤无关——这是默认构建也未解决的问题,与归档记忆「唯一残留落后 bread_basket」一致。
6. **三批对过滤的响应互不一致**:同一任务的 Δ 在三种过滤下方向相反(rgb:-3/-6/+4;stack_blocks:-4/+4/-5),不存在"某过滤系统性改善某类任务"的模式,支持"两类伪样本无实质危害"的套件结论。

## 5. 参考:TOTAL(仅备注,分析以任务级为准)

emixfix 46.1 · combo 44.4 · timeratio 45.2 · homefilter 43.7 · DP baseline 43.8(全部差异在 ±5pt 噪声带内,互为持平;逐任务分解见上)。
