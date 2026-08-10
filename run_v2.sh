#!/bin/bash
# v2 全套實跑：搜尋 → 正式評估 → 架構對照 → 延遲量測。
# 延遲量測刻意排在最後、且不與訓練並行——量測期間機器要閒著，
# 否則量到的是「搶得到多少算力」而不是模型延遲。
cd "$(dirname "$0")"
P=./.venv/bin/python
echo "=== [1/4] 超參搜尋 $(date +%H:%M) ==="
$P src/search.py --configs 32 --folds 3 2>&1 | tee results/search.log
echo "=== [2/4] 正式評估 $(date +%H:%M) ==="
$P src/final_eval.py --seeds 10 2>&1 | tee results/final_eval.log
echo "=== [3/4] 架構對照 $(date +%H:%M) ==="
$P src/arch_compare.py --seeds 10 2>&1 | tee results/arch_compare.log
echo "=== [4/4] 延遲量測 $(date +%H:%M) ==="
$P src/latency.py 2>&1 | tee results/latency.log
echo "=== 全部完成 $(date +%H:%M) ==="
