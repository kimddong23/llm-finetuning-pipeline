#!/bin/bash
LOG_FILE="/Users/shinjuyong/Desktop/사이드 프로젝트/LLM 파인튜닝/llm-finetuning-pipeline/results/baseline_training.log"

echo "=========================================="
echo "Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 현재 스텝
CURRENT_STEP=$(grep -oE "Step [0-9]+" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+")
echo "Current Step: $CURRENT_STEP / 500"

# 진행률
if [ -n "$CURRENT_STEP" ]; then
  PROGRESS=$(echo "scale=1; $CURRENT_STEP * 100 / 500" | bc)
  echo "Progress: $PROGRESS%"
fi

# 최근 Loss
RECENT_LOSS=$(grep "loss=" "$LOG_FILE" | tail -1 | grep -oE "loss=[0-9]+\.[0-9]+" | cut -d= -f2)
echo "Recent Loss: $RECENT_LOSS"

# 메모리
RECENT_MEM=$(grep "mem=" "$LOG_FILE" | tail -1 | grep -oE "mem=[0-9]+\.[0-9]+GB" | cut -d= -f2)
echo "Memory: $RECENT_MEM"

# 평균 스텝 시간 (최근 10개)
echo ""
echo "Recent step times:"
tail -20 "$LOG_FILE" | grep "Training:" | tail -5

echo ""
echo "=========================================="
