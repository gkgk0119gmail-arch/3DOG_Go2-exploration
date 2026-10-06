#!/usr/bin/env bash
# 3DOG sim launcher. Usage: ./run.sh <command> [extra args passed to the Isaac Lab script]
#
#   walk-play     보행 정책 재생 (16마리, GUI)
#   walk-train    보행 정책 학습 (4096마리, 화면 없이)
#   walk-live     4096마리 PPO 보행 학습을 처음부터 GUI로 (발표 영상용, logs/rsl_rl/go2_walk_live에 따로 저장)
#   walk-stage S [overview|close]   학습 단계 재생, 4096마리 학습 장면: S = bad(iter 0) | weird(iter 50) | good(iter 1499) | <iter>
#   demo          창고 + VLP-16 + 고정 웨이포인트 순환 (GUI)
#   bim-demo      BIM 건물(기본 치과 의원, tools/ifc_to_isaac.py) + VLP-16 + 웨이포인트 순환 (GUI)
#   explore       ARiADNE 자율탐사 1마리 (GUI, 오른쪽 "ARiADNE Map" 탭)
#   loop          16마리 ARiADNE 자율탐사 무한 반복: 끝난 로봇은 지도 3초 보여주고 랜덤 리스폰 (GUI)
#   parallel      16마리 독립 탐사, 기본값 vs CEM 파라미터 비교 (GUI, "Parallel Exploration" 탭)
#   noise         16마리 독립 탐사, 깨끗한 라이다 vs VLP-16 노이즈+deskew 비교 (GUI)
#   search        CEM 파라미터 탐색 32마리 (화면 없이, logs/parallel/)
#
# Env vars you can prepend: GO2_CAM=overview|leader|<n>, GO2_LIDAR_VIS=1 (draw LiDAR hits), GO2_LIDAR_NOISE=1,
# GO2_DESKEW=1, GO2_SAVE_SCANS=1. See docs/SIM_GUIDE.md for everything else.
set -e
cd "$(dirname "$0")"
PY=~/miniconda3/envs/isaac/bin/python
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1
CEM=$PWD/logs/parallel/cem_best_2026-10-01.json
cmd=${1:-help}
shift || true

case "$cmd" in
  walk-play)  exec $PY scripts/rl.py play --task Go2-Factory-Walk-Play --num_envs 16 --viz kit "$@" ;;
  walk-live)  GO2_SHOW_CAM=${GO2_SHOW_CAM:-overview} exec $PY scripts/rl.py train --task Go2-Factory-Walk-Showcase \
                --num_envs 4096 --viz kit --experiment_name go2_walk_live "$@" ;;
  walk-stage) stage=${1:-good}; shift || true
              case "$stage" in bad) it=0 ;; weird) it=50 ;; good) it=1499 ;; *) it=$stage ;; esac
              cam=overview; if [ "${1:-}" = close ] || [ "${1:-}" = overview ]; then cam=$1; shift; fi
              GO2_SHOW_CAM=$cam exec $PY scripts/rl.py play --task Go2-Factory-Walk-Showcase --train_env_cfg \
                --num_envs 4096 --viz kit --checkpoint $PWD/logs/rsl_rl/go2_factory_walk/2026-09-30_23-09-18/model_$it.pt "$@" ;;
  walk-train) exec $PY scripts/rl.py train --task Go2-Factory-Walk --num_envs 4096 "$@" ;;
  demo)       exec $PY scripts/rl.py play --task Go2-Warehouse-Demo-Play --num_envs 1 --viz kit --real-time "$@" ;;
  bim-demo)   exec $PY scripts/rl.py play --task Go2-Bim-Demo-Play --num_envs 1 --viz kit --real-time "$@" ;;
  explore)    exec $PY scripts/rl.py play --task Go2-Warehouse-Explore-Play --num_envs 1 --viz kit --real-time "$@" ;;
  loop)       GO2_CAM=${GO2_CAM:-overview} GO2_PAR_MODE=loop GO2_PAR_PARAMS=$CEM \
                exec $PY scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16 --viz kit "$@" ;;
  parallel)   GO2_CAM=${GO2_CAM:-leader} GO2_PAR_MODE=compare GO2_PAR_PARAMS=$CEM \
                exec $PY scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16 --viz kit "$@" ;;
  noise)      GO2_CAM=${GO2_CAM:-leader} GO2_PAR_MODE=compare GO2_COMPARE=noise GO2_LIDAR_NOISE=1 \
                GO2_DESKEW=${GO2_DESKEW:-1} GO2_PAR_PARAMS=$CEM \
                exec $PY scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16 --viz kit "$@" ;;
  search)     GO2_PAR_MODE=search GO2_PAR_STARTS=4 GO2_PAR_GENS=${GO2_PAR_GENS:-10} \
                exec $PY scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 32 "$@" ;;
  *)          sed -n "2,/^set -e/p" "$0" | grep "^#" | sed "s/^# \{0,1\}//" ;;
esac
