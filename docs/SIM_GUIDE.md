# 3DOG sim (Isaac Lab 3.0 / Isaac Sim 6.0.1, RTX 5090)

```bash
conda activate isaac
cd ~/Desktop/3DOG/sim
```
처음 실행 시 EULA 질문에 `Yes`. GUI는 `--viz kit`, 빼면 화면 없이 실행.

**빠른 실행:** `./run.sh` (도움말) · `./run.sh explore` · `./run.sh parallel` · `./run.sh noise` · `./run.sh demo` · `./run.sh walk-play`

## 태스크

| Task | 내용 |
|---|---|
| `Go2-Factory-Walk` | 공장 바닥용 blind 보행 (전진·회전 위주 명령, 몸통 높이·고관절·수평 보상) |
| `Go2-Factory-Walk-Play` | 위 정책 재생 (16~50마리) |
| `Go2-Warehouse-Demo-Play` | 창고 씬 + VLP-16 + 제자리 회전 후 전진 웨이포인트 루프 |
| `Go2-Stock-Blind-Rough(-Play)` | 험지 blind 보행 (초기 버전) |
| `Go2-URDF-Blind-Rough(-Play)` | 자체 URDF→USD 변환본. 관절 불안정, 보류 |

## 명령어

```bash
# 보행 학습 (화면 없이)
python scripts/rl.py train --task Go2-Factory-Walk --num_envs 4096

# 창고 데모 (GUI)
python scripts/rl.py play --task Go2-Warehouse-Demo-Play --num_envs 1 --viz kit

# 창고 데모 녹화 (화면 없이, logs/.../videos/play/*.mp4)
python scripts/rl.py play --task Go2-Warehouse-Demo-Play --num_envs 1 --video --video_length 1500
```
`play`는 `logs/rsl_rl/<experiment>/` 에서 가장 최근 체크포인트를 자동으로 불러옴.

## 구성

- `go2_lab/locomotion.py` — 보행 태스크 설정 (보상, 명령, 지형)
- `go2_lab/waypoint_command.py` — 제자리 회전 → 정렬 후 전진 웨이포인트 추종 (명령 모듈)
- `go2_lab/factory.py` — 창고 씬, VLP-16 (16ch, ±15°, 360°, 0.2°, 10 Hz, 100 m), 웨이포인트
- `go2_lab/lidar_io.py` — VLP-16 스캔 저장 (10 Hz, 센서 좌표계 점군 + ring + 센서 자세) → `logs/lidar_scans/<시각>/*.npz`
- `assets/go2_vlp16.usda` — Go2 + VLP-16 합성 로봇 (base 아래 (0.2, 0, 0.08)에 마운트)
- `assets/vlp16/vlp16_mount.urdf` — VLP-16 + 2.3 cm 스페이서 (메시: velodyne_simulator STL)
- `assets/warehouse/` — Isaac Sim `full_warehouse.usd` 로컬 사본
- `assets/maps/full_warehouse.{png,npz}` — 창고 점유 지도 (0.1 m, 높이 0.05–1.0 m 투영)
- `tools/usd_occupancy_map.py` — USD 씬 → 점유 지도
- `tools/merge_fixed_links.py` — URDF 고정 링크 병합 (질량·관성 합산)
- `assets/go2/` — Unitree 공식 URDF 변환본 (보류)

## ARiADNE 자율탐사

```bash
# GUI (계속 떠 있음)
python scripts/rl.py play --task Go2-Warehouse-Explore-Play --num_envs 1 --viz kit --real-time
# 화면 없이, 탐사 완료(또는 N초) 후 자동 종료
GO2_EXPLORE_MAX_S=400 python scripts/rl.py play --task Go2-Warehouse-Explore-Play --num_envs 1
# 스냅샷 → GIF
python tools/snapshots_to_gif.py logs/exploration/<run>
```

흐름: VLP-16 (10 Hz) → `exploration/occupancy_mapper.py` (octomap `/projected_map` 대체, 0.4 m, 높이 0.15–1.2 m 투영, 20 m)
→ `exploration/ariadne_planner.py` (ARiADNE-ROS-Planner 코어, RA-L 2024 체크포인트, `rl_planner.launch` 파라미터, 2.5 Hz)
→ `exploration/exploration_command.py` (웨이포인트 확정 규칙 + 제자리 회전 후 전진) → 보행 정책.

GUI에서는 오른쪽 위 Stage 패널 옆에 **"ARiADNE Map" 탭**이 생겨 2D 지도(점유 지도, 그래프, 탐사 경계, 궤적, 목표점)와 상태가 0.5초마다 갱신됨. 패널 경계를 끌어 크기 조절 가능.

웨이포인트까지는 점유 지도 위 A*(`exploration/astar.py`: 로봇 반경 0.35 m 차단, 1 m 이내 벽 근접 비용, 8방향, 시야선 단축)로 경로를 만들고 1.2 m 앞 지점을 따라감. ARiADNE 계획은 백그라운드 스레드.

평가: `python tools/eval_exploration.py logs/exploration/<run>` → 지도 정밀도/재현율/오판(장애물→빈칸)/커버리지 + 90·95·99% 도달 시간, `map_eval.png`.

| 버전 (창고, 출발 (0,0), 1회) | 완료 | 99% | 90% | 장애물 재현율 | 장애물→빈칸 |
|---|---|---|---|---|---|
| 직선 추종 | 140 s / 86 m | 132 s | 69 s | – | – |
| A* | 138 s / 85 m | 112 s | 92 s | 93.4 % | 2.55 % |
| A* + 지도 튜닝 + 속도 1.0 m/s·1.5 rad/s | 124 s / 88 m | 99 s | 84 s | 99.7 % | 0.77 % |
| + CEM 파라미터 (아래) + 단일 메시 레이캐스트 | 122 s / 105 m | 81 s | 65 s | 99.7 % | 0.79 % |

결과: `logs/exploration/<run>/` — `exploration_log.csv` (시간, 탐사 면적, GT 대비 커버리지, 이동거리, 계획 시간), `map_*.png`, `final.png`.

환경변수: `GO2_LIDAR_VIS=1` 라이다 점 표시(무거움), `GO2_SAVE_SCANS=1` 스캔 저장, `GO2_CAM_EYE=dx,dy,dz` 로봇 기준 카메라 위치,
`GO2_HIDE_CEILING=1` 천장 패널 숨김(시각만, 높은 카메라용).

## 병렬 탐사 + 보상 기반 파라미터 탐색 (CEM)

같은 창고에 Go2를 N마리 복제. **각자 독립적으로 탐사**(자기 점유 지도, 자기 ARiADNE 워커 프로세스, 자기 A*; 서로 충돌·감지 안 함).
세대마다 무작위 빈 위치에서 출발 → 완료/제한시간/넘어짐까지 탐사 → 점수 = 시간 대비 커버리지 곡선 면적(빨리, 많이 딸수록 높음),
지도 품질(정밀도/재현율/오판)도 정답 지도로 채점.

```bash
# GUI, 16마리, CEM 8세대 (상위 25%가 다음 세대 분포를 결정)
GO2_PAR_MODE=search GO2_PAR_GENS=8 GO2_PAR_EPISODE_S=180 \
  python scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16 --viz kit
# 평가 모드: 모두 같은 파라미터, 출발점만 무작위 (통계용)
GO2_PAR_MODE=eval GO2_PAR_GENS=4 python scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16
```
탐색 파라미터(`exploration/parallel_command.py` `PARAM_SPACE`): 목표 전환 각도, 확정 거리, 추종 거리, 벽 여유, 로봇 반경,
제자리 회전 임계각(≤40°), 정체 판정 시간, ARiADNE `THR_NEXT_WAYPOINT`.
결과: `logs/parallel/<run>/generations.csv` (세대×로봇별 파라미터·점수·품질), `best.json`.

- `GO2_PAR_MODE=loop` (`./run.sh loop`): 무한 반복. 로봇마다 탐사가 끝나면(완료/180 s/넘어짐/충돌) 지도를 3초 보여준 뒤 그 로봇만 랜덤 빈 위치로 리스폰, 새 지도·새 ARiADNE로 재시작. 회차별 기록 `episodes.csv`, 16회차마다 통계 출력.
- `GO2_PAR_STARTS=4`: 후보마다 같은 출발점 4곳에서 평가 (출발점 운을 상쇄). 32마리 = 후보 8 × 출발점 4.
- `GO2_PAR_MODE=compare GO2_PAR_PARAMS=<best.json>`: 절반 기본값 / 절반 찾은 값, 같은 출발점 쌍으로 비교.
- `GO2_FAST_PHYSICS=1`(기본): 물리는 평면, 창고는 외형+라이다 전용, 충돌은 GT 지도(0.25 m)로 판정.
- 라이다는 창고를 한 메시로 구운 `assets/warehouse/warehouse_raycast.usdc` + 단일 메시 RayCaster
  (MultiMeshRayCaster 대비 약 500배 빠름; 32마리가 실시간 약 1.2배로 돔).

2026-10-01 결과 (32마리, 후보 8 × 출발점 4, 10세대 → `logs/parallel/cem_best_2026-10-01.json`),
다른 시드의 48개 출발점 쌍에서 기존 값 대비: 완료 102 s → 95 s (쌍별 −7.2 ± 3.6 s), 완료율 96 % → 100 %,
지도 재현율 0.975 → 0.988. 단, 이 창고 하나에서 찾은 값이라 다른 환경으로의 일반화는 아직 미검증.
GUI: 천장 숨김(시각만), 오른쪽 "Parallel Exploration" 탭에 지도 타일과 리더보드, 상단 카메라 버튼 — Overview(창고 남쪽 18 m 높이 비스듬히) / Follow leader(현재 1등 로봇 뒤 2.5 m) / ◀ ▶(따라갈 로봇 변경, 노란 테두리). 시작 카메라는 `GO2_CAM=overview|leader|<로봇 번호>`.

## 발표 영상 녹화

`--video`의 클립 길이가 `--video_interval`(기본 2000 스텝)보다 길면 2000 프레임마다 잘리고 끝나지 않으므로 항상
`--video_interval 100000`을 같이 준다. 1 스텝 = 0.02 s(50 fps).
```bash
# 1마리 탐사, 위에서 (90 s)
GO2_HIDE_CEILING=1 GO2_CAM_EYE=-4,-4,12 python scripts/rl.py play --task Go2-Warehouse-Explore-Play --num_envs 1 \
  --video --video_length 4500 --video_interval 100000
# 16마리 loop + 로봇별 지도 타일 1초마다 저장
GO2_PAR_MODE=loop GO2_PAR_PARAMS=logs/parallel/cem_best_2026-10-01.json GO2_SAVE_TILES=/tmp/tiles \
  python scripts/rl.py play --task Go2-Warehouse-Parallel-Play --num_envs 16 --video --video_length 9000 --video_interval 100000
# 화면 + 지도 나란히, 4배속
python tools/compose_video.py <clip.mp4> logs/exploration/<run> out.mp4 --prefix map_ --speed 4 --title "..."
python tools/compose_video.py <clip.mp4> /tmp/tiles out.mp4 --prefix tiles_ --speed 4
```

## VLP-16 마운트 기울기 (`GO2_LIDAR_TILT=<deg>`, 기본 0)

`tools/lidar_mount_study.py logs/lidar_scans/<run>`: 녹화한 탐사 경로를 그대로 두고 마운트 각도만 바꿔 GPU ray-cast →
3D 표면 커버리지(높이별) + 앞쪽 저상 장애물 검출 + 상한(창고 어디서든 볼 수 있는 표면) → `logs/mount_study/mount_study.json`,
`tools/make_figures.py` → `figures/fig5_lidar_mount_tilt.png`.

| 기울기 | 같은 경로 3D | 천장 | 상한 | 앞쪽 <0.45 m 검출 |
|---|---|---|---|---|
| 0° | 76.9 % | 50 % | 90.2 % | 100 % |
| 10° | 87.5 % | 84 % | 96.4 % | 101 % |
| **15°** | **89.5 %** | **91 %** | 96.7 % | 98 % |
| 20° | 90.1 % | 94 % | 96.7 % | 54 % |
| 30° | 90.3 % | 96 % | 96.7 % | 9 % |

실제 시뮬 탐사(15°): 2D 완료 71 s / 57.5 m (0°와 같음), 그 시점 3D 81.7 % (0°는 같은 시점 70.5 %).
남은 공백은 ARiADNE가 2D 기준으로 탐사를 끝내기 때문 → 3D 커버리지 보상·종료 조건의 대상.

## 라이다 노이즈 모델 (`exploration/sensor_noise.py`)

`GO2_LIDAR_NOISE=1`: 거리 노이즈 σ 3 cm, 0.9 m 이내 제거, 2 % 누락, 회전 스캔(0.1 s/회전) 모션 왜곡.
`=2`: + 스캔별 위치 오차(σ 5 cm, 0.5°). `GO2_DESKEW=1`: IMU급 속도 추정(σ 0.05)으로 왜곡 보정. `GO2_SKIP_TURN=<rad/s>`: 빠른 회전 중 스캔 무시.
A/B: `GO2_PAR_MODE=compare GO2_COMPARE=noise` (같은 파라미터·출발점, 절반만 노이즈).

| 48쌍 (CEM 파라미터) | 정밀도 | 재현율 | 점수 차이 |
|---|---|---|---|
| 깨끗한 레이캐스트 | 100 % | 98.6–98.9 % | 기준 |
| VLP-16 노이즈, 보정 없음 | 96.1 % | 95.7 % | −0.023 ± 0.013 (유령 장애물로 조기 종료) |
| VLP-16 노이즈 + deskew | 99.99 % | 99.2 % | −0.004 ± 0.003 |

→ 문제는 노이즈가 아니라 제자리 회전(최대 1.5 rad/s) 중 스캔 왜곡. 실로봇은 IMU 기반 deskew(LIO 계열 SLAM) 필수.
ARiADNE 원본 버그: 그래프 원점 노드가 장애물로 바뀌면 `remove_unconnected_nodes`에서 크래시 → 래퍼에서 원점 재설정 + 워커 예외 시 재생성.

## RTX VLP-16 (물리 기반 라이다) — 이 PC에서는 동작 안 함

- `tools/make_vlp16_rtx.py` → `assets/vlp16/rtx/VLP16_rtx.usda`: VLP-16 매뉴얼 사양의 OmniLidar 프로파일
  (16채널, 발사 순서별 고도각, 2.304 µs 간격, 18,080 Hz → 0.199°, 0.9–100 m, ±3 cm, CW, 단일 리턴).
  Isaac Sim 6.0.1 내장 프로파일에는 VLP-16이 없음. 회전당 발사 수가 정수가 아니면 `LidarCore::updateParams() -- invalid parameters`.
- `tools/rtx_vlp16_check.py`: 순정 Isaac Sim용 검증 스크립트 (큐브/창고, 이상적 레이캐스트와 거리 비교).
- **2026-10-01 확인: 이 PC(RTX 5090, 드라이버 595.91, Ubuntu 26.04)에서는 RTX 라이다가 데이터를 전혀 내지 않음.**
  내 프로파일·NVIDIA 예제 프로파일·기본값 모두, Isaac Lab/순정 SimulationApp/전체 앱, GUI/헤드리스, 공식 컨테이너
  `nvcr.io/nvidia/isaac-sim:6.0.1`(지원 OS)에서 **NVIDIA 공식 예제 `inspect_lidar_gmo.py`를 300프레임** 돌려도 0점.
  컨테이너도 같은 드라이버를 쓰므로 드라이버(595)/Blackwell 조합 문제로 추정. 다른 PC(예: 3090 + 드라이버 580 계열)에서:
  ```bash
  docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -v ~/Desktop/3DOG/sim:/sim \
      --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 /sim/tools/rtx_vlp16_check.py --scene warehouse
  ```
- 그 전까지는 레이캐스트 + VLP-16 노이즈 모델(위)이 사실성 보강 수단.

## VLP-16 장착

- 마운트 위치: [anujjain-dev/unitree-go2-ros2](https://github.com/anujjain-dev/unitree-go2-ros2) `go2_description/xacro/velodyne.xacro`
  (base_link → velodyne_base_link xyz 0.2 0 0.08, 스캔 원점 +0.0377 m)
- 메시: [ToyotaResearchInstitute/velodyne_simulator](https://github.com/ToyotaResearchInstitute/velodyne_simulator) (DAE는 XML 손상 → STL 사용)
- 탑재 질량 0.98 kg(센서 0.83 + 브래킷 0.15)과 CoM 이동은 이벤트로 고정 적용 (학습 시 질량·CoM 무작위화 범위 안)
- ray caster의 `data.pos_w`는 부착 링크(base) 위치 → 실제 스캔 원점은 `pos_w + R·offset` (lidar_io.py에서 처리)
- 레이는 창고 메시에만 닿음 (Go2 자기 몸체 가림은 모사 안 함)

## 알려진 문제

- Ubuntu 26.04에 `libxml2.so.2`가 없어 `asset_converter` 확장이 로드 실패 → 로그에 Traceback이 찍히지만 시뮬레이션과는 무관.
- 자체 URDF 변환본은 URDF importer 3.0이 강체를 중첩 구조로 만들어 학습 시 관절 가속도가 비정상적으로 큼.
