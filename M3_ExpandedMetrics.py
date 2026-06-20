# M3_ExpandedRules.py
# ──────────────────────────────────────────────────────────────────
# 목표: 특징 설계·확장
#
# 배경:
#   M1의 지표 3가지로 부족할 것 같아, 근거 있는 특징을 더 만들어 두고 유효한 지표들을 선택
#
# 코드:
#   1) 동작 구분은 M2.Segmentation에서 최종 선택한 HMM을 사용.
#   2) 정밀 move 경계 대신 '정지(IDLE)/이동(MOVING) 구간 통계'로 집계.
#   3) 상태 의존적 지표는 IDLE/MOVING 구분 (예: 직선팔은 'IDLE 상태에서 중요).
#   4) ★루트에 따라 달라지는 양(동작 수·휴식 비율·총 상승량·동작당 상승량 등)은 제외
#      어느 루트에도 공통 적용되는, 스케일·루트에 비교적 견고한 '비율·각도' 기반 특징만 남긴다. (특징 수가 적어도 신뢰도를 우선시)
#
# 특징 9개:
#   직선팔 2 (쉴 때)
#   균형 2 (삼지점 묶음)
#   벽거리 3
#   부드러움 1
#   다리추진 1 (무릎 각도 기반)
#
# 산출물:
#   - compute_features_from_landmarks(): 랜드마크 → 특징 dict (MediaPipe 불필요)
#   - extract_features(): 영상 1개 → 특징 dict
#   - build_dataset(): 폴더(전문가/비전문가) 순회 → features.csv
#
# 주의: 벽거리 계열은 MediaPipe z 기반이라 측정 신뢰도가 낮을 수 있음.
#
# 데이터 구조 (all_landmarks): 프레임들의 리스트. 각 원소는
#   {'frame_idx': 원본 프레임 번호, 'time_sec': 시간(초),
#    'landmarks': [관절 33개]} 이고, 각 관절은 {'x','y','z','visibility'}
#    (x,y는 0~1 정규화).
# ──────────────────────────────────────────────────────────────────

import os
import glob
import csv
import numpy as np

# M1 재사용 (지표 + 추출/전처리 + 상수)
from M1_Rules import (
    calc_angle, check_tripod, fit_wall_plane, calc_wall_distance,
    extract_landmarks, interpolate_low_visibility, compute_velocity,
    MODEL_PATH, TARGET_FPS, VELOCITY_WINDOW,
    MIN_IDLE_DURATION_SEC, MIN_MOVE_DURATION_SEC,
)
# M2 재사용 (최종선별된 HMM)
from M2_Segmentation import segment_hmm

# MediaPipe 33점 인덱스
L_SH, R_SH = 11, 12   # 어깨
L_EL, R_EL = 13, 14   # 팔꿈치
L_WR, R_WR = 15, 16   # 손목
L_HIP, R_HIP = 23, 24 # 골반
L_KN, R_KN = 25, 26   # 무릎
L_AN, R_AN = 27, 28   # 발목

# ──────────────────────────────────────────────
# 특징 문서 (이름 → 설명 / 클라이밍 근거 / 방향)
#   ↑: 클수록 좋은 자세 , ↓: 작을수록 좋은 자세, 중립: 데이터로 판단
# ──────────────────────────────────────────────
FEATURE_DOC = {
    # 직선팔 (쉴 때)
    "elbow_mean_idle":     ("정지 구간 평균 팔꿈치 각도(양팔)", "가만히 버티는 상황에서 팔을 펴면 전완근 부담이 줄어들음.", "↑"),
    "straight_ratio_idle": ("정지 구간 직선팔(≥150°) 비율", "위와 같은 이유로 직선팔 유지 능력 클수록 좋음", "↑"),
    # 균형 (삼지점 묶음)
    "tripod_ratio":        ("삼지점 비율(손목이 양발 x 사이)", "지지 기반 위에 손을 두는 비율","중립"),
    "com_over_base_ratio": ("무게중심(골반) 양발 사이 비율", "균형: COM(Center of Mass)이 발 지지면 위에 있을수록 안정", "↑"),
    # 벽거리 ()
    "wall_dist_idle":      ("정지 구간 평균 벽거리", "평소엔 벽에 붙어 무게를 벽으로 밀어야 부담이 줄어들음 ", "↓"),
    "wall_dist_move":      ("이동 구간 평균 벽거리", "이동 시 거리 변화 비교용", "중립"),
    "wall_push_ratio":     ("이동/정지 벽거리 비", "이동 직전 벽에서 멀어지는 경향(>1)", "중립"),
    # 부드러움
    "jerk_move":           ("이동 구간 저크(속도 2차변화) 평균", "작을수록 부드러운 동작, 전문가는 부드러움", "↓"),
    # 다리 추진
    "leg_drive_ratio":     ("상승 중 다리펴기 vs 팔당기기 비(0~1)", "다리로 밀어 오르면 1쪽, 팔로 당기면 0쪽 — '팔 말고 다리 힘을 많이 써야 함'", "↑"),
}


# ──────────────────────────────────────────────
# 보조 함수
# ──────────────────────────────────────────────
def _safe(arr, fn, default=np.nan): # 에러 회피
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return default
    return float(fn(arr))

def _hip_cy(lm): #골반 중심 y좌표
    return (lm[L_HIP]['y'] + lm[R_HIP]['y']) / 2.0

def _com_in_base(lm): # 골반 중심 x좌표가 양발 x 사이에 있는지
    hx = (lm[L_HIP]['x'] + lm[R_HIP]['x']) / 2.0
    lo = min(lm[L_AN]['x'], lm[R_AN]['x'])
    hi = max(lm[L_AN]['x'], lm[R_AN]['x'])
    return lo <= hx <= hi

def _knee_mean(lm): # 양쪽 무릎 각도 평균
    # 무릎 각도(골반-무릎-발목). 다리가 펴질수록 180에 가까움
    kl = calc_angle([lm[L_HIP]['x'], lm[L_HIP]['y']],
                    [lm[L_KN]['x'], lm[L_KN]['y']],
                    [lm[L_AN]['x'], lm[L_AN]['y']])
    kr = calc_angle([lm[R_HIP]['x'], lm[R_HIP]['y']],
                    [lm[R_KN]['x'], lm[R_KN]['y']],
                    [lm[R_AN]['x'], lm[R_AN]['y']])
    return (kl + kr) / 2.0


# ──────────────────────────────────────────────
# 핵심: 랜드마크 → 특징 벡터 (MediaPipe 불필요 → 단독 테스트 가능)
# ──────────────────────────────────────────────
def compute_features_from_landmarks(all_landmarks, fps, frame_skip,
                                    velocity_window=VELOCITY_WINDOW,
                                    min_still_sec=MIN_IDLE_DURATION_SEC,
                                    min_move_sec=MIN_MOVE_DURATION_SEC):
    n = len(all_landmarks)
    velocities = compute_velocity(all_landmarks)

    # 1단계에서 고른 HMM 으로 정지/이동 분류
    #   반환: moves(이동 구간 목록), smoothed(평활 속도), merged(구간), is_still(프레임별 정지여부 bool)
    moves, smoothed, merged, is_still = segment_hmm(
        all_landmarks, velocities, fps, frame_skip,
        velocity_window, min_still_sec, min_move_sec)
    is_still = np.asarray(is_still, dtype=bool)
    move = ~is_still
    idle = is_still

    # 프레임별 양 계산
    elbow_mean = np.zeros(n)
    knee_mean = np.zeros(n)
    tripod = np.zeros(n, dtype=bool)
    com = np.zeros(n, dtype=bool)
    hip_cy = np.zeros(n)
    wall = np.zeros(n)

    wall_plane = None
    for i, fr in enumerate(all_landmarks):
        lm = fr['landmarks']
        la = calc_angle([lm[L_SH]['x'], lm[L_SH]['y']],
                        [lm[L_EL]['x'], lm[L_EL]['y']],
                        [lm[L_WR]['x'], lm[L_WR]['y']])
        ra = calc_angle([lm[R_SH]['x'], lm[R_SH]['y']],
                        [lm[R_EL]['x'], lm[R_EL]['y']],
                        [lm[R_WR]['x'], lm[R_WR]['y']])
        elbow_mean[i] = (la + ra) / 2.0
        knee_mean[i] = _knee_mean(lm)
        tripod[i] = check_tripod(lm)
        com[i] = _com_in_base(lm)
        hip_cy[i] = _hip_cy(lm)
        # 벽 평면: 정지 프레임에서 갱신, 이동 중 유지 (M1 디버그 영상과 동일 원리)
        if is_still[i] or wall_plane is None:
            wall_plane = fit_wall_plane(lm)
        wall[i] = calc_wall_distance(lm, wall_plane) # 골반-벽 거리

    # 직선팔 여부: 팔꿈치 평균 각도가 150° 이상이면 1, 아니면 0
    straight = (elbow_mean >= 150).astype(float)
    # 저크(jerk) = 속도에 미분을 두 번 하여 계산한 가속도의 변화율.
    jerk = np.abs(np.gradient(np.gradient(smoothed)))

    # 다리 추진: 상승 중(이동 + 골반 상승)에 무릎 펴짐 vs 팔꿈치 굽힘
    dknee = np.diff(knee_mean, prepend=knee_mean[0])
    delbow = np.diff(elbow_mean, prepend=elbow_mean[0])
    dhipy = np.diff(hip_cy, prepend=hip_cy[0])
    rising = move & (dhipy < 0)                           # 이동 중 + 골반이 위로(y 감소)
    if rising.any():
        leg_push = float(np.maximum(0,  dknee[rising]).sum())   # 무릎 펴짐 = 밀기
        arm_pull = float(np.maximum(0, -delbow[rising]).sum())  # 팔꿈치 굽힘 = 당기기
        leg_drive_ratio = leg_push / (leg_push + arm_pull + 1e-8)
    else:
        leg_drive_ratio = np.nan

    wi = _safe(wall[idle], np.mean)
    wm = _safe(wall[move], np.mean)

    feats = {
        # 직선팔
        "elbow_mean_idle":     _safe(elbow_mean[idle], np.mean),
        "straight_ratio_idle": _safe(straight[idle], np.mean),
        # 균형
        "tripod_ratio":        _safe(tripod.astype(float), np.mean),
        "com_over_base_ratio": _safe(com.astype(float), np.mean),
        # 벽거리
        "wall_dist_idle":      wi,
        "wall_dist_move":      wm,
        "wall_push_ratio":     (wm / (wi + 1e-8)) if (wi == wi and wm == wm) else np.nan,
        # 부드러움
        "jerk_move":           _safe(jerk[move], np.mean),
        # 다리 추진
        "leg_drive_ratio":     leg_drive_ratio,
        # 메타(특징 아님, 진단용)
        "_n_frames":           n,
        "_n_moves":            len(moves),
        "_idle_frames":        int(idle.sum()),
    }
    return feats


# ──────────────────────────────────────────────
# 영상 1개 → 특징
# ──────────────────────────────────────────────
def extract_features(video_path, min_frames=10):
    all_landmarks, fps, frame_skip = extract_landmarks(video_path, MODEL_PATH, TARGET_FPS)
    all_landmarks = interpolate_low_visibility(all_landmarks)
    if len(all_landmarks) < min_frames:
        print(f"  포즈 감지 부족({len(all_landmarks)}) → 건너뜀: {video_path}")
        return None
    return compute_features_from_landmarks(all_landmarks, fps, frame_skip)


# ──────────────────────────────────────────────
# 폴더(전문가/비전문가) 순회 → features.csv
# ──────────────────────────────────────────────
def build_dataset(folder_label_map, out_csv="features.csv"):
    feat_keys = list(FEATURE_DOC)                  # 특징 순서 고정(문서 순)
    meta_keys = ["_n_frames", "_n_moves", "_idle_frames"]
    rows = []
    for folder, label in folder_label_map.items():
        vids = sorted(glob.glob(os.path.join(folder, "*.mp4")))
        if not vids:
            print(f"[경고] '{folder}' 에 mp4 없음")
            continue
        for vp in vids:
            name = os.path.basename(vp)
            print(f"[{label}] {name}")
            f = extract_features(vp)
            if f is None:
                continue
            row = {"video": name, "label": label}
            row.update({k: f.get(k) for k in feat_keys + meta_keys})
            rows.append(row)

    cols = ["video", "label"] + feat_keys + meta_keys
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n특징 CSV 저장: {out_csv}  ({len(rows)} 영상 × {len(feat_keys)} 특징)")
    return rows


def save_feature_doc(out_md="feature_doc.md"):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 특징 정의 문서 (M3_ExpandedRules)\n\n")
        f.write("| 특징 | 설명 | 클라이밍 근거 | 방향 |\n|---|---|---|---|\n")
        for k, (desc, why, direction) in FEATURE_DOC.items():
            f.write(f"| `{k}` | {desc} | {why} | {direction} |\n")
    print(f"특징 문서 저장: {out_md}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 폴더 → 레이블 (영상 채운 뒤 실행)
    FOLDERS = {
        "ExpertVideoData":     "expert",      # E00.mp4 ~ E99.mp4
        "NonExpertVideoData": "non_expert",  # N00.mp4 ~ N70.mp4
    }
    save_feature_doc("feature_doc.md")
    build_dataset(FOLDERS, "features.csv")