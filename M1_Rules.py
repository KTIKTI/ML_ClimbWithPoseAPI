# M1_Rules.py
# ──────────────────────────────────────────────────────────────────
# 목표: 규칙 기반 지표 3종 + 정지/이동 구분 + 디버깅
#
# 배경:
#   M0_MediaPipe.py 에서 익힌 MediaPipe를 활용해서 영상 분석
#   - 지표 3가지, 정지 이동 상태 구분을 모두 규칙으로 구현 및 점검
#
# 코드:
#   climbvideodata/ 의 영상들(T*.mp4)을 순회하며 영상마다 다음 절차를 수행
#   1) 포즈 추출 (extract_landmarks)
#   2) 규칙 기반 3지표 계산: 팔꿈치 각도(직선팔), 삼지점, 벽과의 거리
#   3) 속도 기반 정지/이동 구분으로 '동작(move)' 단위 분리 (segment_moves)
#   4) 동작별 지표 집계 + 디버그 영상 + JSON(analysis_result.json) 저장
#   (지표 정의는 '지표 공식.docx' 참고)
#
# 비고:
#   - 정규화 좌표 사용 → 해상도(720p/1080p)에 무관하게 동일 처리.
#   - frame_skip = round(fps / TARGET_FPS) 으로 모든 영상을 의 fps를 통일. 현재 30fps
#   - 동작 구분(규칙):
#     손목·발목 4점의 프레임 간 이동량(속도) 계산
#     이동평균 계산
#     '속도 중앙값 × IDLE_THRESHOLD_RATIO' 이하를 정지로 설정
#     짧은 구간 병합
#   - 벽 거리:
#     손목2 + 발목2로 벽 평면 ax+by+cz+d=0 을 SVD로 피팅(정지 구간에서 갱신),
#     골반 중점에서 평면까지의 거리. (MediaPipe z는 신뢰도가 낮아 3지표 중 가장 불안정할 수 있음)
#
# 파이프라인 구성 (아래 섹션 순서):
#   1.설정  2.지표 계산 함수  3.랜드마크 추출  4.동작 구분
#   5.동작별 지표 계산  6.디버그 영상 생성  7.결과 출력/저장
#   메인(영상 순회)
#
# 한계 / 다음 단계:
#   - 상대 임계를 활용해서 영상별로 잘 작동하지만, 영상에 따라 동작이 수십 초로 뭉치거나 0.1초로 파편화되는 문제 발견
#     K-Means/GMM/HMM 비지도 방법과 비교해 대안을 찾기.
#   - 디버그 영상 생성 코드는 검증용
#
# ──────────────────────────────────────────────────────────────────

import os
#메시지 제거
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
os.environ['GLOG_logtostderr'] = '0'
import glob
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

# ──────────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────────
DATA_DIR = "DebugSingleVideo"  # 분석 영상 경로
VIDEO_PATTERN = os.path.join(DATA_DIR, "T*.mp4") #T00.mp4 ~ T99.mp4 패턴
# 모델 파일 경로, 둘 중 하나 선택
# full: 더 빠른 속도
# heavy: 더 높은 정확성
#MODEL_PATH = "pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_heavy.task"

TARGET_FPS = 30 #분석하려는 FPS
VELOCITY_WINDOW = 5  # 노이즈 제거용 속도 이동평균 프레임 수 (30FPS 기준 약 0.17초)
IDLE_THRESHOLD_RATIO = 0.3  # 정지 판별: 중앙값 속도 대비 비율 (동작이 너무 파편화되면 값을 낮추고, 너무 합쳐지면 값을 올리면 됨)
MIN_IDLE_DURATION_SEC = 0.3  # 최소 정지 구간 길이 (초)
MIN_MOVE_DURATION_SEC = 0.3  # 최소 동작 구간 길이 (초)


# ──────────────────────────────────────────────
# 2. 지표 계산 함수
# ──────────────────────────────────────────────
def calc_angle(a, b, c): #각도 계산 함수. 지표 공식.docx 참고
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8) #1e-8은 분모가 0 되는 것을 방지
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def check_tripod(landmarks): #삼지점 여부 판별 함수. 지표 공식.docx 참고
    lw = landmarks[15]['x'] # 왼쪽 손목 x 좌표
    rw = landmarks[16]['x'] # 오른쪽 손목 x 좌표
    la = landmarks[27]['x'] # 왼쪽 발목 x 좌표
    ra = landmarks[28]['x'] # 오른쪽 발목 x 좌표

    ankle_left = min(la, ra)
    ankle_right = max(la, ra)
    lw_in = ankle_left <= lw <= ankle_right
    rw_in = ankle_left <= rw <= ankle_right
    return lw_in and rw_in

def fit_wall_plane(lm): #SVD로 벽 평면 방정식 제작
    pts = np.array([
        [lm[15]['x'], lm[15]['y'], lm[15]['z']],  # 왼쪽 손목
        [lm[16]['x'], lm[16]['y'], lm[16]['z']],  # 오른쪽 손목
        [lm[27]['x'], lm[27]['y'], lm[27]['z']],  # 왼쪽 발목
        [lm[28]['x'], lm[28]['y'], lm[28]['z']],  # 오른쪽 발목
    ])

    centroid = pts.mean(axis=0)
    svd = np.linalg.svd(pts - centroid)
    normal = svd[2][-1]  # 법선
    a, b, c = normal
    d = -np.dot(normal, centroid)
    return a, b, c, d

def calc_wall_distance(lm, plane): #골반 중심과 벽 사이 거리 계산 함수.
    a, b, c, d = plane
    hip_x = (lm[23]['x'] + lm[24]['x']) / 2
    hip_y = (lm[23]['y'] + lm[24]['y']) / 2
    hip_z = (lm[23]['z'] + lm[24]['z']) / 2
    return abs(a * hip_x + b * hip_y + c * hip_z + d) / (np.sqrt(a**2 + b**2 + c**2) + 1e-8)

# ──────────────────────────────────────────────
# 3. 랜드마크
# ──────────────────────────────────────────────
def _create_landmarker(model_path, use_gpu=True):
    def build(delegate):
        base = python.BaseOptions(model_asset_path=model_path, delegate=delegate)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(opts)

    if use_gpu:
        try:
            lm = build(python.BaseOptions.Delegate.GPU)
            print("[추론 장치] GPU delegate 사용")
            return lm
        except Exception as e:
            print(f"[추론 장치] GPU 사용 불가 → CPU 폴백 ({e})")
            return build(python.BaseOptions.Delegate.CPU)

    print("[추론 장치] CPU 사용 (use_gpu=False)")
    return build(python.BaseOptions.Delegate.CPU)


def extract_landmarks(video_path, model_path, frame_skip):

    landmarker = _create_landmarker(model_path, use_gpu=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #설정한 FPS로 설정
    frame_skip = max(1, round(fps / TARGET_FPS))
    effective_fps = fps / frame_skip

    print(f"영상 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "f"{fps:.1f}fps → skip {frame_skip} → 실질 {effective_fps:.1f}fps")

    all_landmarks = []
    frame_idx = 0
    analyzed_idx = 0
    last_timestamp_ms = -1

    while cap.isOpened(): # 영상을 읽어 올 때
        ret, frame = cap.read()
        if not ret: #프레임을 못 읽는다면 프로그램 종료
            break

        if frame_idx % frame_skip == 0: #frame_skip 간격마다 분석
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe는 RGB 컬러 형식을 사용하기 때문에 형식 변환 필요
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_idx * 1000 / fps)
            # 타임스탬프 단조 증가 보장
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms) #Pose 감지

            if result.pose_landmarks:
                lm_list = list(result.pose_landmarks[0]) #첫 사람의 관절 33개
                landmarks = []
                for lm in lm_list:
                    landmarks.append({
                        'x': lm.x, 'y': lm.y, 'z': lm.z, #x,y,z 좌표
                        'visibility': lm.visibility #가시성
                    })
                all_landmarks.append({
                    'frame_idx': frame_idx,
                    'time_sec': frame_idx / fps,
                    'landmarks': landmarks
                })
            analyzed_idx += 1

            # 진행률 표시
            if frame_idx > 0 and frame_idx % (100 * frame_skip) == 0:
                pct = frame_idx / total_frames * 100
                print(f"  진행: {pct:.1f}% ({frame_idx}/{total_frames})")

        frame_idx += 1

    # 안전하게 종료
    cap.release()
    landmarker.close()

    print(f"완료: 총 {analyzed_idx}프레임 분석, {len(all_landmarks)}프레임에서 포즈 감지")
    return all_landmarks, fps, frame_skip

def interpolate_low_visibility(all_landmarks, min_vis=0.5): #visibility가 낮은 랜드마크 보간
    for joint_idx in range(33):
        xs, ys, vis = [], [], []
        for frame in all_landmarks:
            lm = frame['landmarks'][joint_idx]
            xs.append(lm['x'])
            ys.append(lm['y'])
            vis.append(lm['visibility'])

        #visibility가 높을 때 건너 뜀
        for i in range(len(vis)):
            if vis[i] >= min_vis:
                continue

            # 앞쪽에서 visibility 높은 프레임
            prev = None
            for j in range(i - 1, -1, -1):
                if vis[j] >= min_vis:
                    prev = j
                    break

            # 뒤쪽에서 visibility 높은 프레임
            nxt = None
            for j in range(i + 1, len(vis)):
                if vis[j] >= min_vis:
                    nxt = j
                    break

            # 양쪽 다 있으면 선형 보간
            if prev is not None and nxt is not None:
                ratio = (i - prev) / (nxt - prev)
                all_landmarks[i]['landmarks'][joint_idx]['x'] = xs[prev] + ratio * (xs[nxt] - xs[prev])
                all_landmarks[i]['landmarks'][joint_idx]['y'] = ys[prev] + ratio * (ys[nxt] - ys[prev])
            # 앞쪽만 있으면 그 값 사용
            elif prev is not None:
                all_landmarks[i]['landmarks'][joint_idx]['x'] = xs[prev]
                all_landmarks[i]['landmarks'][joint_idx]['y'] = ys[prev]
            # 뒤쪽만 있으면 그 값 사용
            elif nxt is not None:
                all_landmarks[i]['landmarks'][joint_idx]['x'] = xs[nxt]
                all_landmarks[i]['landmarks'][joint_idx]['y'] = ys[nxt]

    return all_landmarks

# ──────────────────────────────────────────────
# 4. 동작 구분
# ──────────────────────────────────────────────
def compute_velocity(all_landmarks):
    #  - 팬: 손목·발목을 '골반 중심 기준 상대좌표'로 → 화면 평행이동 상쇄
    #  - 줌: 어깨너비(몸통 스케일)로 나눠 → 화면 확대/축소 상쇄
    #  4점 합이 아니라 '중앙값' → 한 사지만 빠르면 중앙값은 낮음 = 정지
    KEY = [15, 16, 27, 28]  # 양 손목, 양 발목

    def body_frame(lm):
        hx = (lm[23]['x'] + lm[24]['x']) / 2.0
        hy = (lm[23]['y'] + lm[24]['y']) / 2.0
        sw = np.hypot(lm[11]['x'] - lm[12]['x'], lm[11]['y'] - lm[12]['y'])  # 어깨너비
        return hx, hy, max(sw, 1e-3)  # scale 0 방지

    vels = [0.0]
    for i in range(1, len(all_landmarks)):
        cur, prev = all_landmarks[i]['landmarks'], all_landmarks[i - 1]['landmarks']
        chx, chy, cs = body_frame(cur)
        phx, phy, ps = body_frame(prev)
        speeds = []
        for idx in KEY:
            c = np.array([(cur[idx]['x'] - chx) / cs, (cur[idx]['y'] - chy) / cs])
            p = np.array([(prev[idx]['x'] - phx) / ps, (prev[idx]['y'] - phy) / ps])
            speeds.append(float(np.linalg.norm(c - p)))
        vels.append(float(np.median(speeds)))   # 중앙값 = 단일 사지 움직임 무시
    return np.array(vels)

def moving_average(data, window): #이동평균 data: 1차원 배열, window: 평균 프레임 수
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')

def segment_moves(all_landmarks, velocities, fps, frame_skip, velocity_window, idle_threshold_ratio, min_still_sec, min_move_sec):
    #속도 기반으로 정지/이동 동작 구분

    effective_fps = fps / frame_skip #실제로 분석할 fps

    # 이동평균으로 노이즈 제거
    smoothed = moving_average(velocities, velocity_window)

    # 임계값: 전체 속도 중앙값 × 비율
    median_vel = np.median(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 0.01
    threshold = median_vel * idle_threshold_ratio
    print(f"속도 중앙값: {median_vel:.6f}, 정지 임계값: {threshold:.6f}")

    is_still = smoothed < threshold  # 프레임별 정지/이동 분류, True = 정지

    # 최소 구간 길이 프레임수 변환
    min_still_frames = int(min_still_sec * effective_fps)
    min_move_frames = int(min_move_sec * effective_fps)

    #연속된 같은 상태를 하나의 구간으로 묶음
    segments = []
    current_state = is_still[0]
    start = 0
    for i in range(1, len(is_still)):
        if is_still[i] != current_state:
            segments.append((start, i - 1, current_state))
            current_state = is_still[i]
            start = i
    segments.append((start, len(is_still) - 1, current_state))

    # 너무 짧은 구간 처리 (짧은 정지 → 이동, 짧은 이동 → 정지)
    filtered = []
    for start, end, state in segments:
        duration = end - start + 1
        if state and duration < min_still_frames:
            filtered.append((start, end, False)) # 짧은 정지 → 이동
        elif not state and duration < min_move_frames:
            filtered.append((start, end, True)) # 너무 짧은 → 정지
        else:
            filtered.append((start, end, state))

    # 인접한 같은 상태 합치기
    merged = [filtered[0]]
    for i in range(1, len(filtered)):
        if filtered[i][2] == merged[-1][2]:
            merged[-1] = (merged[-1][0], filtered[i][1], merged[-1][2])
        else:
            merged.append(filtered[i])

    # 동작(move) 추출: 이동 구간만
    moves = []
    for start, end, state in merged:
        if not state:  # 이동 구간, False일때 작동
            t_start = all_landmarks[start]['time_sec']
            t_end = all_landmarks[min(end, len(all_landmarks) - 1)]['time_sec']
            moves.append({
                'move_idx': len(moves) + 1,
                'start_frame_idx': start,
                'end_frame_idx': end,
                'time_start': round(t_start, 2),
                'time_end': round(t_end, 2),
                'duration': round(t_end - t_start, 2),
            })

    print(f"구간 분류: 총 {len(merged)}개 구간, 동작 {len(moves)}개 감지")
    return moves, smoothed, merged

# ──────────────────────────────────────────────
# 5. 동작별 지표 계산
# ──────────────────────────────────────────────
def analyze_moves(moves, all_landmarks):

    for move in moves:
        s, e = move['start_frame_idx'], move['end_frame_idx']
        e = min(e, len(all_landmarks) - 1) #범위 초과 방지

        l_angles, r_angles, tripods = [], [], []
        wall_dists = []

        # 직전 프레임으로 벽 평면 피팅 (정지 상태에서의 벽 위치 기준)
        plane_frame = max(0, s - 1)
        plane = fit_wall_plane(all_landmarks[plane_frame]['landmarks'])

        for i in range(s, e + 1):
            lm = all_landmarks[i]['landmarks']

            la = calc_angle( #왼팔 각도
                [lm[11]['x'], lm[11]['y']], #왼쪽 어깨 x,y좌표
                [lm[13]['x'], lm[13]['y']], #왼쪽 팔꿈치 x,y좌표
                [lm[15]['x'], lm[15]['y']], #왼쪽 손목 x,y좌표
            )
            ra = calc_angle( #오른팔 각도
                [lm[12]['x'], lm[12]['y']], #오른쪽 어깨 x,y좌표
                [lm[14]['x'], lm[14]['y']], #오른쪽 팔꿈치 x,y좌표
                [lm[16]['x'], lm[16]['y']], #오른쪽 손목 x,y좌표
            )
            l_angles.append(la)
            r_angles.append(ra)
            tripods.append(check_tripod(lm)) #삼지점
            wall_dists.append(calc_wall_distance(lm, plane)) #벽과의 거리 계산

        #직선팔 지표
        move['l_elbow_avg'] = round(float(np.mean(l_angles)), 1)
        move['r_elbow_avg'] = round(float(np.mean(r_angles)), 1)
        move['l_elbow_min'] = round(float(np.min(l_angles)), 1)
        move['r_elbow_min'] = round(float(np.min(r_angles)), 1)
        move['l_straight_ratio'] = round(sum(1 for a in l_angles if a >= 150) / len(l_angles) * 100, 1)
        move['r_straight_ratio'] = round(sum(1 for a in r_angles if a >= 150) / len(r_angles) * 100, 1)

        #삼지점 지표
        move['tripod_ratio'] = round(sum(tripods) / len(tripods) * 100, 1)

        move['frame_count'] = e - s + 1

        # 벽 거리 지표
        move['wall_dist_avg'] = round(float(np.mean(wall_dists)), 4)
        move['wall_dist_max'] = round(float(np.max(wall_dists)), 4)

        # Y축 상승량
        start_y = (all_landmarks[s]['landmarks'][23]['y'] + all_landmarks[s]['landmarks'][24]['y']) / 2
        end_y = (all_landmarks[e]['landmarks'][23]['y'] + all_landmarks[e]['landmarks'][24]['y']) / 2
        move['climb_delta_y'] = round(float(start_y - end_y), 4)  # Y는 아래가 큰 값이므로 반전해줘야 함

    return moves

# ──────────────────────────────────────────────
# 6. 디버그 영상 생성 (안 쓸때 주석처리)
# ──────────────────────────────────────────────

# 연결선 정의
CONNECTIONS = [
    (11, 13), (13, 15),  # 왼팔: 어깨 → 팔꿈치 → 손목
    (12, 14), (14, 16),  # 오른팔: 어깨 → 팔꿈치 → 손목
    (11, 12),  # 어깨 - 어깨
    (11, 23), (12, 24),  # 어깨 → 골반 (몸통)
    (23, 24),  # 골반 - 골반
    (23, 25), (25, 27),  # 왼다리: 골반 → 무릎 → 발목
    (24, 26), (26, 28),  # 오른다리: 골반 → 무릎 → 발목
]

def generate_debug_video(video_path, all_landmarks, moves, fps, frame_skip, output_path):

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상 너비
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상 높이
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4 형식으로 출력 영상 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 원본 프레임 번호 → 랜드마크 매핑
    frame_to_lm = {}
    for entry in all_landmarks:
        frame_to_lm[entry['frame_idx']] = entry['landmarks']

    # 원본 프레임 번호 → 해당 동작(move) 매핑
    frame_to_move = {}
    for move in moves:
        # all_landmarks 인덱스를 원본 프레임 번호로 변환
        s = all_landmarks[move['start_frame_idx']]['frame_idx']
        e = all_landmarks[min(move['end_frame_idx'], len(all_landmarks) - 1)]['frame_idx']
        for fi in range(s, e + 1):
            frame_to_move[fi] = move

    frame_idx = 0
    wall_plane = None  # 현재 벽 평면 (IDLE일 때 갱신)

    while cap.isOpened(): # 영상을 읽어 올 때
        ret, frame = cap.read()
        if not ret: #프레임을 못 읽는다면 프로그램 종료
            break

        # 현재 프레임에 대응하는 랜드마크 조회
        lm = frame_to_lm.get(frame_idx)

        if lm:
            # ── 관절 점 + 뼈대 선 그리기 ──
            points = {}
            for idx, l in enumerate(lm):
                px, py = int(l['x'] * w), int(l['y'] * h)  # 정규화 → 픽셀 변환
                points[idx] = (px, py)
                if l['visibility'] > 0.5:
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)  # 노란 점
            for a, b in CONNECTIONS:
                if lm[a]['visibility'] > 0.5 and lm[b]['visibility'] > 0.5:
                    cv2.line(frame, points[a], points[b], (0, 255, 0), 2)  # 초록 선

            # ── 벽 평면 갱신 ──
            # IDLE(정지) 구간에서만 벽 평면을 갱신
            move_now = frame_to_move.get(frame_idx)
            if not move_now: # 정지 구간
                wall_plane = fit_wall_plane(lm)
            elif wall_plane is None: #첫 프레임이 바로 MOVING일 때
                wall_plane = fit_wall_plane(lm)

            # ── 지표 계산 ──
            l_angle = calc_angle( #왼팔 각도
                [lm[11]['x'], lm[11]['y']], #왼쪽 어깨 x,y좌표
                [lm[13]['x'], lm[13]['y']], #왼쪽 팔꿈치 x,y좌표
                [lm[15]['x'], lm[15]['y']], #왼쪽 손목 x,y좌표
            )
            r_angle = calc_angle( #오른팔 각도
                [lm[12]['x'], lm[12]['y']], #오른쪽 어깨 x,y좌표
                [lm[14]['x'], lm[14]['y']], #오른쪽 팔꿈치 x,y좌표
                [lm[16]['x'], lm[16]['y']], #오른쪽 손목 x,y좌표
            )
            tripod = check_tripod(lm)
            w_dist = calc_wall_distance(lm, wall_plane)

            # ── 우측 하단: 지표 텍스트 ──
            lines_r = [
                f"L Elbow: {l_angle:.1f} ({'Straight' if l_angle >= 150 else 'Bent'})",
                f"R Elbow: {r_angle:.1f} ({'Straight' if r_angle >= 150 else 'Bent'})",
                f"Tripod: {tripod}",
                f"Wall Dist: {w_dist:.4f}",
            ]
            for i, txt in enumerate(lines_r):
                y_pos = h - 20 - (len(lines_r) - 1 - i) * 30
                cv2.putText(frame, txt, (w - 400, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── 좌측 하단: 동작 상태 ──
        move = frame_to_move.get(frame_idx)
        if move:
            # MOVING 구간: 동작 번호 + 시간 구간 표시
            ts = f"{int(move['time_start'] // 60)}:{move['time_start'] % 60:04.1f}"
            te = f"{int(move['time_end'] // 60)}:{move['time_end'] % 60:04.1f}"
            move_txt = f"Move {move['move_idx']}: {ts}~{te}"
            cv2.putText(frame, move_txt, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)  # 주황색
            cv2.putText(frame, "MOVING", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 빨간색
        else:
            # IDLE 구간
            cv2.putText(frame, "IDLE", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)  # 회색

        out.write(frame)  # 프레임 출력 영상에 쓰기
        frame_idx += 1

        # 진행률 (콘솔에 500프레임마다)
        if frame_idx % 500 == 0:
            print(f"  영상 생성: {frame_idx}/{total_frames} "
                  f"({frame_idx / total_frames * 100:.1f}%)")
    # 안전하게 종료
    cap.release()
    out.release()
    print(f"디버그 영상 저장: {output_path}")

# ──────────────────────────────────────────────
# 7. 결과 출력 & 저장
# ──────────────────────────────────────────────
def print_results(moves):
    header = (f"{'동작':>3} | {'시간 구간':<12} | {'길이':>4} | "f"{'L팔꿈치':>5} | {'R팔꿈치':>5} | "f"{'L직선팔':>4} | {'R직선팔':>4} | {'삼지점':>3} | {'벽 거리':>4} | {'등반 높이':>3}" )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for m in moves:
        ts = f"{int(m['time_start']//60)}:{m['time_start']%60:04.1f}"
        te = f"{int(m['time_end']//60)}:{m['time_end']%60:04.1f}"
        cy = m['climb_delta_y']
        cy_mark = "UP" if cy > 0.01 else ("DN" if cy < -0.01 else "--")
        print(f"  {m['move_idx']:>2} | "
              f"{ts}~{te:<8} | "
              f"{m['duration']:>4.1f}s | "
              f"{m['l_elbow_avg']:>6.1f}° | "
              f"{m['r_elbow_avg']:>6.1f}° | "
              f"{m['l_straight_ratio']:>5.1f}% | "
              f"{m['r_straight_ratio']:>5.1f}% | "
              f"{m['tripod_ratio']:>4.1f}%"
              f" {m['wall_dist_avg']:>.4f} |"
              f" {cy:>+.3f}{cy_mark}")

    print("=" * len(header))

    if moves:
        avg_l = np.mean([m['l_elbow_avg'] for m in moves])
        avg_r = np.mean([m['r_elbow_avg'] for m in moves])
        avg_l_str = np.mean([m['l_straight_ratio'] for m in moves])
        avg_r_str = np.mean([m['r_straight_ratio'] for m in moves])
        avg_tri = np.mean([m['tripod_ratio'] for m in moves])
        avg_wall = np.mean([m['wall_dist_avg'] for m in moves])
        total_climb = sum([m['climb_delta_y'] for m in moves])
        print(f"\n[전체 요약] 총 {len(moves)}개 동작")
        print(f"  평균 L팔꿈치: {avg_l:.1f}°, 평균 R팔꿈치: {avg_r:.1f}°")
        print(f"  직선팔 비율: L {avg_l_str:.1f}%, R {avg_r_str:.1f}%")
        print(f"  삼지점 비율: {avg_tri:.1f}%")
        print(f"  벽 거리: {avg_wall:.4f}")
        print(f"  등반 높이: {total_climb:+.3f}")

def save_results(moves, output_path): #결과를 JSON으로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(moves, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

# ──────────────────────────────────────────────
# 메인
if __name__ == "__main__":

    video_files = sorted(glob.glob(VIDEO_PATTERN))

    if not video_files:
        print(f"'{DATA_DIR}' 폴더에 영상 파일이 없음.")
        exit()

    print("=" * 50)
    print(f"  등반 영상 동작 분석 영상 {len(video_files)}개")
    print("=" * 50)

    all_results = {}

    for vi, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n{'#' * 50}")
        print(f"  [{vi + 1}/{len(video_files)}] {video_name}")
        print(f"{'#' * 50}")

        print("\n[1/4] 랜드마크 추출 중...")
        all_landmarks, fps, skip = extract_landmarks(video_path, MODEL_PATH, TARGET_FPS)
        all_landmarks = interpolate_low_visibility(all_landmarks)

        if len(all_landmarks) < 10:
            print(f"  {video_name}: 포즈 감지 부족 (최솟값: 10개), 이 영상은 건너뜀")
            continue

        print("\n[2/4] 속도 계산 중...")
        velocities = compute_velocity(all_landmarks)

        print("\n[3/4] 동작 구분 중...")
        moves, smoothed, segments = segment_moves(
            all_landmarks, velocities, fps, skip,
            VELOCITY_WINDOW, IDLE_THRESHOLD_RATIO,
            MIN_IDLE_DURATION_SEC, MIN_MOVE_DURATION_SEC
        )
        moves = [m for m in moves if m['duration'] >= 0.1]
        for i, m in enumerate(moves):
            m['move_idx'] = i + 1

        print("\n[4/4] 지표 계산 중...")
        moves = analyze_moves(moves, all_landmarks)

        print_results(moves)

        # 디버그 영상 생성 (안 쓸때 주석처리)
        debug_path = os.path.join(DATA_DIR, f"debug_rules_{video_name}")
        generate_debug_video(video_path, all_landmarks, moves, fps, skip, debug_path)

        all_results[video_name] = moves

        # 전체 결과 JSON 파일 저장
    save_results(all_results, "rules_result.json")
    print(f"\n전체 완료: {len(all_results)}개 영상 분석 결과 저장")