# M2_Segmentation.py
# ──────────────────────────────────────────────────────────────────
# 목표: 동작/정지 구분(segmentation)의 대안 3종 비교 및 최종 선택 : K-Means, GMM, HMM
#       디버깅 영상과 그래프 제작 후, 4가지 방법 중 최종 선택
#       최종 선택: HMM 사용
#
# 배경:
#   M1.py에서 규칙 기반으로 "속도 중앙값 × 비율"을 임계값으로 사용
#   시간 순서를 보지 않기 때문에, 영상에 따라 한 동작이 수십 초로 뭉치거나 0.1초로 파편화되는 문제가 관찰됨
#
# 코드:
#   - 입력은 모두 "프레임별 속도(velocities)" 1차원 시계열 (M2.compute_velocity)
#   - 정지/이동을 2개 그룹으로 분류
#   - 평균 속도가 더 낮은 쪽을 정지로 적용(레이블 불필요).
#
# 방법별 차이:
#   - K-Means : 속도값을 2개 군집으로 분류
#               임계값을 데이터가 자동 결정
#               시간순서 무시
#   - GMM     : 속도를 2개 정규분포 혼합으로 보고 확률적 분류
#               분산이 다른 두 분포를 K-Means보다 유연하게 잡음
#               시간순서 무시
#   - HMM     : 2개 은닉상태 + 전이확률로 '시간 순서'까지 모델링.
#               한 번 정지면 잠깐의 속도 튐으로 쉽게 이동으로 바뀌지 않음 (짧은구간 병합이 거의 불필요)
#
# ──────────────────────────────────────────────────────────────────



import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

# 1 의 기존 함수 재사용 (재작성하지 않음)
#  - moving_average : 속도 평활
#  - segment_moves  : 규칙 기반(비교 baseline)
#  - 디버그 영상의 '우측'(스켈레톤 + 지표 텍스트)을 M2와 동일하게 그리기 위한 함수들
from M1_Rules import (moving_average, segment_moves, CONNECTIONS,
                    calc_angle, check_tripod, fit_wall_plane, calc_wall_distance)

# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────
def _prepare_feature(velocities, velocity_window, use_log):
#속도를 평활하고, 군집용 2차원 feature 배열 (N,1)로 변환.
#use_log=True 면 log(1+v) 변환. 클라이밍 속도는 0 근처에 몰리고 이동 구간에서 길게 꼬리를 끄는 분포(우편향)
# 로그 변환이 정지/이동을 더 잘 분리할 수도 있음
# 어느 쪽이 나은지는 데이터로 확인 필요

    smoothed = moving_average(velocities, velocity_window)
    feat = np.log1p(smoothed) if use_log else smoothed.copy()
    return smoothed, feat.reshape(-1, 1)


def states_to_moves(is_still, all_landmarks, fps, frame_skip,
                    min_still_sec, min_move_sec, do_merge=True):
#프레임별 정지여부 배열(is_still, bool) → 동작(move) 리스트.
#2.segment_moves 의 후처리 로직(짧은 구간 병합 + 인접 병합 + 이동 구간만 move로 추출)을 그대로 사용.
#4가지 방법이 동일한 후처리를 쓰도록 분리한 것이라 비교할때 정확
#HMM의 원시 출력 확인용: do_merge=False 면 짧은구간 병합 스킵

    is_still = np.asarray(is_still, dtype=bool)
    n = len(is_still)
    effective_fps = fps / frame_skip
    min_still_frames = int(min_still_sec * effective_fps)
    min_move_frames = int(min_move_sec * effective_fps)

    # 1) 연속된 같은 상태를 구간으로 묶기
    segments = []
    cur = is_still[0]
    start = 0
    for i in range(1, n):
        if is_still[i] != cur:
            segments.append((start, i - 1, cur))
            cur = is_still[i]
            start = i
    segments.append((start, n - 1, cur))

    # 2) 너무 짧은 구간 흡수 (짧은 정지→이동, 짧은 이동→정지)
    if do_merge:
        filtered = []
        for s, e, state in segments:
            dur = e - s + 1
            if state and dur < min_still_frames:
                filtered.append((s, e, False))
            elif (not state) and dur < min_move_frames:
                filtered.append((s, e, True))
            else:
                filtered.append((s, e, state))
    # 3) 인접한 같은 상태 합치기
        merged = [filtered[0]]
        for i in range(1, len(filtered)):
            if filtered[i][2] == merged[-1][2]:
                merged[-1] = (merged[-1][0], filtered[i][1], merged[-1][2])
            else:
                merged.append(filtered[i])
    else:
        merged = segments

    # 4) 이동 구간(state=False)만 move로 추출
    moves = []
    for s, e, state in merged:
        if not state:
            t_start = all_landmarks[s]['time_sec']
            t_end = all_landmarks[min(e, len(all_landmarks) - 1)]['time_sec']
            moves.append({
                'move_idx': len(moves) + 1,
                'start_frame_idx': s,
                'end_frame_idx': e,
                'time_start': round(t_start, 2),
                'time_end': round(t_end, 2),
                'duration': round(t_end - t_start, 2),
            })
    return moves, merged

# 레이블별 평균 속도 중 가장 낮은 레이블 => 정지.
def _pick_still_label(values_per_label):
    return int(np.argmin(values_per_label))


# ──────────────────────────────────────────────
# 방법 2) K-Means
# ──────────────────────────────────────────────
def segment_kmeans(all_landmarks, velocities, fps, frame_skip,
                   velocity_window=5, min_still_sec=0.3, min_move_sec=0.3,
                   use_log=False, do_merge=True, random_state=0):
    smoothed, X = _prepare_feature(velocities, velocity_window, use_log)
    km = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(X)
    still_label = _pick_still_label(km.cluster_centers_.ravel())  # 중심 낮은 군집 = 정지
    is_still = (km.labels_ == still_label)
    moves, merged = states_to_moves(is_still, all_landmarks, fps, frame_skip,
                                    min_still_sec, min_move_sec, do_merge=do_merge)
    return moves, smoothed, merged, is_still


# ──────────────────────────────────────────────
# 방법 3) GMM
# ──────────────────────────────────────────────
def segment_gmm(all_landmarks, velocities, fps, frame_skip,
                velocity_window=5, min_still_sec=0.3, min_move_sec=0.3,
                use_log=False, do_merge=True, random_state=0):
    smoothed, X = _prepare_feature(velocities, velocity_window, use_log)
    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          n_init=5, random_state=random_state).fit(X)
    still_comp = _pick_still_label(gmm.means_.ravel())  # 평균 낮은 성분 = 정지
    is_still = (gmm.predict(X) == still_comp)
    moves, merged = states_to_moves(is_still, all_landmarks, fps, frame_skip,
                                    min_still_sec, min_move_sec, do_merge=do_merge)
    return moves, smoothed, merged, is_still


# ──────────────────────────────────────────────
# 방법 4) HMM
# ──────────────────────────────────────────────
def segment_hmm(all_landmarks, velocities, fps, frame_skip,
                velocity_window=5, min_still_sec=0.3, min_move_sec=0.3,
                use_log=False, do_merge=True, random_state=0):
    smoothed, X = _prepare_feature(velocities, velocity_window, use_log)
    model = hmm.GaussianHMM(n_components=2, covariance_type='diag',
                            n_iter=200, random_state=random_state)
    model.fit(X)
    still_state = _pick_still_label(model.means_.ravel())  # 평균 낮은 상태 = 정지
    is_still = (model.predict(X) == still_state)            # Viterbi 디코딩
    moves, merged = states_to_moves(is_still, all_landmarks, fps, frame_skip,
                                    min_still_sec, min_move_sec, do_merge=do_merge)
    return moves, smoothed, merged, is_still


# ──────────────────────────────────────────────
# 4종 비교
# ──────────────────────────────────────────────
def _move_stats(moves):
    if not moves:
        return dict(n=0, med=0, mn=0, mx=0)
    d = np.array([m['duration'] for m in moves])
    return dict(n=len(moves), med=float(np.median(d)),
                mn=float(d.min()), mx=float(d.max()))


def compare_all_methods(all_landmarks, velocities, fps, frame_skip,
                        velocity_window=5, idle_threshold_ratio=0.3,
                        min_still_sec=0.3, min_move_sec=0.3, use_log=False):
    """규칙기반 + K-Means + GMM + HMM 을 같은 데이터에 돌려 비교.
    반환: {method_name: {'moves':..., 'is_still':..., 'smoothed':...}}
    """
    # 방법별 짧은구간 병합 여부
    #   Rule-based 방식은 예외로 항상 병합
    do_merge_kmeans = True
    do_merge_gmm    = True
    do_merge_hmm    = True

    results = {}

    # 규칙 기반 (1_Rules 그대로) — 병합 항상 적용(설정 없음)
    rb_moves, rb_smoothed, rb_merged = segment_moves(
        all_landmarks, velocities, fps, frame_skip,
        velocity_window, idle_threshold_ratio, min_still_sec, min_move_sec)
    rb_is_still = np.zeros(len(all_landmarks), dtype=bool)
    for s, e, state in rb_merged:
        if state:
            rb_is_still[s:e + 1] = True
    results['Rule'] = dict(moves=rb_moves, is_still=rb_is_still, smoothed=rb_smoothed)

    km_moves, sm, _, km_still = segment_kmeans(
        all_landmarks, velocities, fps, frame_skip,
        velocity_window, min_still_sec, min_move_sec, use_log, do_merge_kmeans)
    results['KMeans'] = dict(moves=km_moves, is_still=km_still, smoothed=sm)

    gm_moves, _, _, gm_still = segment_gmm(
        all_landmarks, velocities, fps, frame_skip,
        velocity_window, min_still_sec, min_move_sec, use_log, do_merge_gmm)
    results['GMM'] = dict(moves=gm_moves, is_still=gm_still, smoothed=sm)

    hm_moves, _, _, hm_still = segment_hmm(
        all_landmarks, velocities, fps, frame_skip,
        velocity_window, min_still_sec, min_move_sec, use_log, do_merge_hmm)
    results['HMM'] = dict(moves=hm_moves, is_still=hm_still, smoothed=sm)

    # 요약 출력
    print(f"\n{'방법':<8} | {'동작수':>5} | {'중앙길이':>7} | {'최소':>5} | {'최대':>6}")
    print("-" * 44)
    for name, r in results.items():
        st = _move_stats(r['moves'])
        print(f"{name:<8} | {st['n']:>5} | {st['med']:>6.2f}s | "
              f"{st['mn']:>4.2f}s | {st['mx']:>5.2f}s")
    return results


def plot_comparison(results, all_landmarks, out_path, true_is_still=None, sec_per_inch=3.6):
    # 속도 타임라인 위에 각 방법의 '이동 구간'을 색칠하는 사진 제작 (검증용)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = np.array([f['time_sec'] for f in all_landmarks])
    smoothed = results['Rule']['smoothed']
    names = list(results.keys())
    rows = len(names) + (1 if true_is_still is not None else 0)

    duration = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    width = max(14, duration / sec_per_inch)  # 영상 길이에 비례해 가로 폭 자동 확대
    fig, axes = plt.subplots(rows, 1, figsize=(width, 1.9 * rows), sharex=True)

    if rows == 1:
        axes = [axes]

    idx = 0
    if true_is_still is not None:
        ax = axes[idx]; idx += 1
        ax.plot(times, smoothed, lw=0.6, color='black')
        _shade_moving(ax, times, ~np.asarray(true_is_still), 'green')
        ax.set_ylabel('GT(true)', rotation=0, ha='right', va='center')
        ax.set_yticks([])

    for name in names:
        ax = axes[idx]; idx += 1
        ax.plot(times, smoothed, lw=0.6, color='black')
        _shade_moving(ax, times, ~results[name]['is_still'], 'tab:red')
        n = len(results[name]['moves'])
        ax.set_ylabel(f'{name}\n(n={n})', rotation=0, ha='right', va='center')
        ax.set_yticks([])

    axes[-1].set_xlabel('time (s)   |   shaded span = MOVING')
    fig.suptitle('Segmentation comparison: smoothed velocity + MOVING spans', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\n비교 플롯 저장: {out_path}")


def _shade_moving(ax, times, is_moving, color):
    is_moving = np.asarray(is_moving, dtype=bool)
    n = len(is_moving)
    i = 0
    while i < n:
        if is_moving[i]:
            j = i
            while j < n and is_moving[j]:
                j += 1
            ax.axvspan(times[i], times[min(j, n - 1)], color=color, alpha=0.3)
            i = j
        else:
            i += 1


# ──────────────────────────────────────────────
# 4종 비교 디버그 영상
#   - 우측(스켈레톤 + 지표 텍스트): 1과 동일
#   - 좌측 하단: 4개 칸(Rule / KMeans / GMM / HMM) 각각 IDLE/MOVING 표시
# ──────────────────────────────────────────────
def generate_debug_video_compare(video_path, all_landmarks, results, fps,
                                  frame_skip, output_path,
                                  method_order=('Rule', 'KMeans', 'GMM', 'HMM')):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 원본 프레임 번호 → 랜드마크 / 분석 인덱스 매핑
    frame_to_lm = {}
    frame_to_aidx = {}
    for i, entry in enumerate(all_landmarks):
        frame_to_lm[entry['frame_idx']] = entry['landmarks']
        frame_to_aidx[entry['frame_idx']] = i

    # 우측 지표/벽평면 갱신은 1 와 동일
    rule_moves = results['Rule']['moves']
    frame_to_move = {}
    for move in rule_moves:
        s = all_landmarks[move['start_frame_idx']]['frame_idx']
        e = all_landmarks[min(move['end_frame_idx'], len(all_landmarks) - 1)]['frame_idx']
        for fi in range(s, e + 1):
            frame_to_move[fi] = move

    # 각 방법의 프레임별 정지여부 (분석 인덱스 기준)
    states = {m: np.asarray(results[m]['is_still'], dtype=bool) for m in method_order}

    COL_IDLE = (200, 200, 200)  # 회색 = 정지
    COL_MOVE = (0, 0, 255)      # 빨강 = 이동
    n_m = len(method_order)

    frame_idx = 0
    last_aidx = 0           # 분석 안 된(skip된) 프레임은 직전 분석 상태를 유지
    wall_plane = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_to_aidx:
            last_aidx = frame_to_aidx[frame_idx]
        lm = frame_to_lm.get(frame_idx)

        # ── 우측: 1 와 동일
        if lm:
            points = {}
            for idx, l in enumerate(lm):
                px, py = int(l['x'] * w), int(l['y'] * h)
                points[idx] = (px, py)
                if l['visibility'] > 0.5:
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
            for a, b in CONNECTIONS:
                if lm[a]['visibility'] > 0.5 and lm[b]['visibility'] > 0.5:
                    cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

            move_now = frame_to_move.get(frame_idx)
            if not move_now:               # 규칙기반 IDLE 구간에서 벽평면 갱신
                wall_plane = fit_wall_plane(lm)
            elif wall_plane is None:
                wall_plane = fit_wall_plane(lm)

            l_angle = calc_angle([lm[11]['x'], lm[11]['y']], [lm[13]['x'], lm[13]['y']],
                                 [lm[15]['x'], lm[15]['y']])
            r_angle = calc_angle([lm[12]['x'], lm[12]['y']], [lm[14]['x'], lm[14]['y']],
                                 [lm[16]['x'], lm[16]['y']])
            tripod = check_tripod(lm)
            w_dist = calc_wall_distance(lm, wall_plane)
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

        # ── 좌측 하단: 4종 방법 칸 (IDLE/MOVING) ──
        for i, m in enumerate(method_order):
            st = states[m]
            still = bool(st[last_aidx]) if last_aidx < len(st) else True
            label = "IDLE" if still else "MOVING"
            col = COL_IDLE if still else COL_MOVE
            y_pos = h - 20 - (n_m - 1 - i) * 30   # 위→아래: Rule, KMeans, GMM, HMM
            cv2.putText(frame, f"{m}: {label}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  비교 영상 생성: {frame_idx}/{total_frames} "
                  f"({frame_idx / total_frames * 100:.1f}%)")

    cap.release()
    out.release()
    print(f"비교 디버그 영상 저장: {output_path}")


# ──────────────────────────────────────────────
# 메인: T00.mp4 등에서 추출 → 4종 비교 → 플롯 + 비교 디버그 영상
#   랜드마크 캐시(.pkl)가 있으면 재사용해 MediaPipe 재실행을 스킵
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import os, glob, pickle
    from M1_Rules import (extract_landmarks, compute_velocity,
                        interpolate_low_visibility, MODEL_PATH, TARGET_FPS,
                        VELOCITY_WINDOW, IDLE_THRESHOLD_RATIO,
                        MIN_IDLE_DURATION_SEC, MIN_MOVE_DURATION_SEC, DATA_DIR)

    MAKE_DEBUG_VIDEO = True # 4종 비교 디버그 영상 생성 여부

    video_files = sorted(glob.glob(os.path.join(DATA_DIR, "T*.mp4")))
    if not video_files:
        print(f"'{DATA_DIR}' 에 영상이 없음.")
        raise SystemExit

    video_path = video_files[0]
    name = os.path.basename(video_path)
    # 랜드마크 전체를 캐시(.pkl)로 저장 → 비교/플롯/디버그영상 모두 재사용 가능
    cache = os.path.join(DATA_DIR, f"_cache_{name}.pkl")

    if os.path.exists(cache):
        print(f"캐시 사용: {cache}")
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        all_landmarks = data['all_landmarks']
        fps = data['fps']; frame_skip = data['frame_skip']
    else:
        print(f"추출: {video_path}")
        all_landmarks, fps, frame_skip = extract_landmarks(video_path, MODEL_PATH, TARGET_FPS)
        all_landmarks = interpolate_low_visibility(all_landmarks)
        with open(cache, 'wb') as f:
            pickle.dump({'all_landmarks': all_landmarks, 'fps': fps,
                         'frame_skip': frame_skip}, f)
        print(f"캐시 저장: {cache}")

    velocities = compute_velocity(all_landmarks)

    results = compare_all_methods(
        all_landmarks, velocities, fps, frame_skip,
        VELOCITY_WINDOW, IDLE_THRESHOLD_RATIO,
        MIN_IDLE_DURATION_SEC, MIN_MOVE_DURATION_SEC, use_log=False)

    plot_comparison(results, all_landmarks,
                    os.path.join(DATA_DIR, f"segmentation_{name}.png"))

    if MAKE_DEBUG_VIDEO:
        generate_debug_video_compare(
            video_path, all_landmarks, results, fps, frame_skip,
            os.path.join(DATA_DIR, f"debug_segmentation_{name}.mp4"))