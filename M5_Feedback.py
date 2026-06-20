# M5_Feedback.py
# ──────────────────────────────────────────────────────────────────
# 5단계: 피드백 콘솔 출력
#
# 코드:
#   영상 1개 → ① 전문가 유사도 점수(분류기) ② 직선팔 피드백 ③ 9지표 참고표를 콘솔에 출력.
#   동시에 analyze()가 dict를 반환하므로 GUI/웹이 그대로 표시 가능.
#
# 설계(4단계 결과 반영):
#   - 개별 피드백은 '검증된 직선팔'(elbow_mean_idle)만 사용. tripod는 약한 참고.
#   - 나머지 지표는 '참고용(평가 미사용)'으로 표에만 표시.
#   - 진행도% = (사용자 - 일반인평균)/(전문가평균 - 일반인평균)×100
#       100%↑ = 전문가 수준 / 0%~100% = 그 사이 / 0%↓ = 일반인 평균 미달
#
# 사용: python M5_Feedback.py <영상경로>
#   같은 폴더에 features.csv, expert_classifier.pkl 필요.
#   FeedbackSingleVideo 폴더의 모든 mp4를 차례로 분석.
#   영상 특징 추출은 M3_ExpandedMetrics(MediaPipe) 사용.
#   GUI/웹에서는 analyze(user_feats) 의 반환 dict를 받아 화면에 그리면 됨.
# ──────────────────────────────────────────────────────────────────
import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib

FEEDBACK_DIR = "FeedbackSingleVideo"  # 이 폴더 안의 모든 mp4를 분석

# 검증된(개별 피드백 가능) 지표 — 4단계 결과
PRIMARY = "elbow_mean_idle"  # 쉴 때 평균 팔꿈치 각도(클수록=직선팔=좋음)

# 지표 한글 이름(표/콘솔 표시용)
LABELS_KO = {
    "elbow_mean_idle": "쉴 때 직선팔(팔꿈치 각도)",
    "straight_ratio_idle": "쉴 때 직선팔 비율",
    "tripod_ratio": "삼지점 비율",
    "com_over_base_ratio": "무게중심 균형",
    "wall_dist_idle": "벽 거리(정지)",
    "wall_dist_move": "벽 거리(이동)",
    "wall_push_ratio": "이동 전 벽에서 멀어짐",
    "jerk_move": "동작 부드러움(저크)",
    "leg_drive_ratio": "다리 추진",
}


def load_reference(csv="features.csv"):
    #features.csv 지표별 전문가/일반인 평균(비교 기준)
    df = pd.read_csv(csv)
    feats = [c for c in df.columns if c in LABELS_KO]
    ref = {}
    for f in feats:
        e = df.loc[df.label == "expert", f].dropna()  # 전문가 값들
        n = df.loc[df.label != "expert", f].dropna()  # 비전문가 값들
        ref[f] = dict(exp_mean=float(e.mean()), non_mean=float(n.mean()))
    return ref, feats


def progress_pct(val, non_mean, exp_mean):
    # 일반인평균=0%, 전문가평균=100% 기준 진행도
    if exp_mean == non_mean:
        return float("nan")
    return (val - non_mean) / (exp_mean - non_mean) * 100.0


def straight_arm_message(val, ref):
    # 직선팔(elbow) 3단계 피드백 문구 + 상태 라벨
    e, n = ref[PRIMARY]["exp_mean"], ref[PRIMARY]["non_mean"]
    p = progress_pct(val, n, e)
    if val >= e:  # 전문가 평균 이상
        return "GOOD", f"전문가 수준입니다 (평균 {e:.0f}° 이상). 좋은 자세를 계속 유지하고 있습니다!"
    elif val >= n:  # 일반인~전문가 사이
        return "MID", (f"전문가 평균({e:.0f}°)까지 +{e - val:.0f}° 더 펴면 됩니다. "
                       f"(현재 {val:.0f}°, 전문가 수준의 {p:.0f}%에 도달한 상태입니다.)")
    else:  # 일반인 평균 미달
        return "LOW", (f"쉴 때 팔이 많이 굽어 있습니다. 일반인 평균({n:.0f}°)까지 +{n - val:.0f}°, "
                       f"전문가 평균({e:.0f}°)까지 +{e - val:.0f}° — 쉴 때 팔을 펴 전완근 부담을 줄이세요.")


def expert_score(user_feats, clf_path="expert_classifier.pkl"):
    # 분류기로 전문가 유사도(%) 산출. 특징은 학습 때 중앙값으로 채움
    d = joblib.load(clf_path)
    x = np.array([[user_feats.get(f, np.nan) for f in d["features"]]], float)
    med = np.asarray(d["imputer_median"], float)
    nanmask = np.isnan(x)
    x[nanmask] = np.take(med, np.where(nanmask)[1])
    return float(d["model"].predict_proba(x)[0, 1]) * 100.0

def analyze(user_feats, csv="features.csv", clf="expert_classifier.pkl"):
    #사용자 특징 dict → 결과 dict (GUI/웹이 이 dict를 그대로 표시).
    #반환:
    #  {
    #    'score': 전문가 유사도(%),
    #    'straight_arm': {value, exp_mean, non_mean, progress, status, message},
    #    'table': [ {name, key, value, exp, non, tag}, ... ]   # 9지표 참고표
    #  }

    ref, feats = load_reference(csv)
    val = user_feats[PRIMARY]
    status, msg = straight_arm_message(val, ref)
    p = progress_pct(val, ref[PRIMARY]["non_mean"], ref[PRIMARY]["exp_mean"])
    # 종합 점수: 분류기 pkl 있으면 그걸로, 없으면 검증된 직선팔 진행도(0~100)로 대체
    try:
        score = expert_score(user_feats, clf);
        score_source = "분류기"
    except Exception:
        score = float(min(100, max(0, p)));
        score_source = "직선팔 기반(분류기 없음)"
    sa = dict(value=val, exp_mean=ref[PRIMARY]["exp_mean"], non_mean=ref[PRIMARY]["non_mean"],
              progress=p, status=status, message=msg)

    table = []
    for f in feats:
        tag = "검증됨" if f == PRIMARY else ("약한참고" if f == "tripod_ratio" else "참고용")
        table.append(dict(name=LABELS_KO[f], key=f, value=float(user_feats.get(f, np.nan)),
                          exp=ref[f]["exp_mean"], non=ref[f]["non_mean"], tag=tag))
    return dict(score=score, score_source=score_source, straight_arm=sa, table=table)


def print_console(result, video_name=""):
    #결과 dict를 콘솔에  출력
    print("\n" + "=" * 56)
    print(f"  클라이밍 자세 분석 리포트   {video_name}")
    print(" !! 프로그램 학습에 사용된 데이터 영상의 갯수가 적어 내용이 정확하지 않을 수도 있습니다.")
    print("=" * 56)
    print(f"  전문가 유사도: {result['score']:.0f}% ")

    sa = result["straight_arm"]
    print("\n  [핵심 피드백 · 쉴 때 직선팔 — 검증된 지표]")
    print(f"   유사도 {sa['progress']:.0f}%  (일반인평균 0% ~ 전문가평균 100%)")
    print(f"   ▶ {sa['message']}")

    print("\n  [참고 지표]  (데이터 영상 부족으로 검증이 확실히 안되어 평가에는 사용하지 않습니다.)")
    print(f"   {'지표':22} {'내값':>8} {'전문가':>8} {'일반인':>8}  구분")
    print("   " + "-" * 56)
    for r in result["table"]:
        print(f"   {r['name']:22} {r['value']:>8.3f} {r['exp']:>8.3f} {r['non']:>8.3f}  {r['tag']}")

    print("\n  ※ 검증된 지표는 '쉴 때 직선팔' 하나입니다. 나머지는 참고용이며,")
    print("     전문가 분포와의 '차이'를 보여줄 뿐 절대적 정답은 아닙니다.")
    print("=" * 56)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else FEEDBACK_DIR
    videos = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    if not videos:
        print(f"'{folder}' 폴더에 mp4가 없습니다.");
        return
    from M3_ExpandedMetrics import extract_features  # MediaPipe로 9지표 추출
    print(f"'{folder}'의 영상 {len(videos)}개 분석 시작")
    for video in videos:
        name = os.path.basename(video)
        feats = extract_features(video)
        if feats is None:
            print(f"\n[건너뜀] 포즈 감지 부족: {name}");
            continue
        result = analyze(feats)
        print_console(result, name)


if __name__ == "__main__":
    main()