# M4_MLAnalysis.py
# ──────────────────────────────────────────────────────────────────
# 목표: 지도학습 + 특징 선택
#
# 코드 :
#   - 5-fold 교차검증(StratifiedKFold)으로 학습/테스트 분리 (4 학습 / 1 테스트)
#   - 베이스라인 비교(무작위 / 단일특징 / 전체 / 직선팔 제거)로 정직하게 활용할 수 있는 지표 선택
#   - AUC: 1.0=완벽, 0.5=찍기.
#   - 분류기 3종 사용 (Logistic / RandomForest / GradientBoosting)
#   - 안정성 선택(부트스트랩) + SHAP(선택) + 특징 부분집합 비교
#   - stage4_summary.png 저장 + 분류기(expert_classifier.pkl) 저장(피드백용)
#
# 사용: features.csv 를 같은 폴더에 두고 실행, (shap 라이브러리 설치는 선택)
#
# ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 9개 특징
FEATURES = ["elbow_mean_idle", "straight_ratio_idle", "tripod_ratio",
            "com_over_base_ratio", "wall_dist_idle", "wall_dist_move",
            "wall_push_ratio", "jerk_move", "leg_drive_ratio"]
# 5-fold 교차검증
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def cv_auc(model, X, y):
    #5-fold 교차검증 평균 ROC-AUC (매 fold마다 학습/테스트 분리)
    s = []
    for tr, te in CV.split(X, y):
        model.fit(X[tr], y[tr])
        s.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return float(np.mean(s)), float(np.std(s))


def main():
    df = pd.read_csv("features.csv")
    feats = [f for f in FEATURES if f in df.columns]
    y = (df['label'] == 'expert').astype(int).values # 전문가=1, 비전문가=0
    X = SimpleImputer(strategy='median').fit_transform(df[feats].values)
    print(f"표본 {len(df)} (전문가 {y.sum()}, 비전문가 {(1-y).sum()})")
    # 분류기 3종
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=0)
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    gb = lambda: GradientBoostingClassifier(random_state=0)

    # 3모델(Logistic/RF/GB) 5-fold  AUC의 평균
    def auc3(Xs, yy):
        return float(np.mean([cv_auc(m(), Xs, yy)[0] for m in (logit, rf, gb)]))

    # ── 베이스라인 (3모델) ──
    print("\n=== 베이스라인 (5-fold CV AUC) ===")
    ys = y.copy(); np.random.default_rng(0).shuffle(ys)
    auc_rand = cv_auc(rf(), X, ys)[0]
    print(f"  무작위(레이블 셔플, LS): {auc_rand:.3f}   (≈0.5면 정상)")
    print(f"\n  {'단일특징':20} {'Logistic':>9} {'RF':>7} {'GB':>7}")
    for f in feats:
        Xi = X[:, [feats.index(f)]]
        print(f"  {f:20} {cv_auc(logit(), Xi, y)[0]:>9.3f} "
              f"{cv_auc(rf(), Xi, y)[0]:>7.3f} {cv_auc(gb(), Xi, y)[0]:>7.3f}")

    # 전체 9특징, 모델 3종 (결과가 모델 종류에 의존하지 않는지 확인)
    print("\n=== 전체 9특징 (모델 3종) ===")
    auc_all = None
    for nm, mk in [("Logistic", logit), ("RandomForest", rf), ("GradientBoosting", gb)]:
        mu, sd = cv_auc(mk(), X, y)
        if nm == "RandomForest":
            auc_all = mu
        print(f"  {nm:16}: {mu:.3f} ± {sd:.3f}")

    # 직선팔 2개를 빼면 얼마나 떨어지나
    keep = [i for i, f in enumerate(feats) if f not in ("elbow_mean_idle", "straight_ratio_idle")]
    auc_nostraight = cv_auc(rf(), X[:, keep], y)[0]
    print(f"  직선팔 2개 제거(나머지)    : {auc_nostraight:.3f}")

    # 특징 선택용: 부분집합 비교
    print("\n=== 특징 선택: 부분집합 (RF CV AUC) ===")
    SUBSETS = {
        "elbow only":          ["elbow_mean_idle"],
        "elbow+tripod":        ["elbow_mean_idle", "tripod_ratio"],
        "선택5(약신호4 제거)": ["elbow_mean_idle", "straight_ratio_idle", "tripod_ratio",
                                "com_over_base_ratio", "wall_dist_idle"],
        "전체9":               feats,
    }
    for nm, cols in SUBSETS.items():
        cols = [c for c in cols if c in feats]
        Xs = X[:, [feats.index(c) for c in cols]]
        print(f"  {nm:22}: {auc3(Xs, y):.3f}  ({len(cols)}개)")

    # 안정성 선택 (부트스트랩 50회, top-3 포함 빈도, 자주 뽑힐수록 안정적 지표)
    print("\n=== 안정성 선택 (부트스트랩 50회, top-3 포함 빈도%) ===")
    rng = np.random.default_rng(1); top3 = np.zeros(len(feats))
    for r in range(50):
        idx = rng.choice(len(y), len(y), replace=True) # 복원추출 표본
        m = RandomForestClassifier(200, random_state=r).fit(X[idx], y[idx])
        for j in np.argsort(m.feature_importances_)[::-1][:3]: # 중요도 상위 3개
            top3[j] += 1
    for f, c in sorted(zip(feats, top3 / 50 * 100), key=lambda t: -t[1]):
        print(f"  {f:20}: {c:5.0f}%")

    # SHAP (각 특징이 예측에 기여한 평균 크기, 있으면 적용됨)
    m_full = RandomForestClassifier(300, random_state=0).fit(X, y)
    try:
        import shap
        sv = shap.TreeExplainer(m_full).shap_values(X)
        sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        print("\n=== SHAP 평균 |값| (내림차순) ===")
        for f, v in sorted(zip(feats, np.abs(sv1).mean(0)), key=lambda t: -t[1]):
            print(f"  {f:20}: {v:.4f}")
    except Exception as e:
        print("\n(SHAP 생략:", e, ")")

    # 분류기 저장(피드백의 종합점수용)
    joblib.dump({"model": m_full, "features": feats,
                 "imputer_median": np.nanmedian(df[feats].values, axis=0)},
                "expert_classifier.pkl")
    print("저장: expert_classifier.pkl")


if __name__ == "__main__":
    main()