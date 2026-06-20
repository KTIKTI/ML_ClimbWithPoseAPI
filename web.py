# web_app.py
# ──────────────────────────────────────────────────────────────────
# 목표: 클라이밍 자세 분석 웹 (업로드 → 분석 → 결과 메시지)
#   - 데스크탑을 서버로 켜두고, 폰/PC에서 영상 업로드 -> 그 자리에서 분석 -> 결과 페이지 출력
#   - 분석 로직은 기존 파일들 (M3(추출)·M5(analyze)) 재사용
#
# 실행: 같은 폴더에 M1_Rules.py, M2_Segmentation.py, M3_ExpandedMetrics.py, M5_Feedback.py,
#      pose_landmarker_heavy.task, features.csv, expert_classifier.pkl 이 있어야 함.
#
#        1) 터미널 A:  .\.venv\Scripts\Activate.ps1 -> python web_app.py
#        2) 터미널 B:  cloudflared tunnel --url http://localhost:5000
#        출력되는 https://xxxx.trycloudflare.com 주소를 공유 (URL은 켤 때마다 바뀜)
#
# 중지: 터미널에 Ctrl+C 입력
#
# 접속:   (디버깅용) 같은 PC: http://localhost:5000
#        (같은 와이파이) 내 다른 기기:  http://<이 PC의 IPv4>:5000
#        외부망에서 접속: 터미널 B에서 출력된 https://xxxx.trycloudfflare.com
#
# 비고:
#     - 524 타임아웃 오류 방지: 업로드 즉시 결과 페이지로 보내고, 그 페이지가 4초마다 자동 새로고침.
#     - 한 번에 하나씩 순차 처리.
#
# 저장위치:  results/<job_id>.json (리포트),
#           uploads/ (원본 영상)
# ──────────────────────────────────────────────────────────────────

import os
import re
import csv
import json
import time
import glob
import uuid
import queue
import threading
from flask import Flask, request, render_template_string, redirect, url_for, send_file, abort
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

from M5_Feedback import analyze

# ── 설정 ──
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXAMPLE_VIDEO = "WebExampleVideo/T00.mp4"  # 예시 영상 경로. 없으면 플레이어 숨김.
GUIDE_LINES = [  # 업로드 안내 문구(접이식)
    "클라이머의 전신(손끝~발끝)이 화면에 잘 보이는 영상을 업로드해 주세요.",
    "클라이머를 제외한 다른 사람이 영상에 나오지 않도록 해주세요.",
    "길이는 30초~3분, 한 번의 등반(루트 1개)을 업로드해 주세요.",
    "화면 전환 없이 연속적인 영상을 업로드해 주세요.",
    "영상 길이 1분당 약 2~3분이 소요되며, 앞사람의 영상 분석이 끝나면 자동으로 다음 영상을 분석합니다. 대기열을 확인해 주세요.",
    "아래 예시 영상(T00.mp4)을 참고해 주세요."
]

# 구글 폼 사용:
SURVEY_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScMUARQRpHeQwr2l4F-8oWwr-h6hdQn_oY-3OaunGrYL9wcrw/viewform?usp=dialog"  # 비우면 버튼 숨김
# (앱 내장 설문용 — 지금은 미사용)
SURVEY_CSV = "survey_responses.csv"
GMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@gmail\.com$")  # 미사용. 설문: @gmail.com 형식 검증
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")  # 리포트 받을 이메일(일반)

# 리포트 이메일 발송용, SENDER_GMAIL 비우면 이메일 버튼 숨김.
#   발신 Gmail 계정에 2단계 인증 -> '앱 비밀번호'(16자리) 발급.
#   비밀번호는 깃허브 유출 방지로 환경변수 사용: 터미널에 setx GMAIL_APP_PW "앱비번16자리"
SENDER_GMAIL = "ofri8879s@gmail.com"
SENDER_APP_PASSWORD = os.environ.get("GMAIL_APP_PW")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def _save_result(job_id):
    with jobs_lock:
        job = dict(jobs[job_id])
    job["job_id"] = job_id
    job["saved_at"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(RESULTS_DIR, f"{job_id}.json"), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)


def _load_result(job_id):
    p = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _reports_by_nickname(nick):
    out = []
    for fn in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        try:
            with open(fn, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("nickname") == nick and d.get("status") == "done":
            out.append(d)
    out.sort(key=lambda d: d.get("saved_at", ""), reverse=True)
    return out


def _report_email_html(job):
    # 리포트를 HTML 메일 본문으로 구성.
    r = job["result"];
    sa = r["straight_arm"]
    rows = "".join(
        f"<tr><td style='text-align:left'>{t['name']}</td><td>{t['value']:.3f}</td>"
        f"<td>{t['exp']:.3f}</td><td>{t['non']:.3f}</td><td>{t['tag']}</td></tr>"
        for t in r["table"])
    return (
        "<div style='font-family:Malgun Gothic,Arial,sans-serif;max-width:640px'>"
        "<h2>클라이밍 자세 분석 리포트</h2>"
        f"<p style='color:#888'>영상: {job['filename']} · ⚠ 학습에 쓴 데이터 영상 수가 적어 결과가 정확하지 않을 수 있습니다.</p>"
        f"<p style='font-size:30px;color:#1565c0;font-weight:bold;margin:4px 0'>전문가 유사도 {r['score']:.0f}%</p>"
        "<div style='background:#fff3e0;border-left:6px solid #ef6c00;padding:12px;border-radius:8px'>"
        "<b>핵심 피드백 · 쉴 때 직선팔 (검증된 지표)</b><br>"
        f"유사도 {sa['progress']:.0f}% (일반인 0% ~ 전문가 100%)<br>▶ {sa['message']}</div>"
        "<h3>참고 지표 <span style='color:#888;font-size:13px'>데이터 영상을 수작업으로 모으는 한계로 미검증된 요소들이 많으며, 이는 피드백에 사용되지 않습니다.</span></h3>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<tr style='background:#f5f5f5'><th>지표</th><th>내 값</th><th>전문가</th><th>일반인</th><th>구분</th></tr>"
        f"{rows}</table>"
        "<p style='color:#888;font-size:12px'>※ 검증된 지표는 '쉴 때 직선팔' 하나입니다. 나머지는 참고용이며, 전문가 분포와의 '차이'일 뿐 절대적 정답은 아닙니다.</p></div>")


def send_report_email(to_addr, job):
    # HTML 리포트를 Gmail SMTP로 발송.
    msg = MIMEText(_report_email_html(job), "html", "utf-8")
    msg["Subject"] = "클라이밍 자세 분석 리포트"
    msg["From"] = formataddr(("Beta Finder", SENDER_GMAIL))
    msg["To"] = to_addr
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(SENDER_GMAIL, SENDER_APP_PASSWORD)
        s.sendmail(SENDER_GMAIL, [to_addr], msg.as_string())


def _worker():
    from M3_ExpandedMetrics import extract_features
    while True:
        job_id, path = job_queue.get()
        with jobs_lock:
            jobs[job_id]["status"] = "running"
        try:
            feats = extract_features(path)
            upd = (dict(status="error", error="포즈 감지가 부족해 분석할 수 없습니다.")
                   if feats is None else dict(status="done", result=analyze(feats)))
        except Exception as e:
            upd = dict(status="error", error=str(e))
        with jobs_lock:
            jobs[job_id].update(upd)
        _save_result(job_id)
        job_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _ahead_count(job_id):
    # 현 작업 앞에 '대기 중(pending)'인 작업 수.
    with jobs_lock:
        pending = [j for j, v in jobs.items() if v["status"] == "pending"]
    return pending.index(job_id) if job_id in pending else 0


def _inprogress_by_nickname(nick):
    #해당 닉네임의 진행 중(대기/분석) 작업 목록 (메모리 기준)
    rows = []
    with jobs_lock:
        pending = [j for j, v in jobs.items() if v["status"] == "pending"]
        for jid, v in jobs.items():
            if v.get("nickname") == nick and v["status"] in ("pending", "running"):
                if v["status"] == "running":
                    st = "분석 중…"
                else:
                    a = pending.index(jid) if jid in pending else 0
                    st = f"대기 중 (앞에 {a}명)" if a > 0 else "곧 시작"
                rows.append(dict(job_id=jid, filename=v.get("filename", ""), status_text=st))
    return rows


# ── 템플릿 ──
_CSS = """body{font-family:'Malgun Gothic',Arial,sans-serif;max-width:760px;margin:0 auto;padding:18px;color:#222}
.muted{color:#888;font-size:13px}.btn{display:inline-block;margin:8px 6px 0 0;padding:10px 16px;background:#1565c0;color:#fff;border-radius:8px;text-decoration:none;border:none;cursor:pointer;font-size:15px}
.btn2{background:#2e7d32}.box{border-radius:10px;padding:14px 16px;margin:14px 0;line-height:1.6}
.good{background:#e8f5e9;border-left:6px solid #2e7d32}.mid{background:#fff3e0;border-left:6px solid #ef6c00}.low{background:#ffebee;border-left:6px solid #c62828}
.score{font-size:42px;font-weight:800;color:#1565c0;margin:2px 0 10px}.card{border:1px solid #ddd;border-radius:12px;padding:20px;margin-top:14px}
input[type=text]{padding:8px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:center}th{background:#f5f5f5}.hl{background:#e7f5e7;font-weight:700}
details{margin-top:10px}summary{cursor:pointer;font-weight:700;padding:8px}video{width:100%;max-width:480px;border-radius:8px;margin-top:8px}"""

UPLOAD_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>클라이밍 자세 분석</title>
<style>""" + _CSS + """ body{text-align:center}.err{color:#c62828}</style></head><body>
<h2>🧗 클라이밍 자세 분석</h2>

<details {{ 'open' if open_guide else '' }}>
  <summary>📋 업로드 안내 (예시 보기)</summary>
  <div class="box good" style="text-align:left">
    <ul style="margin:0;padding-left:20px">{% for g in guide %}<li>{{ g }}</li>{% endfor %}</ul>
    {% if has_example %}<div style="text-align:center"><video controls src="/example.mp4"></video><div class="muted">예시 영상 (T00.mp4)</div></div>{% endif %}
  </div>
</details>

<div class="card">
  <h3 style="margin-top:0">새 영상 분석</h3>
  <form method="post" action="/analyze" enctype="multipart/form-data"
        onsubmit="this.querySelector('button').innerText='업로드 중…';this.querySelector('button').disabled=true;">
    <input type="text" name="nickname" placeholder="닉네임(내 리포트 모아보기용)"><br>
    <input type="file" name="video" accept="video/*" required style="margin:12px 0"><br>
    <button class="btn" type="submit">분석하기</button>
  </form>
  {% if error %}<p class="err">{{ error }}</p>{% endif %}
</div>

<div class="card">
  <h3 style="margin-top:0">내 리포트 보기</h3>
  <form method="get" action="/my">
    <input type="text" name="nickname" placeholder="닉네임 입력" required>
    <button class="btn btn2" type="submit">리포트 보기</button>
  </form>
  <p class="muted">분석하지 않아도, 닉네임만 입력하면 그동안의 리포트와 현재 분석 진행상황을 볼 수 있습니다.</p>
</div>
</body></html>
"""

WAIT_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="4"><title>분석 중…</title>
<style>""" + _CSS + """ body{text-align:center;padding-top:40px}.spin{font-size:40px;animation:p 1.2s linear infinite}@keyframes p{to{transform:rotate(360deg)}}</style></head><body>
<div class="spin">⏳</div><h2>분석 중입니다…</h2><p>{{ wait_text }}</p>
<p class="muted">4초마다 자동 새로고침됩니다. 기다리는 동안 다른 리포트를 봐도 됩니다.</p>
<div class="box good">📌 이 페이지 주소를 저장해두면 나갔다 와도 결과를 다시 볼 수 있습니다. 하단의 "리포트 보기"를 눌러 분석이 완료된 영상들의 결과를 볼 수 있습니다.</div>
{% if nickname %}<a class="btn btn2" href="/my?nickname={{ nickname|urlencode }}">📁 내 리포트 보기</a>{% endif %}
<a class="btn" href="/">홈으로</a>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>분석 결과</title>
<style>""" + _CSS + """</style></head><body>
<h2>클라이밍 자세 분석 리포트</h2>
<p class="muted">{% if nickname %}{{ nickname }} · {% endif %}영상: {{ name }} · ⚠ 학습에 쓴 데이터 영상 수가 적어 결과가 정확하지 않을 수 있습니다.</p>
{% if error %}<div class="box low">{{ error }}</div>
{% else %}
  <div>전문가 유사도</div><div class="score">{{ '%.0f'|format(result.score) }}%</div>
  {% set sa = result.straight_arm %}
  <div class="box {{ 'good' if sa.status=='GOOD' else ('mid' if sa.status=='MID' else 'low') }}">
    <b>핵심 피드백 · 쉴 때 직선팔 (검증된 지표)</b><br>
    유사도 {{ '%.0f'|format(sa.progress) }}% &nbsp;(일반인평균 0% ~ 전문가평균 100%)<br>▶ {{ sa.message }}</div>
  <h3>참고 지표</h3>
  <p class="muted">데이터 영상을 수작업으로 모으는 한계로 미검증된 요소들이 많으며, 이는 피드백에 사용되지 않습니다.
  학습에 사용된 전문가 영상 데이터셋은 리드의 비중이 높으며, 일반인 영상 데이터셋은 리드의 비중이 낮습니다.</p>
  <table><tr><th>지표</th><th>내 값</th><th>전문가</th><th>일반인</th><th>구분</th></tr>
  {% for r in result.table %}<tr class="{{ 'hl' if r.tag=='검증됨' else '' }}">
    <td style="text-align:left">{{ r.name }}</td><td>{{ '%.3f'|format(r.value) }}</td>
    <td>{{ '%.3f'|format(r.exp) }}</td><td>{{ '%.3f'|format(r.non) }}</td><td>{{ r.tag }}</td></tr>{% endfor %}</table>
  <p class="muted">※ 검증된 지표는 '쉴 때 직선팔' 하나입니다. 나머지는 참고용이며, 전문가 분포와의 '차이'일 뿐 절대적 정답은 아닙니다.</p>
  {% if survey_url %}<a class="btn btn2" href="{{ survey_url }}" target="_blank">📝 설문 참여하기</a>{% endif %}
  {% if email_enabled %}
  <form method="post" action="/email_report" style="margin-top:12px">
    <input type="hidden" name="job_id" value="{{ job_id }}">
    <input type="email" name="email" placeholder="이메일로 리포트 받기(서버 꺼져도 보관)" required style="padding:8px;width:55%">
    <button class="btn" type="submit">📧 전송</button>
  </form>{% endif %}
{% endif %}
{% if nickname %}<a class="btn" href="/my?nickname={{ nickname|urlencode }}">📁 {{ nickname }} 님 리포트 모아보기</a>{% endif %}
<a class="btn" href="/">다른 영상 분석</a>
</body></html>
"""

MY_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{% if inprogress %}<meta http-equiv="refresh" content="5">{% endif %}
<title>{{ nickname }} 님 리포트</title>
<style>""" + _CSS + """</style></head><body>
<h2>📁 {{ nickname }} 님의 리포트</h2>

{% if inprogress %}
<h3>진행 중 <span class="muted">(5초마다 자동 새로고침)</span></h3>
<table><tr><th>영상</th><th>상태</th><th>보기</th></tr>
{% for d in inprogress %}<tr>
  <td style="text-align:left">{{ d.filename }}</td><td>{{ d.status_text }}</td>
  <td><a href="/result/{{ d.job_id }}">진행 보기</a></td></tr>{% endfor %}</table>
{% endif %}

<h3>완료된 리포트</h3>
{% if not reports %}<p class="muted">아직 완료된 리포트가 없습니다.</p>{% endif %}
{% if reports %}<table><tr><th>날짜</th><th>영상</th><th>전문가 유사도</th><th>보기</th></tr>
{% for d in reports %}<tr>
  <td>{{ d.saved_at }}</td><td style="text-align:left">{{ d.filename }}</td>
  <td>{{ '%.0f'|format(d.result.score) }}%</td>
  <td><a href="/result/{{ d.job_id }}">열기</a></td></tr>{% endfor %}</table>{% endif %}
<a class="btn" href="/">새 영상 분석</a>
</body></html>
"""

SURVEY_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>설문</title>
<style>""" + _CSS + """ .q{margin:18px 0}label{display:block;margin:6px 0}textarea{width:100%;height:80px}.req{color:#c62828}</style></head><body>
<h2>📝 Beta Finder 설문</h2>
<p class="muted">분석 결과가 도움이 되었는지 알려주세요. (Gmail 필수)</p>
<form method="post" action="/survey">
  <input type="hidden" name="job_id" value="{{ job_id }}">
  <div class="q"><b>Gmail 주소 <span class="req">*</span></b><br>
    <input type="email" name="gmail" placeholder="you@gmail.com" required
           pattern="[A-Za-z0-9._%+\\-]+@gmail\\.com" title="@gmail.com 주소만 입력하세요" style="width:70%"></div>
  <div class="q"><b>1. 본 프로그램에 전반적으로 만족하십니까? <span class="req">*</span></b>
    <label><input type="radio" name="q1" value="네" required> 네</label>
    <label><input type="radio" name="q1" value="아니오"> 아니오</label></div>
  <div class="q"><b>2. 본인의 클라이밍 약점 파악에 도움이 되셨습니까? <span class="req">*</span></b>
    <label><input type="radio" name="q2" value="네" required> 네</label>
    <label><input type="radio" name="q2" value="아니오"> 아니오</label></div>
  <div class="q"><b>3. 이외에 아쉬웠던 점 (선택)</b><br><textarea name="q3"></textarea></div>
  {% if error %}<p class="req">{{ error }}</p>{% endif %}
  <button class="btn btn2" type="submit">제출</button>
</form></body></html>
"""

THANKS_HTML = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>감사합니다</title>
<style>""" + _CSS + """ body{text-align:center;padding-top:50px}</style></head><body>
<h2>🙏 설문 참여 감사합니다!</h2><p class="muted">소중한 의견이 저장되었습니다.</p>
<a class="btn" href="/">홈으로</a></body></html>
"""


@app.route("/")
def index():
    return render_template_string(UPLOAD_HTML, error=None, guide=GUIDE_LINES,
                                  has_example=os.path.exists(EXAMPLE_VIDEO), open_guide=False)


@app.route("/example.mp4")
def example_video():
    if not os.path.exists(EXAMPLE_VIDEO):
        abort(404)
    return send_file(EXAMPLE_VIDEO, mimetype="video/mp4", conditional=True)


@app.route("/analyze", methods=["POST"])
def do_analyze():
    file = request.files.get("video")
    if not file or file.filename == "":
        return render_template_string(UPLOAD_HTML, error="영상을 선택하세요.", guide=GUIDE_LINES,
                                      has_example=os.path.exists(EXAMPLE_VIDEO), open_guide=True)
    job_id = uuid.uuid4().hex[:8]
    nickname = (request.form.get("nickname") or "").strip()
    path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    file.save(path)
    with jobs_lock:
        jobs[job_id] = dict(status="pending", filename=file.filename,
                            nickname=nickname, result=None, error=None)
    job_queue.put((job_id, path))
    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        job = _load_result(job_id)
    if not job:
        return "결과를 찾을 수 없습니다.", 404
    if job.get("status") in ("pending", "running"):
        if job["status"] == "running":
            wait_text = "분석하는 중…"
        else:
            ahead = _ahead_count(job_id)
            wait_text = (f"대기 중 (앞에 {ahead}명)" if ahead > 0 else "곧 분석을 시작합니다…")
        return render_template_string(WAIT_HTML, wait_text=wait_text, nickname=job.get("nickname"))
    return render_template_string(RESULT_HTML, job_id=job_id, name=job["filename"],
                                  nickname=job.get("nickname"), error=job.get("error"),
                                  result=job.get("result"), email_enabled=bool(SENDER_GMAIL),
                                  survey_url=SURVEY_FORM_URL)


@app.route("/my")
@app.route("/my/<nickname>")
def my_reports(nickname=None):
    if nickname is None:
        nickname = (request.args.get("nickname") or "").strip()
    if not nickname:
        return redirect(url_for("index"))
    return render_template_string(MY_HTML, nickname=nickname,
                                  reports=_reports_by_nickname(nickname),
                                  inprogress=_inprogress_by_nickname(nickname))


@app.route("/survey/<job_id>")
def survey_form(job_id):
    return render_template_string(SURVEY_HTML, job_id=job_id, error=None)


# 미사용, 구글 폼 사용
@app.route("/survey", methods=["POST"])
def survey_submit():
    job_id = request.form.get("job_id", "")
    gmail = (request.form.get("gmail") or "").strip()
    q1 = request.form.get("q1", "")
    q2 = request.form.get("q2", "")
    q3 = (request.form.get("q3") or "").strip()
    # 서버 검증: gmail 형식 + 필수 1·2
    if not GMAIL_RE.match(gmail) or q1 not in ("네", "아니오") or q2 not in ("네", "아니오"):
        return render_template_string(SURVEY_HTML, job_id=job_id,
                                      error="Gmail(@gmail.com)과 1·2번 항목을 올바르게 입력하세요.")
    new = not os.path.exists(SURVEY_CSV)
    with open(SURVEY_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["시각", "job_id", "gmail", "만족", "약점파악도움", "아쉬운점"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), job_id, gmail, q1, q2, q3])
    return render_template_string(THANKS_HTML)


@app.route("/email_report", methods=["POST"])
def email_report():
    job_id = request.form.get("job_id", "")
    to_addr = (request.form.get("email") or "").strip()
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        job = _load_result(job_id)
    if not job or job.get("status") != "done":
        return "리포트를 찾을 수 없습니다.", 404
    if not EMAIL_RE.match(to_addr):
        return "이메일 주소 형식이 올바르지 않습니다.", 400
    try:
        send_report_email(to_addr, job)
    except Exception as e:
        return render_template_string(
            "<meta charset='utf-8'><p style='font-family:Malgun Gothic'>전송 실패: {{e}}<br>"
            "<a href='/result/{{j}}'>돌아가기</a></p>", e=str(e), j=job_id), 500
    return render_template_string(
        "<meta charset='utf-8'><div style='font-family:Malgun Gothic;text-align:center;padding-top:40px'>"
        "<h2>📧 전송 완료!</h2><p>{{to}} 로 리포트를 보냈습니다.</p>"
        "<a href='/'>홈으로</a></div>", to=to_addr)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
