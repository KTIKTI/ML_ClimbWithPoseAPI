# M0_MediaPipe.py
# ──────────────────────────────────────────────────────────────────
# 목표: MediaPipe 익히기
#
# 배경:
#   수제 API 제작 전 앞서 포즈 추정 라이브러리 MediaPipe 사용법에 익숙해지는 것이 목표.
#   - 구버전 mp.solutions 는 Python 3.13에서 막혀, 신버전 Tasks API (PoseLandmarker)로 작성.
#   - 분석 대신 실시간으로 관절 추출 상태 확인
#
# 코드:
#   내장 웹캠(0번 카메라) 영상을 받아 매 프레임마다
#   1) MediaPipe 33개 관절 추출 → 화면에 스켈레톤(점 + 뼈대선) 오버레이
#   2) 팔꿈치 각도(직선팔 여부)와 삼지점 지표를 실시간 텍스트로 표시
#   q 키로 종료.
#
# 비고:
#   - RunningMode.VIDEO + detect_for_video(timestamp_ms): 영상/스트림용 모드.
#     timestamp 는 단조 증가해야 하므로 frame_idx 로 계산한다.
#   - OpenCV는 BGR, MediaPipe는 RGB → cv2.cvtColor 변환 필수.
#   - 좌표는 정규화(0~1)로 나오므로 (w, h)를 곱해 픽셀로 변환해 그림.
#   - calc_angle / check_tripod / CONNECTIONS 를 여기서 처음 만든 뒤 M1_Rules.py 에서 활용.
#     (지표 정의는 '지표 공식.docx' 참고)
#
# ──────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calc_angle(a, b, c): #각도 계산 함수. 지표 공식.docx 참고
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8) #1e-8은 분모가 0 되는 것을 방지
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def check_tripod(landmarks): #삼지점 여부 판별 함수. 지표 공식.docx 참고
    lw = landmarks[15].x
    rw = landmarks[16].x
    la = landmarks[27].x
    ra = landmarks[28].x

    ankle_left = min(la, ra)
    ankle_right = max(la, ra)
    lw_in = ankle_left <= lw <= ankle_right
    rw_in = ankle_left <= rw <= ankle_right
    return lw_in and rw_in

CONNECTIONS = [
    (11, 13), (13, 15),  # 왼팔: 어깨-팔꿈치-손목
    (12, 14), (14, 16),  # 오른팔: 어깨-팔꿈치-손목
    (11, 12),            # 어깨-어깨
    (11, 23), (12, 24),  # 어깨-골반
    (23, 24),            # 골반-골반
    (23, 25), (25, 27),  # 왼쪽 다리
    (24, 26), (26, 28),  # 오른쪽 다리
]

def draw_pose(frame, landmark, h, w): #원본 화면에 선을 그려주는 함수
    points = {}
    for i, lm in enumerate(landmark):
        px, py = int(lm.x * w), int(lm.y * h)
        points[i] = (px, py)
        if lm.visibility > 0.5:
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

    for (a, b) in CONNECTIONS:
        if a in points and b in points:
            if landmark[a].visibility > 0.5 and landmark[b].visibility > 0.5:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions( #사람 감지 혹은 프레임마다 추적시 필요한 최소 신뢰도: 50%
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0) #0번 카메라. 노트북의 경우 보통 내장 웹캠을 이용하나 다른 외장 카메라들을 사용하고 싶다면 0을 1, 2 등으로 변경하면 됨
frame_idx = 0

while cap.isOpened(): #카메라가 켜졌을 때
    ret, frame = cap.read() #ret: 성공여부 Boolean, frame: 이미지(numpy배열로 받으며, 이 때는 BGR 컬러 형식임)
    if not ret: #프레임을 못 읽는다면 프로그램 종료
        break

    h, w = frame.shape[:2] #창 높이와 너비 저장. 디버깅용 텍스트 출력용

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#MediaPipe는 RGB 컬러 형식을 사용하기 때문에 형식 변환 필요
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(frame_idx * 1000 / 30)  # 30fps (ms 단위)
    frame_idx += 1

    result = landmarker.detect_for_video(mp_image, timestamp_ms)#포즈 감지

    if result.pose_landmarks: #사람이 감지되었을 때만 실행
        lm = result.pose_landmarks[0]#사람
        draw_pose(frame, lm, h, w)#원본 화면에 선을 그려주는 함수

        l_angle = calc_angle( # 왼팔 각도
            [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
        )
        r_angle = calc_angle( # 오른팔 각도
            [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
        )

        l_straight = l_angle >= 150 #좌측 직선팔 여부 확인
        r_straight = r_angle >= 150 #우측 직선팔 여부 확인
        tripod = check_tripod(lm) #삼지점 여부 확인

        #디버깅용 텍스트 출력용
        lines = [
            f"L Elbow: {l_angle:.1f} ({'Straight' if l_straight else 'Non-Straight'})",
            f"R Elbow: {r_angle:.1f} ({'Straight' if r_straight else 'Non-Straight'})",
            f"Tripod: {tripod}",
        ]
        for i, txt in enumerate(lines):
            y_pos = h - 20 - (len(lines) - 1 - i) * 35
            cv2.putText(frame, txt, (w - 420, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("CLimbWithBlazepose_Prototype", frame) #창 이름
    if cv2.waitKey(1) & 0xFF == ord('q'): #q 누르면 프로그램 종료
        break

#안전하게 프로그램 종료
cap.release()
cv2.destroyAllWindows()
landmarker.close()