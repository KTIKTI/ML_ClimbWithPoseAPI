# H0_HandmadeAPIDebug.py
# ──────────────────────────────────────────────────────────────────
# 목표: 수제 API모델이 잘 작동하는지 디버그 영상을 통해 눈으로 확인
#
# 코드:
#   클라이밍 영상 -> 프레임마다 YOLO crop -> 모델 → 3D 스켈레톤 -> mp4 저장
#   (M1_Rules.py와 다르게 모델 출력이 3D 미터좌표라 영상 위 오버레이 불가 -> 좌:입력크롭 / 우:예측 3D 스켈레톤)
#
# 사용: 같은 폴더에 my_model2_epoch_40.pth 두어야 함
# ──────────────────────────────────────────────────────────────────
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ultralytics import YOLO

# ── 설정 ──
MODEL_PT = "my_model2_epoch_40.pth"   # 학습 가중치 (ResNet 5층)
DEBUG_DIR = "DebugSingleVideo"        # 이 폴더 안의 모든 T*.mp4 처리
IMG_SIZE = 112
MARGIN   = 0.25
TARGET_FPS = 8       # 디버그 출력 fps(낮춰 빠르게)
MAX_FRAMES = 200     # 너무 길면 제한

# H36M 17관절 뼈대
H36M_BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


# ── 모델 정의 (load_model.ipynb 와 동일 구조) ──
class CnnBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride),
                                          nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=17):
        super().__init__()
        self.in_channels = 3
        self.layer1 = self.make_layer(block, num_blocks[0], 32, 1)
        self.layer2 = self.make_layer(block, num_blocks[1], 64, 2)
        self.layer3 = self.make_layer(block, num_blocks[2], 128, 2)
        self.layer4 = self.make_layer(block, num_blocks[3], 512, 2)
        self.layer5 = self.make_layer(block, num_blocks[4], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.fc = nn.Linear(self.in_channels, num_classes * 3)

    def make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x); out = self.layer2(out); out = self.layer3(out)
        out = self.layer4(out); out = self.layer5(out)
        out = self.avgpool(out); out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.view(out.size(0), 17, 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[장치] {device}")
model = ResNet(CnnBlock, [10, 10, 10, 10, 10])
model.load_state_dict(torch.load(MODEL_PT, map_location=device))
model.eval().to(device)
yolo = YOLO("yolov8n.pt")


def crop_person(frame_bgr):
    res = yolo(frame_bgr, classes=[0], conf=0.25, verbose=False)
    b = res[0].boxes
    if b is None or len(b) == 0:
        return None
    xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy()
    x1, y1, x2, y2 = xyxy[int(conf.argmax())]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * MARGIN; y1 -= bh * MARGIN; x2 += bw * MARGIN; y2 += bh * MARGIN
    h, w = frame_bgr.shape[:2]
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(w, x2)), int(min(h, y2))
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def predict(crop_bgr):
    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   # RGB, /255
    t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        out = model(t)
    j = out[0].cpu().numpy()
    return j - j[0:1]


def render(crop_bgr, joints, idx, total):
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    if crop_bgr is not None:
        ax1.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Model input (YOLO crop)"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if joints is not None:
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        ax2.scatter(x, y, z, c="red", s=15)
        for i, jj in H36M_BONES:
            ax2.plot([x[i], x[jj]], [y[i], y[jj]], [z[i], z[jj]], c="royalblue")
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    else:
        ax2.text2D(0.5, 0.5, "no person", transform=ax2.transAxes, ha="center")
    ax2.set_title(f"Predicted 3D pose  {idx}/{total}")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.view_init(elev=15, azim=-70)
    fig.tight_layout(); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, round(fps / TARGET_FPS))
    writer = None
    fi = done = 0
    while True:
        ret, frame = cap.read()
        if not ret or done >= MAX_FRAMES:
            break
        if fi % skip == 0:
            crop = crop_person(frame)
            joints = predict(crop) if crop is not None else None
            img = render(crop, joints, done + 1, MAX_FRAMES)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         TARGET_FPS, (w, h))
            writer.write(img)
            done += 1
            if done % 20 == 0:
                print(f"  {done} 프레임 렌더")
        fi += 1
    cap.release()
    if writer:
        writer.release()
    print(f"저장: {output_path}  ({done} 프레임)")

def main():
    videos = sorted(glob.glob(os.path.join(DEBUG_DIR, "T*.mp4")))
    if not videos:
        print(f"'{DEBUG_DIR}' 폴더에 T*.mp4가 없습니다.")
        return
    print(f"'{DEBUG_DIR}'의 영상 {len(videos)}개 디버그 시작")
    for video in videos:
        name = os.path.basename(video)
        output_path = os.path.join(DEBUG_DIR, f"Handmade_debug_{name}")
        print(f"\n[{name}] 처리 중…")
        process_video(video, output_path)


if __name__ == "__main__":
    main()