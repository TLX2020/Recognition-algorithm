import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# 初始化YOLO模型
model = YOLO('yolov8n.pt')

# 初始化MediaPipe
mp_pose = mp.solutions.pose.Pose()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    frame_with_edges = cv2.addWeighted(frame, 0.8, edge_colored, 0.5, 0)

    # YOLO目标检测
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

    # MediaPipe人体识别
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = mp_pose.process(rgb)
    hand_results = mp_hands.process(rgb)

    # 绘制YOLO检测框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame_with_edges, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制人体关键点
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_with_edges, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

    # 绘制手部关键点
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_with_edges, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

    # 伪深度图（白底黑线）
    depth = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.cvtColor(255 - depth, cv2.COLOR_GRAY2BGR)

    # 缩小伪深度图并嵌入主画面右下角
    h, w = frame_with_edges.shape[:2]
    small_depth = cv2.resize(depth_colored, (w // 4, h // 4))
    frame_with_edges[-h // 4 - 10:-10, -w // 4 - 10:-10] = small_depth

    # 显示
    cv2.imshow('YOLO+MediaPipe+Edge+Depth', frame_with_edges)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
