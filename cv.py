import cv2
import mediapipe as mp

# 初始化MediaPipe解决方案
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 初始化视频捕捉
cap = cv2.VideoCapture(0)

# 配置MediaPipe模型
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5) as pose, \
    mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7) as hands:

    # 跟踪器状态
    tracker = None
    tracking = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # 镜像翻转帧
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸跟踪/检测逻辑
        if tracking:
            track_success, bbox = tracker.update(frame)
            if track_success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                tracking = False
                tracker = None

        if not tracking:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # 姿态识别
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2)
            )

        # 手部识别
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 255), thickness=2)
                )

        # 显示状态信息
        cv2.putText(frame, f"Face: {'Tracking' if tracking else 'Detecting'}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Hands: {len(hand_results.multi_hand_landmarks or [])}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Multi Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
