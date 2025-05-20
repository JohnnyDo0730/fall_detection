from flask import Flask, Response
import cv2
import mediapipe as mp
import joblib
import numpy as np

# 初始化 Flask
app = Flask(__name__)

# 初始化攝影機
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 載入模型與標準化器
rf = joblib.load("../RFC/rf_fall_model.pkl")
scaler = joblib.load("../RFC/rf_scaler.pkl")

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 每幀處理與推論
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        status_text = "Unknown"
        color = (128, 128, 128)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(keypoints) == 132:
                X_input = scaler.transform([keypoints])
                result = rf.predict(X_input)
                if result[0] == 1:
                    status_text = "FALLEN"
                    color = (0, 0, 255)
                else:
                    status_text = "NOT FALLEN"
                    color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 傳給瀏覽器
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 網頁首頁
@app.route('/')
def index():
    return '''
        <html><body>
        <h1>即時跌倒偵測 (隨機森林)</h1>
        <img src="/video_feed">
        </body></html>
    '''

# 串流路由
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 執行伺服器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 