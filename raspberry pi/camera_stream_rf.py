# 匯入所需的函式庫
from flask import Flask, Response         # 用於建立網頁伺服器與串流回應
import cv2                                # OpenCV，用於影像擷取與處理
import mediapipe as mp                    # MediaPipe，用於骨架關鍵點偵測
import joblib                             # 用於載入事先訓練好的模型與 scaler
import numpy as np                        # 用於資料處理

# 初始化 Flask 應用
app = Flask(__name__)

# 初始化攝影機（通常為 /dev/video0）
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 載入先前訓練好的隨機森林模型與標準化器
rf = joblib.load("../RFC/rf_fall_model.pkl")    # 路徑請依實際位置調整
scaler = joblib.load("../RFC/rf_scaler.pkl")

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 定義生成即時影像串流的函式
def gen_frames():
    while True:
        success, frame = camera.read()  # 讀取一幀影像
        if not success:
            break

        # 將影像從 BGR 轉為 RGB（MediaPipe 使用 RGB）
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測骨架關鍵點
        results = pose.process(image_rgb)

        # 預設狀態文字與顏色
        status_text = "Unknown"
        color = (128, 128, 128)

        # 若有偵測到關鍵點，則進行推論
        if results.pose_landmarks:
            # 繪製關鍵點骨架連線圖
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 取得 33 個關鍵點，每個有 x, y, z, visibility，共 132 維特徵
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            # 確保輸入長度正確，再標準化與進行分類推論
            if len(keypoints) == 132:
                X_input = scaler.transform([keypoints])     # 標準化輸入資料
                result = rf.predict(X_input)                # 模型預測

                # 根據預測結果顯示狀態
                if result[0] == 1:
                    status_text = "FALLEN"
                    color = (0, 0, 255)  # 紅色警示
                else:
                    status_text = "NOT FALLEN"
                    color = (0, 255, 0)  # 綠色正常

        # 在畫面左上角顯示狀態文字
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 將處理後的影像編碼成 JPEG 格式以供瀏覽器串流
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 網頁首頁：顯示標題與串流影像
@app.route('/')
def index():
    return '''
        <html><body>
        <h1>即時跌倒偵測 (隨機森林)</h1>
        <img src="/video_feed">
        </body></html>
    '''

# 串流路由，提供即時影像串流內容
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 啟動 Flask 應用（0.0.0.0 可讓外部裝置連線觀看）
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
