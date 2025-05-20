## 樹梅派安裝一些套件
sudo apt update  
sudo apt install python3-opencv python3-flask python3-pip  
pip3 install mediapipe --break-system-packages  
pip3 install scikit-learn --break-system-packages  
  
  
### 在自己的電腦訓練svm(不要裝在樹梅派上)
pip install mediapipe opencv-python scikit-learn matplotlib pandas  
  
#### 跌倒判斷訓練資料(kaggle)  
https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset?resource=download  

#### mediapipe
https://github.com/google-ai-edge/mediapipe?tab=readme-ov-file

在做mediapipe label標記的時候要先對圖片做預處理，改成跟相機拍下來同個解析度。
鏡頭調整成640*480。

nano camera_stream_svm.py
svm = joblib.load("./svm/svm_fall_model.pkl")
scaler = joblib.load("./svm/svm_scaler.pkl")

按ctrl+o儲存 ctrl+x離開nano編輯器

scp 本地資料位置 pi名稱@ip地址:/home/pi名稱/
