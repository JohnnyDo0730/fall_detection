# 跌倒偵測系統 (Fall Detection System)

本專案實現了一個基於機器學習的即時跌倒偵測系統，使用MediaPipe骨架偵測與支援向量機(SVM)或隨機森林(Random Forest)分類器來判斷人體姿態是否處於跌倒狀態。系統可部署於樹莓派上，並透過網頁介面提供即時視訊串流與跌倒狀態顯示。

## 專案結構

```
fall_detection/
├── README.md                    # 專案說明文件
├── mediapipelabel.py            # 處理訓練資料集並產生特徵CSV檔案
├── pose_train_data.csv          # 經過處理的姿勢特徵資料集
├── fall_dataset/                # 跌倒影像資料集目錄
├── raspberry pi/                # 樹莓派部署程式
│   ├── camera_stream_svm.py     # 使用SVM模型的即時偵測程式
│   └── camera_stream_rf.py      # 使用Random Forest模型的即時偵測程式
├── svm/                         # SVM模型相關檔案
│   ├── fall_detect_svm.py       # SVM模型訓練程式
│   ├── svm_fall_model.pkl       # 訓練完成的SVM模型
│   └── svm_scaler.pkl           # SVM資料標準化器
└── RFC/                         # Random Forest模型相關檔案
    ├── fall_detect_rf.py        # Random Forest模型訓練程式
    ├── rf_fall_model.pkl        # 訓練完成的Random Forest模型
    └── rf_scaler.pkl            # Random Forest資料標準化器
```

## 系統需求

### 樹莓派端
- Raspberry Pi 4 (建議)
- 網路攝影機
- Python 3.7+
- 樹莓派作業系統

### 開發/訓練環境 (個人電腦)
- Python 3.7+
- 足夠的運算資源以訓練機器學習模型

## 安裝步驟

### 樹莓派安裝
```bash
# 更新套件資訊
sudo apt update

# 安裝必要套件
sudo apt install python3-opencv python3-flask python3-pip

# 安裝Python套件
pip3 install mediapipe --break-system-packages
pip3 install scikit-learn --break-system-packages
```

### 開發環境安裝 (個人電腦)
```bash
# 安裝必要Python套件
pip install mediapipe opencv-python scikit-learn matplotlib pandas
```

## 資料集準備

本專案使用Kaggle上的跌倒偵測資料集：
https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset

下載後，將資料集放入`fall_dataset`目錄中。

## 訓練模型

1. 首先，使用`mediapipelabel.py`處理資料集並產生特徵資料：
   ```bash
   python mediapipelabel.py
   ```
   這將產生`pose_train_data.csv`檔案，包含骨架關鍵點特徵。

2. 訓練SVM模型：
   ```bash
   cd svm
   python fall_detect_svm.py
   ```

3. 訓練Random Forest模型：
   ```bash
   cd RFC
   python fall_detect_rf.py
   ```

## 部署至樹莓派

1. 在樹莓派上使用Git Clone下載專案：
   ```bash
   # 安裝git (如果尚未安裝)
   sudo apt install git
   
   # 複製專案
   git clone https://github.com/JohnnyDo0730/fall_detection.git
   
   # 進入專案目錄
   cd fall_detection
   ```

2. 在樹莓派上設定相機解析度：
   ```bash
   # 編輯相機串流程式
   nano raspberry\ pi/camera_stream_svm.py
   # 或
   nano raspberry\ pi/camera_stream_rf.py
   ```

   確認以下設定符合您的相機：
   ```python
   camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

## 執行系統

在樹莓派上執行：

```bash
# 進入樹莓派程式目錄
cd raspberry\ pi/

# 使用SVM模型
python camera_stream_svm.py
# 或使用Random Forest模型
python camera_stream_rf.py
```

然後通過瀏覽器訪問：
```
http://[樹莓派IP]:5000
```

即可查看即時跌倒偵測畫面。系統會在畫面上顯示骨架追蹤並標示目前狀態：
- 綠色「NOT FALLEN」：表示正常姿態
- 紅色「FALLEN」：表示偵測到跌倒狀態

## 技術細節

- **MediaPipe**：用於即時骨架關鍵點偵測
- **SVM/Random Forest**：用於姿態分類
- **Flask**：提供網頁介面與視訊串流
- **OpenCV**：處理影像擷取與顯示

## 注意事項

- 相機解析度需設定為640x480以符合訓練資料
- 確保樹莓派與查看裝置在同一網路中
- 若系統效能不佳，可考慮降低影像解析度或幀率
- 專案原始碼可在[GitHub](https://github.com/JohnnyDo0730/fall_detection)上查看