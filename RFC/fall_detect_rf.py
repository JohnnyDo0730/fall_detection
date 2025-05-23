# 匯入所需的套件
import pandas as pd  # 用於資料讀取與處理
from sklearn.model_selection import train_test_split  # 用於切分訓練與測試集
from sklearn.preprocessing import StandardScaler  # 用於資料標準化（z-score）
from sklearn.ensemble import RandomForestClassifier  # 隨機森林分類器
from sklearn.metrics import classification_report, confusion_matrix  # 評估指標
import joblib  # 用於模型與物件的儲存

# 讀取骨架特徵資料，資料格式為 CSV，每列包含 132 維關鍵點特徵與標籤
df = pd.read_csv("../pose_train_data.csv")

# 分離特徵（X）與標籤（y）
X = df.drop("label", axis=1)  # 除去標籤欄，留下特徵資料
y = df["label"]  # 取得標籤欄，0 表示未跌倒，1 表示跌倒

# 建立 StandardScaler 進行 z-score 標準化，使特徵具有相同的尺度
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 擬合並轉換資料

# 將資料分為訓練集與測試集，比例為 8:2，並設定隨機種子確保可重現性
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 建立並訓練隨機森林分類模型（使用 100 顆樹）
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = rf.predict(X_test)

# 輸出分類評估報告（包含 Accuracy, Precision, Recall, F1-score）
print("評估報告：")
print(classification_report(y_test, y_pred, target_names=["Not Fallen", "Fallen"]))

# 儲存訓練完成的模型與標準化器，以便部署使用
joblib.dump(rf, "rf_fall_model.pkl")        # 儲存模型
joblib.dump(scaler, "rf_scaler.pkl")        # 儲存 scaler（用於部署時標準化新資料）

print("模型與標準化已儲存")
