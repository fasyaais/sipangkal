import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import os
import warnings
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
from datetime import timedelta

warnings.filterwarnings('ignore')

# --- Konfigurasi Aplikasi ---
# Pindahkan path ke variabel global agar mudah diubah
BASE_OUTPUT_DIR = "models"
DATA_PATH = "data/preprocessing-data.csv"
HOLIDAY_PATH = "data/holiday.csv"

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="API Prediksi Harga Komoditas",
    description="API untuk memprediksi harga komoditas di masa depan menggunakan model XGBoost.",
    version="1.0.0"
)

# Variabel global untuk menyimpan data yang sudah dimuat
# Ini untuk efisiensi, agar tidak membaca file CSV setiap kali ada request
df_main = pd.DataFrame()
df_holidays = pd.DataFrame()

# --- Event Handler untuk Memuat Model saat Startup ---
@app.on_event("startup")
def load_data():
    """Memuat data CSV ke dalam memori saat server FastAPI pertama kali dijalankan."""
    global df_main, df_holidays
    print("Mulai memuat data historis dan hari libur...")
    try:
        df_main = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df_holidays = pd.read_csv(HOLIDAY_PATH, parse_dates=["date"])
        # Pastikan data diurutkan
        df_main = df_main.sort_values(['commodity', 'date']).reset_index(drop=True)
        print("✅ Data berhasil dimuat ke dalam memori.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: File data tidak ditemukan. Pastikan path sudah benar. Error: {e}")
        # Hentikan aplikasi jika data tidak bisa dimuat
        raise RuntimeError("Gagal memuat file data penting.") from e


# --- Model Pydantic untuk Struktur Response ---
class PredictionItem(BaseModel):
    date: str
    predicted_price: float

class PredictionResponse(BaseModel):
    commodity: str
    predictions: List[PredictionItem]
    
# --- FUNGSI FEATURE ENGINEERING (Disalin dari skrip asli) ---
def create_features_improved(df):
    sub_df = df.copy()
    sub_df['day'] = sub_df['date'].dt.day
    sub_df['month'] = sub_df['date'].dt.month
    sub_df['year'] = sub_df['date'].dt.year
    sub_df['dayofweek'] = sub_df['date'].dt.dayofweek
    sub_df['dayofyear'] = sub_df['date'].dt.dayofyear
    sub_df['weekofyear'] = sub_df['date'].dt.isocalendar().week.astype(int)
    sub_df['quarter'] = sub_df['date'].dt.quarter
    sub_df['is_weekend'] = (sub_df['dayofweek'] >= 5).astype(int)
    for lag in [1, 2, 3, 7, 14, 30]:
        sub_df[f'price_lag_{lag}'] = sub_df['price'].shift(lag)
    for window in [3, 7, 14, 30]:
        sub_df[f'rolling_mean_{window}'] = sub_df['price'].rolling(window=window).mean()
        sub_df[f'rolling_std_{window}'] = sub_df['price'].rolling(window=window).std()
    sub_df['price_diff_1'] = sub_df['price'].diff(1)
    return sub_df

# --- FUNGSI PREDIKSI (Disalin dan sedikit dimodifikasi) ---
def predict_future_prices(
    commodity_name: str,
    future_dates: pd.Series,
    historical_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    base_output_dir: str
):
    safe_commodity_name = commodity_name.replace(' ', '_').replace('/', '_')
    model_dir = os.path.join(base_output_dir, safe_commodity_name)
    model_path = os.path.join(model_dir, "model.json")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    
    if not os.path.exists(model_path):
        # Alih-alih mengembalikan None, kita akan raise exception yang akan ditangkap oleh FastAPI
        raise HTTPException(status_code=404, detail=f"Model untuk komoditas '{commodity_name}' tidak ditemukan.")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        
    last_known_data = historical_df[historical_df['commodity'] == commodity_name].tail(60).copy()
    predictions = []

    for future_date in sorted(future_dates):
        new_row = pd.DataFrame([{'date': future_date, 'price': np.nan, 'commodity': commodity_name}])
        temp_df = pd.concat([last_known_data, new_row], ignore_index=True)
        cols_to_drop = ['holiday_name', 'holiday_encoded', 'is_holiday']
        temp_df = temp_df.drop(columns=cols_to_drop, errors='ignore')
        df_with_features = create_features_improved(temp_df)
        df_with_features = df_with_features.merge(holidays_df, on='date', how='left')
        df_with_features['holiday_name'] = df_with_features['holiday_name'].fillna('No Holiday')
        df_with_features['holiday_encoded'] = le.transform(df_with_features['holiday_name'])
        df_with_features['is_holiday'] = df_with_features['holiday_name'].apply(lambda x: 0 if x == 'No Holiday' else 1)
        prediction_input = df_with_features.tail(1)
        feature_cols = model.get_booster().feature_names
        X_pred = prediction_input[feature_cols]
        # Pastikan prediksi adalah float standar python
        predicted_price = float(model.predict(X_pred)[0])
        predictions.append(predicted_price)
        update_row = prediction_input.copy()
        update_row.loc[:, 'price'] = predicted_price
        last_known_data = pd.concat([last_known_data, update_row], ignore_index=True)
        
    result_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
    return result_df

# --- ENDPOINT API ---
@app.get("/predict", response_model=PredictionResponse, tags=["Predictions"])
def get_prediction(
    commodity_name: str = Query(..., description="Nama komoditas yang ingin diprediksi. Contoh: `CABE MERAH Besar`"),
    days_to_predict: int = Query(7, description="Jumlah hari ke depan yang ingin diprediksi.", ge=1, le=30)
):
    """
    Endpoint untuk mendapatkan prediksi harga suatu komoditas untuk beberapa hari ke depan.
    """
    # Validasi apakah komoditas ada di data
    if commodity_name not in df_main['commodity'].unique():
        raise HTTPException(status_code=404, detail=f"Komoditas '{commodity_name}' tidak ditemukan dalam dataset.")

    last_date = df_main[df_main['commodity'] == commodity_name]['date'].max()
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)])

    predictions_df = predict_future_prices(
        commodity_name=commodity_name,
        future_dates=future_dates,
        historical_df=df_main,
        holidays_df=df_holidays,
        base_output_dir=BASE_OUTPUT_DIR
    )

    predictions_df['date'] = predictions_df['date'].dt.strftime('%Y-%m-%d')
    prediction_list = predictions_df.to_dict(orient='records')
    
    return {
        "commodity": commodity_name,
        "predictions": prediction_list
    }

@app.get("/commodities", tags=["Information"])
def get_available_commodities() -> Dict[str, List[str]]:
    """
    Endpoint untuk mendapatkan daftar semua komoditas yang tersedia untuk prediksi.
    """
    commodities = df_main['commodity'].unique().tolist()
    return {"available_commodities": commodities}

if __name__ == "__main__":
    import uvicorn
    # Perintah ini memberitahu uvicorn untuk menjalankan aplikasi 'app'
    # yang ada di dalam file ini ('main.py')
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)