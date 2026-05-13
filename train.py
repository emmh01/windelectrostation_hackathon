import random
import numpy as np
import os

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Загрузка данных
train = pd.read_csv('data/train_dataset.csv', parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])

# --- Вставьте эти строки ---
# print("--- Столбцы в train ---")
# print(train.columns)  # Выведет Index([...])

# Генерация признаков
for df in [train]:
    df['hour'] = df['METEOFORECASTHOUR_OPENM_Datetime'].dt.hour
    df['dayofweek'] = df['METEOFORECASTHOUR_OPENM_Datetime'].dt.dayofweek

# Признаки и целевая переменная
features = ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m', 'wind_direction_10m', 'wind_direction_80m',
            'wind_direction_120m', 'wind_direction_180m', 'wind_gusts_10m', 'temperature_80m', 'temperature_120m','pressure_msl',
            'rain', 'showers', 'snowfall', 'cloud_cover_low', 'Кол-во_ВЭУ_в_ремонте', 'hour', 'dayofweek']
target = 'Выработка. Результирующий расчет'

X_train = train[features]
y_train = train[target]

# Разделяем исходный train на train и validation (например, 80% на 20%)
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED)

# Подготовка данных для LightGBM (создание Dataset-объектов)
# Это нужно для эффективного хранения и работы с данными в LightGBM
lgb_train = lgb.Dataset(X_train_split, label=y_train_split)
lgb_valid = lgb.Dataset(X_valid_split, label=y_valid_split) # Создаем валидационный датасет

# Параметры модели
params = {
    'objective': 'regression',
    'metric': 'mae', # Метрика для валидации (Mean Absolute Error)

    # Параметры обучения (параметры для ранней остановки)
    'early_stopping_rounds': 50,

    'random_state': 42,
    'verbosity': -1 # Аналог verbose=False
}

# Обучение модели с использованием основного API
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid], # Наборы для валидации
)

# Сохранение модели
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')