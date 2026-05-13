import pandas as pd
import joblib
import os

# (Блок с SEED здесь не обязателен для инференса,
# но если модель была обучена с фиксированным SEED, его можно оставить для консистентности)
SEED = 42

# 1. Проверка существования файла и папки с моделью
# Это предотвратит внезапное падение скрипта, если путь указан неверно.
if not os.path.exists('models/model.pkl'):
    raise FileNotFoundError("Файл модели models/model.pkl не найден. Сначала обучите модель.")

if not os.path.exists('data/valid_features.csv'):
    raise FileNotFoundError("Файл данных data/valid_features.csv не найден.")

# 2. Загрузка данных и модели
print("Загрузка данных и модели...")
valid = pd.read_csv('data/valid_features.csv', parse_dates=['METEOFORECASTHOUR_OPENM_Datetime'])
model = joblib.load('models/model.pkl')

# 3. Генерация признаков (ИСПРАВЛЕНО: используем правильное имя колонки)
# Важно: Используем то же самое имя, что и при обучении!
print("Генерация новых признаков...")
valid['hour'] = valid['METEOFORECASTHOUR_OPENM_Datetime'].dt.hour
valid['dayofweek'] = valid['METEOFORECASTHOUR_OPENM_Datetime'].dt.dayofweek

# 4. Подготовка данных для модели
# Важно: Порядок и названия признаков должны ТОЧНО совпадать с теми, что были при обучении.
features = ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m', 'wind_direction_10m',
            'wind_direction_80m', 'wind_direction_120m', 'wind_direction_180m', 'wind_gusts_10m', 'temperature_80m',
            'temperature_120m', 'pressure_msl', 'rain', 'showers', 'snowfall', 'cloud_cover_low', 'Кол-во_ВЭУ_в_ремонте', 'hour', 'dayofweek']
X_valid = valid[features]

# 5. Предсказание
print("Выполнение предсказания...")
predictions = model.predict(X_valid)

# 6. Сохранение результата
output_path = 'prediction.csv'
pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)

print(f"Готово! Файл с предсказаниями сохранен по пути: {os.path.abspath(output_path)}")