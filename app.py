import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
import os

# Путь к файлу (конвертированный в .xlsx!)
DATA_PATH = 'aux/Sample - Superstore.xlsx'

st.title("Прогноз продаж категории Furniture (Superstore)")

# Проверка файла
if not os.path.exists(DATA_PATH):
    st.error(f"Файл не найден: {DATA_PATH}")
    st.stop()

# Кэшированная загрузка данных
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

df = load_data()

# Классы препроцессинга
class DataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, category='Furniture'):
        self.category = category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X[X['Category'] == self.category].copy()
        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data = data.set_index('Order Date')
        return data

class AddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags=12):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        data_monthly = X.resample('M').agg({'Sales': 'sum'})
        data_monthly.index.name = 'Order Date'
        data_monthly = data_monthly.reset_index()

        # Лаги
        for i in range(1, self.lags + 1):
            data_monthly[f'lag_{i}'] = data_monthly['Sales'].shift(i)

        # Роллинги
        for w in [3, 6, 12]:
            data_monthly[f'rolling_mean_{w}'] = data_monthly['Sales'].rolling(w).mean()
            data_monthly[f'rolling_std_{w}'] = data_monthly['Sales'].rolling(w).std()

        data_monthly['trend'] = np.arange(len(data_monthly))
        data_monthly['month'] = data_monthly['Order Date'].dt.month
        data_monthly['year'] = data_monthly['Order Date'].dt.year
        data_monthly['is_quarter_end'] = data_monthly['Order Date'].dt.day.isin([30, 31])
        data_monthly['is_holiday'] = data_monthly['Order Date'].dt.weekday >= 5

        data_monthly['diff_1'] = data_monthly['Sales'].diff(1)
        data_monthly['pct_change'] = data_monthly['Sales'].pct_change()
        data_monthly['max_6'] = data_monthly['Sales'].rolling(6).max()
        data_monthly['min_6'] = data_monthly['Sales'].rolling(6).min()
        data_monthly['range_6'] = data_monthly['max_6'] - data_monthly['min_6']

        data_monthly['month_encoded'] = data_monthly['month'].astype('category')
        data_monthly['year_encoded'] = data_monthly['year'].astype('category')

        # Статистики по lag_1
        for i in range(1, 2):
            col = f'lag_{i}'
            data_monthly[f'{col}_mean'] = data_monthly[col].rolling(3).mean()
            data_monthly[f'{col}_median'] = data_monthly[col].rolling(3).median()
            data_monthly[f'{col}_range'] = data_monthly[col].rolling(3).max() - data_monthly[col].rolling(3).min()

        data_monthly.dropna(inplace=True)
        data_monthly = data_monthly.set_index('Order Date')
        return data_monthly

# Обучение модели (всегда при запуске — надёжно)
with st.spinner("Обработка данных и обучение модели..."):
    preprocess_pipeline = Pipeline([
        ('preprocessing', DataPreprocessing(category='Furniture')),
        ('features', AddFeatures(lags=12))
    ])

    features_data = preprocess_pipeline.fit_transform(df)

    X = features_data.drop(columns=['Sales'])
    y = features_data['Sales']

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = ['month_encoded', 'year_encoded', 'is_quarter_end', 'is_holiday']
    cat_indices = list(range(len(numeric_features), len(numeric_features) + len(categorical_features)))

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features)
    ], remainder='passthrough')

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', CatBoostRegressor(
            iterations=2000,
            learning_rate=0.1,
            depth=1,
            random_seed=42,
            verbose=False,
            eval_metric='MAE',
            cat_features=cat_indices
        ))
    ])

    model_pipeline.fit(X, y)


# Предсказания на истории
predictions = model_pipeline.predict(X)
predictions_rounded = np.round(predictions).astype(int)

# График истории
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(features_data.index, features_data['Sales'], label='Фактические', marker='o')
ax1.plot(features_data.index, predictions_rounded, label='Предсказанные', marker='x', linestyle='--')
ax1.set_title('Продажи Furniture по месяцам (факт vs модель)')
ax1.set_ylabel('Продажи')
ax1.legend()
st.pyplot(fig1)

# Прогноз на будущее
st.header("Прогноз на будущие месяцы")
months_ahead = st.slider("Месяцев вперёд", 1, 36, 12)

last_date = features_data.index.max()
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=months_ahead, freq='M')

future_predictions = []
current_row = features_data.iloc[-1:].copy()

for i in range(months_ahead):
    X_future = current_row.drop(columns=['Sales'])
    pred = model_pipeline.predict(X_future)[0]
    future_predictions.append(pred)

    next_row = current_row.copy()
    next_row['Sales'] = pred

    # Сдвиг лагов
    for lag in range(12, 1, -1):
        next_row[f'lag_{lag}'] = current_row[f'lag_{lag-1}'].values[0]
    next_row[f'lag_1'] = pred

    # Обновление фич даты
    next_date = future_dates[i]
    next_row['month'] = next_date.month
    next_row['year'] = next_date.year
    next_row['month_encoded'] = pd.Categorical([next_date.month], categories=range(1, 13))
    next_row['year_encoded'] = pd.Categorical([next_date.year])
    next_row['is_quarter_end'] = next_date.day in [30, 31]
    next_row['is_holiday'] = next_date.weekday() >= 5
    next_row['trend'] = current_row['trend'].values[0] + 1

    current_row = next_row

future_predictions_rounded = np.round(future_predictions).astype(int)

# График прогноза
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(features_data.index, features_data['Sales'], label='Исторические', marker='o')
ax2.plot(future_dates, future_predictions_rounded, label='Прогноз', color='green', marker='s')
ax2.set_title(f'Прогноз продаж Furniture на {months_ahead} месяцев')
ax2.set_ylabel('Продажи')
ax2.legend()
st.pyplot(fig2)

# Таблица прогноза
forecast_df = pd.DataFrame({
    'Месяц': future_dates.strftime('%Y-%m'),
    'Прогноз продаж': future_predictions_rounded
})
forecast_df['Прогноз продаж'] = forecast_df['Прогноз продаж'].map('{:,}'.format)
st.table(forecast_df)