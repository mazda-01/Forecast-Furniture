import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap

st.set_page_config(page_title="Универсальный прогноз временных рядов", layout="wide")
st.title("Универсальный прогнозатор временных рядов")

st.markdown("""
**Загрузите свой файл с временным рядом**  
Поддерживаемые форматы: CSV, Excel (.xlsx, .xls)  
Требования:
- Колонка с датой
- Колонка со значениями (продажи, количество и т.д.)
""")

uploaded_file = st.file_uploader("Загрузите файл", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Чтение файла
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Файл загружен успешно!")
        st.write("Первые строки данных:")
        st.dataframe(df.head())

        # Автоопределение колонок
        date_candidates = [col for col in df.columns if any(k in col.lower() for k in ['date', 'time', 'order'])]
        value_candidates = [col for col in df.columns if any(k in col.lower() for k in ['sales', 'value', 'amount', 'revenue', 'quantity'])]

        date_col = st.selectbox("Колонка с датой", options=date_candidates or df.columns)
        value_col = st.selectbox("Колонка со значениями", options=value_candidates or df.columns)

        if st.button("Запустить прогноз"):
            with st.spinner("Обработка данных и обучение модели..."):
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[[date_col, value_col]].sort_values(date_col)
                df = df.set_index(date_col)

                # Месячная агрегация
                monthly = df.resample('M')[value_col].sum()
                data = pd.DataFrame({'value': monthly})
                data = data.reset_index()
                data = data.rename(columns={date_col: 'date'})

                # Feature Engineering
                class FeatureEngineer(BaseEstimator, TransformerMixin):
                    def __init__(self, lags=12):
                        self.lags = lags

                    def fit(self, X, y=None):
                        return self

                    def transform(self, X):
                        df = X.copy()
                        for i in range(1, self.lags + 1):
                            df[f'lag_{i}'] = df['value'].shift(i)

                        for w in [3, 6, 12]:
                            df[f'rolling_mean_{w}'] = df['value'].rolling(w).mean()
                            df[f'rolling_std_{w}'] = df['value'].rolling(w).std()

                        df['trend'] = np.arange(len(df))
                        df['month'] = df['date'].dt.month
                        df['year'] = df['date'].dt.year
                        df['is_quarter_end'] = df['date'].dt.day.isin([30, 31])
                        df['is_holiday'] = df['date'].dt.weekday >= 5

                        df['month_encoded'] = df['month'].astype('category')
                        df['year_encoded'] = df['year'].astype('category')

                        df.dropna(inplace=True)
                        df = df.set_index('date')
                        return df

                fe = FeatureEngineer(lags=12)
                features_data = fe.transform(data)

                if len(features_data) == 0:
                    st.error("Недостаточно данных для обучения. Нужно минимум 13 месяцев исторических данных (из-за 12 лагов).")
                    st.stop()

                X = features_data.drop(columns=['value'])
                y = features_data['value']

                numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
                cat_cols = ['month_encoded', 'year_encoded', 'is_quarter_end', 'is_holiday']
                cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

                # Масштабирование
                scaler = StandardScaler()
                X_scaled = X.copy()
                X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

                # Обучение модели
                model = CatBoostRegressor(
                    iterations=1500,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    cat_features=cat_indices
                )
                model.fit(X_scaled, y)

                # Сохраняем в session_state
                st.session_state.features_data = features_data
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.numeric_cols = numeric_cols
                st.session_state.actual_rounded = np.round(y).astype(int)

                # Предсказания на истории
                predictions = model.predict(X_scaled)
                predictions_rounded = np.round(predictions).astype(int)

                mae = mean_absolute_error(st.session_state.actual_rounded, predictions_rounded)
                r2 = r2_score(y, predictions)

                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:,.0f}")
                col2.metric("R² Score", f"{r2:.4f}")

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(features_data.index, st.session_state.actual_rounded, label='Фактические', marker='o')
                ax.plot(features_data.index, predictions_rounded, label='Предсказанные', marker='x', linestyle='--')
                ax.set_title('Фактические vs Предсказанные (история)')
                ax.legend()
                st.pyplot(fig)

                # SHAP
                with st.expander("Интерпретация модели (важность признаков)"):
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_scaled)
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False)
                    st.pyplot(fig_shap)

                st.session_state.last_forecast_months = None

        # Прогноз на будущее (всегда доступен после запуска)
        if 'features_data' in st.session_state:
            st.header("Прогноз на будущее")
            months_ahead = st.slider("Месяцев вперёд", 1, 36, 12, key="forecast_slider")

            # Пересчитываем только при изменении количества месяцев
            if 'last_forecast_months' not in st.session_state or st.session_state.last_forecast_months != months_ahead:
                with st.spinner("Расчёт прогноза..."):
                    last_row = st.session_state.features_data.iloc[-1:].copy()
                    future_pred = []
                    current = last_row.copy()

                    future_dates = pd.date_range(st.session_state.features_data.index.max() + pd.DateOffset(months=1), periods=months_ahead, freq='M')

                    for i in range(months_ahead):
                        X_f = current.drop(columns=['value'])
                        X_f[st.session_state.numeric_cols] = st.session_state.scaler.transform(X_f[st.session_state.numeric_cols])
                        pred = st.session_state.model.predict(X_f)[0]
                        future_pred.append(pred)

                        next_row = current.copy()
                        next_row['value'] = pred

                        for lag in range(12, 1, -1):
                            next_row[f'lag_{lag}'] = current[f'lag_{lag-1}'].values[0]
                        next_row['lag_1'] = pred

                        next_date = future_dates[i]
                        next_row['month'] = next_date.month
                        next_row['year'] = next_date.year
                        next_row['month_encoded'] = pd.Categorical([next_date.month], categories=range(1, 13))
                        next_row['year_encoded'] = pd.Categorical([next_date.year])
                        next_row['is_quarter_end'] = next_date.day in [30, 31]
                        next_row['is_holiday'] = next_date.weekday() >= 5
                        next_row['trend'] = current['trend'].values[0] + 1

                        next_row = next_row.set_index(pd.Index([next_date]))
                        current = next_row

                    future_predictions_rounded = np.round(future_pred).astype(int)

                    st.session_state.future_predictions_rounded = future_predictions_rounded
                    st.session_state.future_dates = future_dates
                    st.session_state.last_forecast_months = months_ahead

            # Отображаем сохранённый прогноз (мгновенно)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(st.session_state.features_data.index, st.session_state.actual_rounded, label='Исторические')
            ax2.plot(st.session_state.future_dates, st.session_state.future_predictions_rounded, label='Прогноз', color='green', marker='s')
            ax2.set_title(f'Прогноз на {st.session_state.last_forecast_months} месяцев')
            ax2.legend()
            st.pyplot(fig2)

            forecast_df = pd.DataFrame({
                'Месяц': st.session_state.future_dates.strftime('%Y-%m'),
                'Прогноз': st.session_state.future_predictions_rounded
            })
            forecast_df['Прогноз'] = forecast_df['Прогноз'].map('{:,}'.format)
            st.table(forecast_df)

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.write("Проверьте формат файла и наличие нужных колонок.")
else:
    st.info("Загрузите файл, чтобы начать.")