import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap

st.set_page_config(page_title="Универсальный прогноз временных рядов", layout="wide")
st.title("Универсальный прогнозатор временных рядов")

st.markdown("""
Загрузите файл с временным рядом (CSV или Excel).  
Требования к файлу:
- Колонка с датой (например: `date`, `Date`, `Order Date`, `timestamp` и т.д.)
- Колонка с значениями (например: `sales`, `Sales`, `value`, `amount`)
- Формат даты: любой распознаваемый pandas
""")

uploaded_file = st.file_uploader("Загрузите файл", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Автоматическое чтение в зависимости от типа файла
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Файл загружен успешно!")
        st.write("Первые строки данных:")
        st.dataframe(df.head())

        # Автоопределение колонок
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        value_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'value', 'amount', 'revenue', 'quantity'])]

        date_col = st.selectbox("Выберите колонку с датой", options=date_cols or df.columns)
        from sklearn.metrics import mean_absolute_error, r2_score
        value_col = st.selectbox("Выберите колонку со значениями", options=value_cols or df.columns)

        if st.button("Запустить прогноз"):
            with st.spinner("Обработка данных и обучение модели..."):
                # Подготовка данных
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[[date_col, value_col]].sort_values(date_col)
                df = df.set_index(date_col)

                # Месячная агрегация
                monthly = df.resample('M')[value_col].sum()
                data = pd.DataFrame({value_col: monthly})
                data = data.reset_index()
                data = data.rename(columns={date_col: 'date', value_col: 'value'})

                # Класс добавления фич
                class FeatureEngineer(BaseEstimator, TransformerMixin):
                    def __init__(self, lags=12):
                        self.lags = lags

                    def fit(self, X, y=None):
                        return self

                    def transform(self, X):
                        df = X.copy()
                        for i in range(1, self.lags + 1):
                            df[f'lag_{i}'] = df['value'].shift(i)

                        df['rolling_mean_3'] = df['value'].rolling(3).mean()
                        df['rolling_mean_6'] = df['value'].rolling(6).mean()
                        df['rolling_mean_12'] = df['value'].rolling(12).mean()
                        df['rolling_std_3'] = df['value'].rolling(3).std()
                        df['rolling_std_6'] = df['value'].rolling(6).std()

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

                X = features_data.drop(columns=['value'])
                y = features_data['value']

                numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
                cat_cols = ['month_encoded', 'year_encoded', 'is_quarter_end', 'is_holiday']
                cat_indices = [X.columns.get_loc(col) for col in cat_cols if col in X.columns]

                preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), numeric_cols)
                ], remainder='passthrough')

                model = CatBoostRegressor(
                    iterations=1500,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    cat_features=cat_indices
                )

                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                pipe.fit(X, y)
                predictions = pipe.predict(X)
                predictions_rounded = np.round(predictions).astype(int)
                actual_rounded = np.round(y).astype(int)

                # Метрики: только MAE и R²
                mae = mean_absolute_error(actual_rounded, predictions_rounded)
                r2 = r2_score(y, predictions)  # R² на исходных значениях

                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:,.0f}")
                col2.metric("R² Score", f"{r2:.4f}")

                # График на истории
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(features_data.index, actual_rounded, label='Фактические', marker='o')
                ax.plot(features_data.index, predictions_rounded, label='Предсказанные', marker='x', linestyle='--')
                ax.set_title('Фактические vs Предсказанные (на истории)')
                ax.legend()
                st.pyplot(fig)

                # SHAP интерпретация
                with st.expander("Интерпретация модели (SHAP)"):
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X)
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    st.pyplot(fig_shap)

                # Прогноз на будущее
                st.header("Прогноз на будущее")
                months_ahead = st.slider("Месяцев вперёд", 1, 36, 12)

                last_row = features_data.iloc[-1:].copy()
                future_pred = []
                current = last_row.copy()

                for _ in range(months_ahead):
                    X_f = current.drop(columns=['value'])
                    pred = pipe.predict(X_f)[0]
                    future_pred.append(pred)

                    next_row = current.copy()
                    next_row['value'] = pred
                    for lag in range(12, 1, -1):
                        next_row[f'lag_{lag}'] = current[f'lag_{lag-1}'].values[0]
                    next_row['lag_1'] = pred

                    next_date = current.index[0] + pd.DateOffset(months=1)
                    next_row['month'] = next_date.month
                    next_row['year'] = next_date.year
                    next_row['month_encoded'] = next_row['month'].astype('category')
                    next_row['year_encoded'] = next_row['year'].astype('category')
                    next_row['is_quarter_end'] = next_date.day in [30, 31]
                    next_row['is_holiday'] = next_date.weekday() >= 5
                    next_row['trend'] = current['trend'].values[0] + 1

                    next_row = next_row.set_index(pd.Index([next_date]))
                    current = next_row

                future_pred_rounded = np.round(future_pred).astype(int)
                future_dates = pd.date_range(features_data.index.max() + pd.DateOffset(months=1), periods=months_ahead, freq='M')

                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(features_data.index, actual_rounded, label='Исторические')
                ax2.plot(future_dates, future_pred_rounded, label='Прогноз', color='green', marker='s')
                ax2.set_title(f'Прогноз на {months_ahead} месяцев')
                ax2.legend()
                st.pyplot(fig2)

                forecast_df = pd.DataFrame({
                    'Месяц': future_dates.strftime('%Y-%m'),
                    'Прогноз': future_pred_rounded
                })
                forecast_df['Прогноз'] = forecast_df['Прогноз'].map('{:,}'.format)
                st.table(forecast_df)

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.write("Проверьте формат файла и колонки.")
else:
    st.info("Загрузите файл (CSV или Excel), чтобы начать.")