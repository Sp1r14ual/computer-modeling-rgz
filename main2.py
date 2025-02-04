import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis


def recurrent_forecast(series, L, M):
    current_window = list(series[-L+1:])
    forecast = []
    for _ in range(M):
        next_val = np.mean(current_window[-5:])  # Простейший метод
        forecast.append(next_val)
        current_window = current_window[1:] + [next_val]
    return forecast


# Прогноз для data.csv
data_df = pd.read_csv('data.csv', parse_dates=[
                      'ds'], quotechar='"', decimal='.')
series_data = data_df['y'].values.astype(float)

# Параметры (настройте под ваши данные)
L_data, r_data, M_data = 30, 10, 30

# SSA декомпозиция
ssa_data = SingularSpectrumAnalysis(window_size=L_data, groups=[
                                    [i] for i in range(r_data)])
components_data = ssa_data.fit_transform(series_data.reshape(1, -1))
reconstructed_data = components_data.sum(axis=0).flatten()

# Прогноз
forecast_data = recurrent_forecast(reconstructed_data, L=L_data, M=M_data)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(data_df['ds'], series_data, label='Исходный ряд', marker='o')

# Генерация дат для прогноза
last_date = data_df['ds'].iloc[-1]
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=M_data
)

plt.plot(forecast_dates, forecast_data,
         label='Прогноз', marker='x', linestyle='--')
plt.title('Прогноз методом SSA')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data_forecast_plot.png', dpi=300)
plt.close()

# Сохранение прогноза
forecast_df = pd.DataFrame({
    'ds': forecast_dates,
    'y': forecast_data
})
forecast_df.to_csv('data_forecast.csv', index=False)
