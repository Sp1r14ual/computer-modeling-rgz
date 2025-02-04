import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis


def recurrent_forecast(series, L, M):
    forecast = []
    current_window = list(series[-L+1:])  # Последние L-1 точек
    for _ in range(M):
        # Простейший метод: среднее последних значений (замените на ARIMA/регрессию при необходимости)
        # Пример: среднее 10 последних точек
        next_val = np.mean(current_window[-10:])
        forecast.append(next_val)
        current_window = current_window[1:] + [next_val]
    return forecast


# Прогноз для ряда x
x_series = pd.read_csv('x.csv', header=None).values.flatten()
L_x, r_x, M_x = 365, 50, 100
ssa_x = SingularSpectrumAnalysis(window_size=L_x, groups=[
                                 [i] for i in range(r_x)])
components_x = ssa_x.fit_transform(x_series.reshape(1, -1))
reconstructed_x = components_x.sum(axis=0).flatten()
forecast_x = recurrent_forecast(reconstructed_x, L=L_x, M=M_x)

# Прогноз для ряда y
y_series = pd.read_csv('y.csv', header=None, decimal=',',
                       encoding='utf-8').values.flatten()
y_series = y_series[~np.isnan(y_series)]
L_y, r_y, M_y = 240, 5, 60
ssa_y = SingularSpectrumAnalysis(window_size=L_y, groups=[
                                 [i] for i in range(r_y)])
components_y = ssa_y.fit_transform(y_series.reshape(1, -1))
reconstructed_y = components_y.sum(axis=0).flatten()
forecast_y = recurrent_forecast(reconstructed_y, L=L_y, M=M_y)

# Сохранение результатов
np.savetxt('forecast_x_pyts.csv', forecast_x, delimiter=',')
np.savetxt('forecast_y_pyts.csv', forecast_y, delimiter=',')
