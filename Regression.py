import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

def linear_regression(x, y):
    n = len(x)

    # Обчислюємо суми
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    # Обчислюємо оцінки параметрів
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    # Обчислюємо стандартні помилки оцінок параметрів
    y_hat = m * x + b
    se_m = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2) / np.sum((x - np.mean(x)) ** 2))
    se_b = se_m * np.sqrt(np.sum(x ** 2) / n)

    # Обчислюємо довірчі інтервали
    alpha = 0.05  # Рівень значущості
    t_critical = stats.t.ppf(1 - alpha / 2, n - 2)  # Критичне значення t-розподілу

    # Довірчий інтервал для нахилу (m)
    m_interval = (m - t_critical * se_m, m + t_critical * se_m)

    # Довірчий інтервал для перетину з віссю y (b)
    b_interval = (b - t_critical * se_b, b + t_critical * se_b)

    return m, b, se_m, se_b, m_interval, b_interval

def calculate_r_squared(x, y, m, b):
    y_hat = m * x + b
    mean_y = np.mean(y)
    numerator = np.sum((y - y_hat) ** 2)
    denominator = np.sum((y - mean_y) ** 2)
    r_squared = 1 - (numerator / denominator)
    return r_squared

def calculate_correlation_coefficient(x, y, m, b):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)
    sum_y_squared = np.sum(y ** 2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2))

    correlation_coefficient = numerator / denominator
    return correlation_coefficient

def calculate_statistical_significance(x, y, m, b):
    # Додаємо стовбець з одиницями для константи b
    X = sm.add_constant(x)

    # Створюємо модель лінійної регресії
    model = sm.OLS(y, X)

    # Виконуємо підгонку моделі
    results = model.fit()

    # Отримуємо значення F-статистики та p-значення для оцінки значущості моделі
    f_statistic = results.fvalue
    p_value_f = results.f_pvalue

    # Отримуємо значення t-статистики та p-значення для коефіцієнта кореляції
    t_statistic = results.tvalues[1]  # Індекс 1, оскільки індекс 0 - це константа b
    p_value_t = results.pvalues[1]

    return f_statistic, p_value_f, t_statistic, p_value_t

def calculate_prediction_interval(x, m, b, alpha=0.05):
    n = len(x)
    x_mean = np.mean(x)

    # Обчислюємо прогнозоване значення для кожного x
    y_hat = m * x + b

    # Оцінка зворотної матриці
    X = np.column_stack((np.ones_like(x), x))
    XTX_inv = np.linalg.inv(np.dot(X.T, X))

    # Діагональні елементи зворотної матриці для стандартних помилок
    diag_XTX_inv = np.diag(XTX_inv)

    # Стандартні помилки нахилу та перетину
    se_m = np.sqrt(diag_XTX_inv[1])  # Для нахилу (коефіцієнта x)
    se_b = np.sqrt(diag_XTX_inv[0])  # Для перетину (константи)

    # Обчислюємо критичне значення t-розподілу
    t_critical = stats.t.ppf(1 - alpha / 2, n - 2)

    # Обчислюємо стандартну помилку прогнозу
    se_y_hat = np.sqrt((se_m ** 2) * (x ** 2) + (se_b ** 2) + ((x_mean - np.mean(x)) ** 2))

    # Обчислюємо довірчий інтервал
    lower_bound = y_hat - t_critical * se_y_hat
    upper_bound = y_hat + t_critical * se_y_hat

    return lower_bound, upper_bound, se_y_hat

# Зчитуємо дані з Excel-файлу
df = pd.read_excel('Data.xlsx')

# Замість згенерованих даних, використовуємо зчитані з Excel
x = df['Index'].values
y = df['Accidents'].values

m, b, se_m, se_b, m_interval, b_interval = linear_regression(x, y)

print("Оцінка нахилу (m):", m)
print("Оцінка перетину з віссю y (b):", b)
print("Стандартна помилка оцінки нахилу:", se_m)
print("Стандартна помилка оцінки перетину з віссю y:", se_b)
print("Довірчий інтервал для нахилу (m):", m_interval)
print("Довірчий інтервал для перетину з віссю y (b):", b_interval)

r_squared = calculate_r_squared(x, y, m, b)
correlation_coefficient = calculate_correlation_coefficient(x, y, m, b)

print("Коефіцієнт детермінації (R^2):", r_squared)
print("Коефіцієнт кореляції (r):", correlation_coefficient)

# Виконаємо парну лінійну регресію
a, d = np.polyfit(x, y, 1)

f_statistic, p_value_f, t_statistic, p_value_t = calculate_statistical_significance(x, y, a, d)

print("F-статистика:", f_statistic)
print("p-значення для F-тесту:", p_value_f)
print("t-статистика для коефіцієнта кореляції:", t_statistic)
print("p-значення для t-тесту коефіцієнта кореляції:", p_value_t)

# Згенеруємо значення для лінії регресії
x_regression = np.linspace(min(x), max(x), 100)
y_regression = a * x_regression + d

# Побудова графіка
plt.scatter(x, y, label='Дані')
plt.plot(x_regression, y_regression, color='red', label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Парна лінійна регресія')
plt.legend()
plt.grid(True)
plt.show()

# Обчислюємо довірчий інтервал для прогнозованого середнього значення залежної змінної з надійністю 0.95
lower_bound, upper_bound, se_y_hat = calculate_prediction_interval(x, a, d)

print("Довірчий інтервал для прогнозованого середнього значення залежної змінної з надійністю 0.95:")
for i, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
    print(f"Для x={x[i]}, [{lb:.2f}, {ub:.2f}], SE_y_hat={se_y_hat[i]:.2f}")

# Побудова графіку
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Спостереження')
plt.plot(x, a * x + d, color='red', label='Лінійна регресія')
plt.fill_between(x, lower_bound, upper_bound, color='orange', alpha=0.3, label='Довірчий інтервал (95%)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Теоретична лінійна регресія та довірчий інтервал (95%)')
plt.legend()
plt.grid(True)
plt.show()
