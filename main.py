import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor

# Загрузка данных
df = pd.read_csv("df_for_ML.csv")


# Вычисляем IQR
Q1 = df['Цена'].quantile(0.25)
Q3 = df['Цена'].quantile(0.75)
IQR = Q3 - Q1

# Определяем границы
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Заменяем выбросы на границы
df['Цена'] = df['Цена'].clip(lower=lower_bound, upper=upper_bound)


X = df.drop(columns=['Id', 'Цена'])
y = df['Цена']


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Сохранение имен признаков
feature_names = X_train.columns


# Масштабирование признаков
scaler = MinMaxScaler((-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Параметры моделей
params = {
    'max_depth': 3, 
    'learning_rate': 0.1,
    'n_estimators': 500,
}

rf_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

lgb_params = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'num_leaves': 31,   
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_child_samples': 30,
    'verbose': -1
}


# Модели
model = RandomForestRegressor(**rf_params)
xgb_model = xgb.XGBRegressor(**params)
lgb_model = LGBMRegressor(**lgb_params)


# Стекинг моделей
estimators = [
    ('rf', RandomForestRegressor(**rf_params)),
    ('xgb', xgb.XGBRegressor(**params)),
    ('lgb', LGBMRegressor(**lgb_params)),
]

stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Ансамбль моделей
ensemble = VotingRegressor(estimators=[
    ('rf', model),
    ('xgb', xgb_model),
    ('lgb', lgb_model),
])


# Обучение модели
ensemble.fit(X_train, y_train)
model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
stack.fit(X_train, y_train)



# Предсказания моделей
y_pred = model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
y_pred_stack = stack.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)


# Оценка качества моделей
r2_ensemble = r2_score(y_test, y_pred_ensemble)

# Результаты работы моделей
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
print(f'MAE Ensemble: {mae_ensemble:.4f}')
print(f'R^2: {r2_ensemble}')
print('')
mae_stack = mean_absolute_error(y_test, y_pred_stack)
print(f'MAE Stacking: {mae_stack:.4f}')
print('')

# Сохранение моделей
saved_model_ensemble = joblib.dump('ensemble_model.pkl', ensemble)
saved_scaler = joblib.dump('scaler.pkl', scaler)
saved_model_stack = joblib.dump('stack_model.pkl', stack)


# Гистограммы распределения признаков
plt.figure(figsize=(15, 5))

# Гистограмма распределения Километража
plt.subplot(1, 3, 1)
sns.histplot(df['Километраж'], kde=True)
plt.title('Распределение Километража')

# Гистограмма распределения Лошадиных сил
plt.subplot(1, 3, 2)
sns.histplot(df['Лошадиные силы'], kde=True)
plt.title('Распределение Лошадиных сил')

# Гистограмма распределения Разгона до 100
plt.subplot(1, 3, 3)
sns.histplot(df['Разгон до 100'], kde=True)
plt.title('Распределение Разгона до 100')

plt.tight_layout()
plt.show()

# Boxplot для признаков
plt.figure(figsize=(15, 5))

# Boxplot для Километража
plt.subplot(1, 3, 1)
sns.boxplot(x=df['Километраж'])
plt.title('Boxplot Километража')

# Boxplot для Лошадиных сил
plt.subplot(1, 3, 2)
sns.boxplot(x=df['Лошадиные силы'])
plt.title('Boxplot Лошадиных сил')

# Boxplot для Разгона до 100
plt.subplot(1, 3, 3)
sns.boxplot(x=df['Разгон до 100'])
plt.title('Boxplot Разгона до 100')

plt.tight_layout()
plt.show()

# Гистограмма распределения цен на автомобили
plt.figure(figsize=(10, 6))
sns.histplot(df['Цена'], bins=50, kde=True)
plt.title('Распределение цен на автомобили')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()

# График зависимости цены от года выпуска
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Год выпуска', y='Цена', data=df)
plt.title('Зависимость цены от года выпуска')
plt.xlabel('Год выпуска')
plt.ylabel('Цена')
plt.show()

# Тепловая карта корреляции числовых признаков
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция числовых признаков')
plt.show()

# Преобразование признаков класса автомобиля
df['Класс автомобиля'] = df[['Класс автомобиля_B', 'Класс автомобиля_C', 'Класс автомобиля_D', 
                             'Класс автомобиля_E', 'Класс автомобиля_F', 'Класс автомобиля_J', 
                             'Класс автомобиля_M']].idxmax(axis=1)

# Убираем префикс "Класс автомобиля_"
df['Класс автомобиля'] = df['Класс автомобиля'].str.replace('Класс автомобиля_', '')

# Boxplot зависимости цены от класса автомобиля
plt.figure(figsize=(12, 6))
sns.boxplot(x='Класс автомобиля', y='Цена', data=df)
plt.title('Зависимость цены от класса автомобиля')
plt.xlabel('Класс автомобиля')
plt.ylabel('Цена')
plt.xticks(rotation=45)
plt.show()
