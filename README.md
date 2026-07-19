# Прогнозування рівня стресу студентів за допомогою машинного навчання

## Опис проєкту

Проєкт розроблено для прогнозування рівня стресу студентів за допомогою методів машинного навчання.

Система аналізує соціально-демографічні та академічні характеристики студента, а також відповіді на запитання психометричних шкал:

- **PSS-10** — шкала сприйнятого стресу;
- **GAD-7** — шкала тривожності;
- **PHQ-9** — шкала депресивних симптомів.

Результатом роботи моделі є прогноз одного з трьох рівнів стресу:

- Low Stress — низький рівень стресу;
- Moderate Stress — помірний рівень стресу;
- High Perceived Stress — високий рівень сприйнятого стресу.

Для взаємодії з моделлю розроблено вебзастосунок на базі **Streamlit**, у якому користувач може заповнити анкету та отримати прогнозований рівень стресу.


## Використані технології

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Pickle
- Matplotlib / Seaborn
- XGBoost
- PyTorch

## Машинне навчання

У межах дослідження було протестовано декілька алгоритмів машинного навчання, зокрема:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Gaussian Naive Bayes
- Feedforward Neural Network

Для підготовки даних використовувалися кодування категоріальних ознак, стандартизація числових даних та метод SMOTE для балансування класів. Налаштування гіперпараметрів моделей виконувалося за допомогою GridSearchCV та крос-валідації.

За результатами порівняння моделей для вебзастосунку було обрано модель **Logistic Regression**, яка продемонструвала найкращі результати на тестових даних.

## Запуск проєкту локально

1. Клонування репозиторію

git clone <https://github.com/erfvntr/Stress_Prediction.git>
cd project_diploma

2. Створення віртуального середовища

*  Windows:
python -m venv venv
venv\Scripts\activate

*  macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Встановлення залежностей

pip install -r requirements.txt

4. Запуск Streamlit-застосунку
streamlit run app.py або python -m streamlit run app.py

Після запуску застосунок буде доступний у браузері за локальною адресою, яку Streamlit покаже в терміналі. 

Даним веб-застосунком можна скористатись тут: 

## Використання
Заповніть загальну інформацію про студента.
Дайте відповіді на запитання шкал PSS-10, GAD-7 та PHQ-9.
Натисніть кнопку «Визначити рівень стресу».
Система сформує прогноз рівня стресу за допомогою навченої моделі машинного навчання.


## Структура проєкту

```text
Stress_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── setup.py
│
└── src/
    ├── __init__.py
    ├── exception.py
    │
    └── notebook_info/
        ├── Diploma_Qualification.ipynb
        ├── Processed.csv
        └── stress_logistic_model.pkl

Основні файли:

*  app.py — Streamlit-застосунок для взаємодії з моделлю.
*  stress_logistic_model.pkl — збережена навчена модель машинного навчання.
*  Diploma_Qualification.ipynb — Jupyter Notebook з аналізом даних, підготовкою датасету, навчанням та оцінюванням моделей.
*  Processed.csv — підготовлений набір даних.
*  exception.py — модуль для обробки винятків.
*  requirements.txt — список необхідних Python-бібліотек.
*  setup.py — конфігурація Python-пакета.
