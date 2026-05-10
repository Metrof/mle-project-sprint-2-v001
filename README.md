Улучшение baseline-модели предсказания стоимости недвижимости
Описание проекта
Цель проекта — повышение точности предсказания стоимости недвижимости в Санкт-Петербурге. В ходе работы реализован полный цикл разработки: от разведочного анализа данных до автоматизированного подбора гиперпараметров и регистрации модели в Model Registry.

Технологии
Python, Pandas, NumPy, Matplotlib

Scikit-learn, CatBoost

AutoFeat, MLxtend (SFS)

Optuna

MLflow (Tracking & Registry)

Инструкция по установке и запуску
Клонируйте репозиторий:

Bash
git clone https://github.com/Metrof/mle-project-sprint-2-v001.git
cd mle-project-sprint-2-v001
Установите необходимые зависимости:

Bash
pip install -r requirements.txt
Настройте переменные окружения для доступа к MLflow и S3.

Запустите ноутбук project_template_sprint_2.ipynb.

Руководство по проекту
Этап 1: Разворачивание MLflow и регистрация модели
Развернут сервер MLflow для отслеживания экспериментов. Базовая модель CatBoost зарегистрирована в Model Registry.

Shell-скрипт: mlflow_server/run_mlflow_server.sh

Имя S3 бакета: s3-student-mle-20250717-d331044a6c-freetrack

Этап 2: Проведение EDA
Проведен анализ распределения цен и характеристик жилья. Выявлена сильная зависимость стоимости от общей площади и удаленности от географического центра города. Обнаружены и обработаны аномалии в данных о годе постройки. 

Название эксперимента: EDA_Research

MLflow Experiment ID: 11

Название запуска в MLflow: EDA_Visualizations_Final

Этап 3: Генерация признаков и обучение модели
Созданы дополнительные признаки: расстояние до центра города, возраст здания, соотношение жилой и общей площади. Для автоматической генерации нелинейных признаков использована библиотека AutoFeat.

Название эксперимента: Feature_Engineering_Stage

MLflow Experiment ID: 12

Название запуска в MLflow: AutoFeat_Plus_Manual

Этап 4: Отбор признаков и обучение новой версии модели
Применена стратегия последовательного отбора признаков (SFS). Сначала через Forward Selection отобраны 15 кандидатов, затем через Backward Elimination количество признаков сокращено до 10 наиболее значимых.

Название эксперимента: Feature_Selection_Stage

MLflow Experiment ID: 13

Название запуска в MLflow: Selection_Optimization

Этап 5: Подбор гиперпараметров и обучение новой версии модели
Выполнен поиск оптимальных параметров (learning_rate, depth, l2_leaf_reg) с помощью библиотеки Optuna (15 итераций) и RandomizedSearchCV. Финальная модель обучена на расширенном количестве итераций.

Название эксперимента: Final_Hyperparameter_Tuning

MLflow Experiment ID: 15

Название запуска в MLflow: Final_Optimized_Model