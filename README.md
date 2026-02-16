# HR Sentiment Analysis — Индекс Лосада

Анализ тональности HR-переписок и расчёт индекса Лосада (Losada Ratio) для оценки эффективности коммуникации в команде.

## Установка

```bash
pip install -r requirements.txt
python setup_model.py
```

## Использование

```bash
# Демо с встроенными примерами
python hr_sentiment.py

# Анализ из файла (txt — одно сообщение на строку)
python hr_sentiment.py -f chat.txt

# Анализ из CSV (нужна колонка "message")
python hr_sentiment.py -f chat.csv

# Анализ из JSON (список строк)
python hr_sentiment.py -f chat.json

# Сохранить отчёт в JSON
python hr_sentiment.py -f chat.txt -o report.json

# Изменить порог уверенности модели
python hr_sentiment.py -f chat.txt --threshold 0.6
```

## Индекс Лосада

| Значение | Интерпретация |
|----------|---------------|
| < 2.9    | Зона застоя — слишком много критики |
| 2.9–7.0  | Зона процветания — баланс в норме |
| > 7.0    | Чрезмерная позитивность — проблемы могут замалчиваться |
