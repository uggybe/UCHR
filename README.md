# HR Sentiment Analysis — Индекс Лосада

Анализ тональности HR-переписок и расчёт индекса Лосада (Losada Ratio) для оценки эффективности коммуникации в команде.

## Возможности

- Анализ тональности сообщений (позитив / негатив / нейтрально)
- Расчёт индекса Лосада — общий и **по каждому сотруднику**
- Поддержка форматов: `.txt`, `.csv`, `.json`
- Web-интерфейс с визуализацией
- Деплой на Vercel (serverless)

## Быстрый старт

### CLI (с моделью Dostoevsky)

```bash
pip install -r requirements.txt
python setup_model.py

# Общий анализ
python hr_sentiment.py

# Анализ по сотрудникам
python hr_sentiment.py -e

# Из файла с разбивкой по сотрудникам
python hr_sentiment.py -e -f chat.csv

# Экспорт в JSON
python hr_sentiment.py -e -f chat.csv -o report.json
```

### Web UI (Vercel)

```bash
# Локальный запуск (нужен Vercel CLI)
npm i -g vercel
vercel dev

# Деплой
vercel --prod
```

## Форматы входных данных

### TXT (с сотрудниками)
```
Иванов А.: Отличная работа с отчетом!
Петрова М.: Почему снова сорвали дедлайн?
```

### CSV
```csv
employee,message
Иванов А.,Отличная работа с отчетом!
Петрова М.,Почему снова сорвали дедлайн?
```

### JSON
```json
[
  {"employee": "Иванов А.", "message": "Отличная работа с отчетом!"},
  {"employee": "Петрова М.", "message": "Почему снова сорвали дедлайн?"}
]
```

## Индекс Лосада

| Значение | Интерпретация |
|----------|---------------|
| < 2.9    | Зона застоя — слишком много критики |
| 2.9–7.0  | Зона процветания — баланс в норме |
| > 7.0    | Чрезмерная позитивность — проблемы могут замалчиваться |

## Архитектура

```
├── hr_sentiment.py        # CLI: полный анализ с моделью Dostoevsky
├── setup_model.py         # Загрузка ML-модели
├── requirements.txt       # Зависимости для CLI
├── api/
│   ├── analyze.py         # Vercel serverless endpoint
│   ├── sentiment_lite.py  # Лёгкий анализатор (словарный, без ML)
│   └── requirements.txt   # Зависимости для Vercel
├── public/
│   └── index.html         # Web UI
└── vercel.json            # Конфиг деплоя
```

> **Примечание:** Web-версия на Vercel использует облегчённый словарный анализатор (`sentiment_lite.py`),
> т.к. ML-модель Dostoevsky (~300 МБ) не помещается в serverless-окружение.
> Для точного анализа используйте CLI-версию с полной моделью.
