"""
HR Sentiment Analysis — анализ тональности переписок HR-отдела
и расчёт индекса Лосада (Losada Ratio).

Использование:
    python hr_sentiment.py                          # демо с примерами
    python hr_sentiment.py -f chat.txt              # анализ из текстового файла (одно сообщение на строку)
    python hr_sentiment.py -f chat.csv              # анализ из CSV (колонка "message")
    python hr_sentiment.py -f chat.json             # анализ из JSON (список строк)
    python hr_sentiment.py -f chat.txt -o report.json   # сохранить отчёт в JSON
    python hr_sentiment.py -f chat.txt --threshold 0.6  # порог уверенности (по умолчанию 0.5)
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel


# ── Константы ────────────────────────────────────────────────────────────────

LOSADA_LOW = 2.9    # нижняя граница зоны процветания
LOSADA_HIGH = 7.0   # верхняя граница (выше — подозрительно)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# ── Датаклассы ───────────────────────────────────────────────────────────────

@dataclass
class MessageResult:
    """Результат анализа одного сообщения."""
    message: str
    label: str          # positive / negative / neutral
    confidence: float
    raw_scores: dict
    employee: str = ""  # имя сотрудника (если указано)


@dataclass
class EmployeeReport:
    """Отчёт по отдельному сотруднику."""
    name: str
    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    losada_ratio: float = 0.0
    verdict: str = ""
    messages: List[MessageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Полный отчёт по анализу чата."""
    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    losada_ratio: float = 0.0
    verdict: str = ""
    advice: str = ""
    messages: List[MessageResult] = field(default_factory=list)
    employees: List[EmployeeReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Загрузка сообщений из файлов ─────────────────────────────────────────────

def load_messages(filepath: str) -> List[str]:
    """Загрузить сообщения из .txt, .csv или .json файла."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(m) for m in data if str(m).strip()]
        raise ValueError("JSON должен содержать список строк (list of strings).")

    if suffix == ".csv":
        messages = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "message" not in (reader.fieldnames or []):
                raise ValueError('CSV должен содержать колонку "message".')
            for row in reader:
                text = row["message"].strip()
                if text:
                    messages.append(text)
        return messages

    # По умолчанию — текстовый файл, одно сообщение на строку
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_employee_messages(filepath: str) -> List[dict]:
    """
    Загрузить сообщения с привязкой к сотрудникам.

    Поддерживаемые форматы:
    - CSV с колонками "employee" и "message"
    - JSON: список объектов {"employee": "...", "message": "..."}
    - TXT: формат «Имя: сообщение» на каждой строке

    Возвращает список {"employee": str, "message": str}.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            for d in data:
                if "employee" not in d or "message" not in d:
                    raise ValueError(
                        'Каждый объект JSON должен содержать ключи "employee" и "message".'
                    )
            return [{"employee": d["employee"], "message": d["message"]} for d in data]
        raise ValueError(
            'JSON должен содержать список объектов: [{"employee": "...", "message": "..."}].'
        )

    if suffix == ".csv":
        items = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if "employee" not in fields or "message" not in fields:
                raise ValueError('CSV должен содержать колонки "employee" и "message".')
            for row in reader:
                text = row["message"].strip()
                name = row["employee"].strip()
                if text and name:
                    items.append({"employee": name, "message": text})
        return items

    # TXT: «Имя: сообщение»
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, text = line.split(":", 1)
            name, text = name.strip(), text.strip()
            if name and text:
                items.append({"employee": name, "message": text})
    return items


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _compute_losada(positive: int, negative: int) -> float:
    if negative > 0:
        return round(positive / negative, 2)
    if positive > 0:
        return float(positive)
    return 0.0


def _interpret_losada(ratio: float) -> tuple:
    """Возвращает (verdict, advice)."""
    if ratio < LOSADA_LOW:
        return (
            "Зона застоя. Слишком много критики или безразличия.",
            "Рекомендуется увеличить количество позитивной обратной связи: "
            "минимум 3 поддерживающих сообщения на каждое критическое замечание.",
        )
    if ratio <= LOSADA_HIGH:
        return (
            "Зона процветания. Команда работает эффективно.",
            "Поддерживайте текущий баланс коммуникации.",
        )
    return (
        "Зона чрезмерной позитивности. Возможно, реальные проблемы замалчиваются.",
        "Создайте безопасное пространство для конструктивной критики. "
        "Честная обратная связь важна для развития.",
    )


def _classify_sentiment(sentiment: dict, threshold: float) -> tuple:
    """Возвращает (label, confidence)."""
    top = max(sentiment, key=sentiment.get)
    score = sentiment[top]
    if score < threshold:
        label = "neutral"
    elif top not in ("positive", "negative"):
        label = "neutral"
    else:
        label = top
    return label, score


# ── Основная логика анализа ───────────────────────────────────────────────────

def analyze_hr_chat(
    messages: List[str],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> AnalysisReport:
    """
    Анализирует список сообщений, определяет тональность каждого,
    рассчитывает индекс Лосада и возвращает структурированный отчёт.
    """
    if not messages:
        return AnalysisReport(verdict="Нет сообщений для анализа.")

    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    results = model.predict(messages, k=2)

    report = AnalysisReport(total=len(messages))

    for message, sentiment in zip(messages, results):
        label, score = _classify_sentiment(sentiment, confidence_threshold)

        if label == "positive":
            report.positive += 1
        elif label == "negative":
            report.negative += 1
        else:
            report.neutral += 1

        report.messages.append(MessageResult(
            message=message,
            label=label,
            confidence=round(score, 4),
            raw_scores={k: round(v, 4) for k, v in sentiment.items()},
        ))

    report.losada_ratio = _compute_losada(report.positive, report.negative)
    report.verdict, report.advice = _interpret_losada(report.losada_ratio)

    return report


def analyze_by_employee(
    items: List[dict],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> AnalysisReport:
    """
    Анализирует сообщения с привязкой к сотрудникам.

    Каждый элемент items — dict с ключами "employee" и "message".
    Возвращает общий AnalysisReport, в поле employees — разбивка по каждому сотруднику.
    """
    if not items:
        return AnalysisReport(verdict="Нет сообщений для анализа.")

    messages = [item["message"] for item in items]
    employees = [item["employee"] for item in items]

    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(messages, k=2)

    report = AnalysisReport(total=len(messages))
    emp_data: dict = {}  # name -> EmployeeReport

    for item, sentiment in zip(items, results):
        name = item["employee"]
        message = item["message"]
        label, score = _classify_sentiment(sentiment, confidence_threshold)

        msg_result = MessageResult(
            message=message,
            label=label,
            confidence=round(score, 4),
            raw_scores={k: round(v, 4) for k, v in sentiment.items()},
            employee=name,
        )

        # Общая статистика
        report.messages.append(msg_result)
        if label == "positive":
            report.positive += 1
        elif label == "negative":
            report.negative += 1
        else:
            report.neutral += 1

        # Статистика по сотруднику
        if name not in emp_data:
            emp_data[name] = EmployeeReport(name=name)
        emp = emp_data[name]
        emp.total += 1
        emp.messages.append(msg_result)
        if label == "positive":
            emp.positive += 1
        elif label == "negative":
            emp.negative += 1
        else:
            emp.neutral += 1

    # Рассчитываем индексы
    report.losada_ratio = _compute_losada(report.positive, report.negative)
    report.verdict, report.advice = _interpret_losada(report.losada_ratio)

    for emp in emp_data.values():
        emp.losada_ratio = _compute_losada(emp.positive, emp.negative)
        emp.verdict, _ = _interpret_losada(emp.losada_ratio)

    report.employees = sorted(emp_data.values(), key=lambda e: e.losada_ratio)

    return report


# ── Вывод в консоль ──────────────────────────────────────────────────────────

LABEL_DISPLAY = {
    "positive": "✅ Позитив",
    "negative": "❌ Негатив",
    "neutral":  "⚪ Нейтрально",
}


def print_report(report: AnalysisReport) -> None:
    """Красивый вывод отчёта в терминал."""
    col_w = 55
    emp_col = 15

    has_employees = any(m.employee for m in report.messages)

    if has_employees:
        print(f"\n{'СОТРУДНИК':<{emp_col}} | {'СООБЩЕНИЕ':<{col_w}} | ТОНАЛЬНОСТЬ")
        print("-" * (emp_col + col_w + 30))
    else:
        print(f"\n{'СООБЩЕНИЕ':<{col_w}} | ТОНАЛЬНОСТЬ")
        print("-" * (col_w + 25))

    for m in report.messages:
        display = LABEL_DISPLAY.get(m.label, m.label)
        text = m.message if len(m.message) <= col_w else m.message[: col_w - 3] + "..."
        if has_employees:
            name = m.employee if len(m.employee) <= emp_col else m.employee[: emp_col - 2] + ".."
            print(f"{name:<{emp_col}} | {text:<{col_w}} | {display} ({m.confidence:.2f})")
        else:
            print(f"{text:<{col_w}} | {display} ({m.confidence:.2f})")

    print("-" * (col_w + (emp_col + 30 if has_employees else 25)))

    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего сообщений:  {report.total}")
    print(f"  Позитивных:       {report.positive}")
    print(f"  Негативных:       {report.negative}")
    print(f"  Нейтральных:      {report.neutral}")
    print(f"  ─────────────────────────")
    print(f"  🏆 Индекс Лосада (P/N): {report.losada_ratio:.2f}")
    print(f"\n  Вердикт: {report.verdict}")
    print(f"  Совет:   {report.advice}")

    # Вывод по сотрудникам
    if report.employees:
        print(f"\n📋 ИНДЕКС ЛОСАДА ПО СОТРУДНИКАМ:")
        print(f"  {'ИМЯ':<20} | {'P':>3} {'N':>3} {'=':>3} | {'ИНДЕКС':>7} | СТАТУС")
        print(f"  " + "-" * 65)
        for emp in report.employees:
            if emp.losada_ratio < LOSADA_LOW:
                status = "📉 Застой"
            elif emp.losada_ratio <= LOSADA_HIGH:
                status = "🚀 Процветание"
            else:
                status = "⚠️ Слишком позитивно"
            name = emp.name if len(emp.name) <= 20 else emp.name[:18] + ".."
            print(
                f"  {name:<20} | {emp.positive:>3} {emp.negative:>3} {emp.neutral:>3} "
                f"| {emp.losada_ratio:>7.2f} | {status}"
            )

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ тональности HR-переписок и расчёт индекса Лосада."
    )
    parser.add_argument(
        "-f", "--file",
        help="Путь к файлу с сообщениями (.txt, .csv или .json).",
    )
    parser.add_argument(
        "-e", "--employees",
        action="store_true",
        help="Режим анализа по сотрудникам (файл должен содержать поле employee).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Сохранить отчёт в JSON-файл.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Порог уверенности модели (по умолчанию {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.employees:
        if args.file:
            items = load_employee_messages(args.file)
        else:
            # Демо-набор с сотрудниками
            items = [
                {"employee": "Иванов А.",  "message": "Коллеги, отличная работа с отчетом, спасибо всем!"},
                {"employee": "Петрова М.", "message": "Я не понимаю, почему мы снова сорвали дедлайн."},
                {"employee": "Иванов А.",  "message": "Давайте встретимся в 15:00 в переговорке."},
                {"employee": "Сидоров К.", "message": "Этот кандидат просто ужасен, зря потратили время."},
                {"employee": "Петрова М.", "message": "Поздравляю с закрытием вакансии, ты молодец!"},
                {"employee": "Иванов А.",  "message": "Супер, так держать!"},
                {"employee": "Сидоров К.", "message": "Документы лежат на столе."},
                {"employee": "Петрова М.", "message": "Всё плохо, проект горит."},
                {"employee": "Сидоров К.", "message": "Спасибо за помощь, очень выручили!"},
            ]
        report = analyze_by_employee(items, confidence_threshold=args.threshold)
    else:
        if args.file:
            messages = load_messages(args.file)
        else:
            messages = [
                "Коллеги, отличная работа с отчетом, спасибо всем!",
                "Я не понимаю, почему мы снова сорвали дедлайн.",
                "Давайте встретимся в 15:00 в переговорке.",
                "Этот кандидат просто ужасен, зря потратили время.",
                "Поздравляю с закрытием вакансии, ты молодец!",
                "Супер, так держать!",
                "Документы лежат на столе.",
            ]
        report = analyze_hr_chat(messages, confidence_threshold=args.threshold)

    print_report(report)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Отчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
