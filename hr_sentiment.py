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
        top_sentiment = max(sentiment, key=sentiment.get)
        score = sentiment[top_sentiment]

        if score < confidence_threshold:
            label = "neutral"
        else:
            label = top_sentiment

        # speech / skip тоже считаем нейтральными
        if label not in ("positive", "negative"):
            label = "neutral"

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

    # Расчёт индекса Лосада
    if report.negative > 0:
        report.losada_ratio = round(report.positive / report.negative, 2)
    elif report.positive > 0:
        report.losada_ratio = float(report.positive)  # нет негатива вообще
    else:
        report.losada_ratio = 0.0

    # Интерпретация
    if report.losada_ratio < LOSADA_LOW:
        report.verdict = "Зона застоя. Слишком много критики или безразличия."
        report.advice = (
            "Рекомендуется увеличить количество позитивной обратной связи: "
            "минимум 3 поддерживающих сообщения на каждое критическое замечание."
        )
    elif report.losada_ratio <= LOSADA_HIGH:
        report.verdict = "Зона процветания. Команда работает эффективно."
        report.advice = "Поддерживайте текущий баланс коммуникации."
    else:
        report.verdict = (
            "Зона чрезмерной позитивности. "
            "Возможно, реальные проблемы замалчиваются."
        )
        report.advice = (
            "Создайте безопасное пространство для конструктивной критики. "
            "Честная обратная связь важна для развития."
        )

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
    print(f"\n{'СООБЩЕНИЕ':<{col_w}} | ТОНАЛЬНОСТЬ")
    print("-" * (col_w + 25))

    for m in report.messages:
        display = LABEL_DISPLAY.get(m.label, m.label)
        text = m.message if len(m.message) <= col_w else m.message[: col_w - 3] + "..."
        print(f"{text:<{col_w}} | {display} ({m.confidence:.2f})")

    print("-" * (col_w + 25))
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  Всего сообщений:  {report.total}")
    print(f"  Позитивных:       {report.positive}")
    print(f"  Негативных:       {report.negative}")
    print(f"  Нейтральных:      {report.neutral}")
    print(f"  ─────────────────────────")
    print(f"  🏆 Индекс Лосада (P/N): {report.losada_ratio:.2f}")
    print(f"\n  Вердикт: {report.verdict}")
    print(f"  Совет:   {report.advice}")
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

    if args.file:
        messages = load_messages(args.file)
    else:
        # Демо-набор
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
