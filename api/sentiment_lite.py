"""
Лёгкий анализатор тональности русского текста на основе словаря.

Используется в Vercel serverless (dostoevsky + fasttext слишком тяжёлые
для serverless-окружения). Для production-анализа используйте hr_sentiment.py
с моделью Dostoevsky.
"""

import re
from typing import List

# ── Словари ──────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "отлично", "отличная", "отличный", "отличное", "отличные",
    "спасибо", "молодец", "молодцы", "супер", "здорово", "прекрасно",
    "поздравляю", "поздравляем", "браво", "замечательно", "класс",
    "великолепно", "хорошо", "хорошая", "хороший", "хорошее",
    "рад", "рада", "рады", "радует", "благодарю", "благодарим",
    "успех", "успешно", "выручили", "помогли", "помощь",
    "круто", "победа", "ура", "держать", "вперёд", "вперед",
    "нравится", "понравилось", "приятно", "удачно", "удача",
    "эффективно", "продуктивно", "достижение", "прогресс",
    "поддержка", "поддерживаю", "согласен", "согласна",
    "восхитительно", "превосходно", "идеально", "безупречно",
    "рекомендую", "впечатляет", "вдохновляет",
}

NEGATIVE_WORDS = {
    "ужасен", "ужасно", "ужасный", "ужасная", "ужасные",
    "плохо", "плохой", "плохая", "плохие", "плохое",
    "зря", "сорвали", "провал", "провалили", "срыв",
    "не понимаю", "невозможно", "проблема", "проблемы",
    "критично", "катастрофа", "кошмар", "отвратительно",
    "разочарован", "разочарована", "недовольна", "недоволен",
    "горит", "сгорел", "дедлайн", "опоздали", "затянули",
    "ошибка", "ошибки", "баг", "сбой", "жалоба", "жалобы",
    "увольнение", "штраф", "конфликт", "скандал",
    "отстой", "бесит", "раздражает", "надоело",
    "некомпетентен", "некомпетентна", "бездарно", "безобразие",
    "хуже", "худший", "неприемлемо", "недопустимо",
}

INTENSIFIERS = {"очень", "крайне", "чрезвычайно", "невероятно", "слишком", "абсолютно"}
NEGATIONS = {"не", "ни", "нет", "без", "никак", "ничуть", "вовсе"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r'[а-яёА-ЯЁa-zA-Z]+', text.lower())


def analyze_message(text: str) -> dict:
    """
    Анализирует одно сообщение. Возвращает:
    {"label": "positive"|"negative"|"neutral", "confidence": float}
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"label": "neutral", "confidence": 0.5}

    pos_score = 0.0
    neg_score = 0.0
    prev_token = ""

    for token in tokens:
        is_negated = prev_token in NEGATIONS
        multiplier = 1.5 if prev_token in INTENSIFIERS else 1.0

        if token in POSITIVE_WORDS:
            if is_negated:
                neg_score += multiplier
            else:
                pos_score += multiplier
        elif token in NEGATIVE_WORDS:
            if is_negated:
                pos_score += multiplier * 0.5  # «не плохо» слабее чем «хорошо»
            else:
                neg_score += multiplier

        prev_token = token

    # Проверка биграмм
    text_lower = text.lower()
    for phrase in ("не понимаю", "так держать", "зря потратили"):
        if phrase in text_lower:
            if phrase == "так держать":
                pos_score += 1.5
            else:
                neg_score += 1.5

    total = pos_score + neg_score
    if total == 0:
        return {"label": "neutral", "confidence": 0.5}

    if pos_score > neg_score:
        return {"label": "positive", "confidence": round(min(pos_score / (total + 1), 0.99), 4)}
    elif neg_score > pos_score:
        return {"label": "negative", "confidence": round(min(neg_score / (total + 1), 0.99), 4)}
    else:
        return {"label": "neutral", "confidence": 0.5}


def analyze_batch(messages: List[str]) -> List[dict]:
    """Анализирует список сообщений."""
    return [analyze_message(m) for m in messages]
