"""
Vercel Serverless Function — POST /api/analyze

Принимает JSON:
{
  "messages": [
    {"employee": "Иванов", "message": "Отличная работа!"},
    ...
  ],
  "threshold": 0.5   // необязательно
}

Если employee не указан — общий анализ.
"""

import json
import sys
import os
from http.server import BaseHTTPRequestHandler

# Добавляем корень проекта и текущую директорию в path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from _sentiment_lite import analyze_batch

LOSADA_LOW = 2.9
LOSADA_HIGH = 7.0


def _compute_losada(positive, negative):
    if negative > 0:
        return round(positive / negative, 2)
    if positive > 0:
        return float(positive)
    return 0.0


def _interpret_losada(ratio):
    if ratio < LOSADA_LOW:
        return "Зона застоя. Слишком много критики или безразличия."
    if ratio <= LOSADA_HIGH:
        return "Зона процветания. Команда работает эффективно."
    return "Зона чрезмерной позитивности. Возможно, проблемы замалчиваются."


def _build_report(items, threshold=0.5):
    """Строит отчёт из списка {employee?, message}."""
    messages = [item["message"] for item in items]
    sentiments = analyze_batch(messages)

    total_pos, total_neg, total_neu = 0, 0, 0
    analyzed = []
    employee_stats = {}

    for item, sent in zip(items, sentiments):
        label = sent["label"]
        confidence = sent["confidence"]
        employee = item.get("employee", "")

        if confidence < threshold and label != "neutral":
            label = "neutral"

        entry = {
            "message": item["message"],
            "employee": employee,
            "label": label,
            "confidence": confidence,
        }
        analyzed.append(entry)

        if label == "positive":
            total_pos += 1
        elif label == "negative":
            total_neg += 1
        else:
            total_neu += 1

        if employee:
            if employee not in employee_stats:
                employee_stats[employee] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
            employee_stats[employee]["total"] += 1
            employee_stats[employee][label] += 1

    losada = _compute_losada(total_pos, total_neg)

    employees_report = []
    for name, stats in sorted(employee_stats.items(), key=lambda x: _compute_losada(x[1]["positive"], x[1]["negative"])):
        emp_losada = _compute_losada(stats["positive"], stats["negative"])
        employees_report.append({
            "name": name,
            "positive": stats["positive"],
            "negative": stats["negative"],
            "neutral": stats["neutral"],
            "total": stats["total"],
            "losada_ratio": emp_losada,
            "verdict": _interpret_losada(emp_losada),
        })

    return {
        "total": len(items),
        "positive": total_pos,
        "negative": total_neg,
        "neutral": total_neu,
        "losada_ratio": losada,
        "verdict": _interpret_losada(losada),
        "messages": analyzed,
        "employees": employees_report,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            items = data.get("messages", [])
            threshold = data.get("threshold", 0.5)

            if not items:
                self._respond(400, {"error": "Поле 'messages' обязательно и не может быть пустым."})
                return

            # Нормализация: если прислали просто строки, оборачиваем
            if isinstance(items[0], str):
                items = [{"message": m} for m in items]

            report = _build_report(items, threshold)
            self._respond(200, report)

        except json.JSONDecodeError:
            self._respond(400, {"error": "Невалидный JSON."})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _respond(self, status, data):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
