"""Скрипт для первоначальной загрузки модели Dostoevsky."""
from dostoevsky.data import DataDownloader

if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download(source="fasttext-social-network-model")
    print("Модель успешно загружена.")
