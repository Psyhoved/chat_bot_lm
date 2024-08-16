import os
import sys
import pytest
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_community.vectorstores import FAISS

# Добавьте корневую директорию проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorstore import make_vectorstore, load_vectorstore


@pytest.fixture
def setup_vectorstore():
    """
    Фикстура для подготовки тестового векторного хранилища и создания временного PDF-файла.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "test_document.pdf")

        # Создание временного PDF-файла
        create_temp_pdf(pdf_path)

        yield temp_dir, pdf_path

        # Удаление всех файлов в директории, если они существуют
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))


def create_temp_pdf(path):
    """
    Функция для создания временного PDF-файла.
    """
    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "This is a temporary PDF file for testing purposes.")
    c.save()


def test_make_vectorstore(setup_vectorstore):
    """
    Тест функции make_vectorstore.
    """
    temp_dir, pdf_path = setup_vectorstore
    vectorstore_save_path = os.path.join(temp_dir, "vectorstore")

    make_vectorstore(pdf_path, vectorstore_save_path)
    assert os.path.exists(vectorstore_save_path), "Vectorstore was not saved."


def test_load_vectorstore(setup_vectorstore):
    """
    Тест функции load_vectorstore.
    """
    temp_dir, pdf_path = setup_vectorstore
    vectorstore_save_path = os.path.join(temp_dir, "vectorstore")

    make_vectorstore(pdf_path, vectorstore_save_path)
    vectorstore = load_vectorstore(vectorstore_save_path)
    assert isinstance(vectorstore, FAISS), "Loaded object is not a FAISS instance."


def test_vectorstore_content(setup_vectorstore):
    """
    Тест содержимого векторного хранилища.
    """
    temp_dir, pdf_path = setup_vectorstore
    vectorstore_save_path = os.path.join(temp_dir, "vectorstore")

    make_vectorstore(pdf_path, vectorstore_save_path)
    vectorstore = load_vectorstore(vectorstore_save_path)
    # Проверка, что векторное хранилище содержит данные
    assert vectorstore.index.ntotal > 0, "Vectorstore is empty."
