import uuid

import pytest
import os
import sys
import tempfile

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableBinding

# Добавьте корневую директорию проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorstore import make_vectorstore
from tests.test_vectorstore import create_temp_pdf
from libs.lib import MyPSQLChatMessageHistory
from libs.llm_chat import (check_question,
                           get_history_aware_retriever, define_openai,
                           define_promt, create_chain, contextualize_q_system_prompt,
                           OPENAI_KEY, MODEL, MAX_TOKENS, TEMPERATURE, check_llm,
                           SYSTEM_PROMT, sync_connection)


question = 'Напиши слово: Тест'
table_name = 'test_table'

def create_table(table_name_: str, sync_connection_) -> None:
    # запускать только один раз
    MyPSQLChatMessageHistory.create_tables(sync_connection_, table_name_)

def drop_table(table_name_:str, sync_connection_) -> None:
    MyPSQLChatMessageHistory.drop_table(sync_connection_, table_name_)

@pytest.fixture
def setup_vectorstore():
    """
    Фикстура для подготовки тестового векторного хранилища и создания временного PDF-файла.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "test_document.pdf")
        vec_store_save_path = os.path.join(temp_dir, "test_vectorstore.db")
        # Создание временного PDF-файла
        create_temp_pdf(pdf_path)
        # создание vectorstore
        make_vectorstore(pdf_path, vec_store_save_path)

        yield vec_store_save_path

        # Удаление всех файлов в директории, если они существуют
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))


@pytest.fixture(scope="function")
def setup_test_db():
    # создаём тестовую таблицу
    create_table(table_name, sync_connection)

    session_id = '00001'

    # Initialize the chat history manager
    chat_history = MyPSQLChatMessageHistory(
        table_name=table_name,
        session_id=session_id,
        sync_connection=sync_connection
    )

    # Add messages to the chat history
    chat_history.add_messages([
        AIMessage(content="Test message 1"),
        HumanMessage(content="Test message 2"),
    ])

    yield chat_history

    # удаляем тестовую таблицу
    drop_table(table_name, sync_connection)


def test_check_question():
    assert check_question("Это сообщение содержит слово хуй.") == "оператор"
    assert check_question("Нужна поддержка, пожалуйста.") == "оператор"
    assert check_question("Спасибо за помощь!") == "спасибо"
    assert check_question("Какой у вас режим работы?") == "Какой у вас режим работы?"


def test_get_session_history(setup_test_db):
    chat_history = setup_test_db

    # Проверяем, что объект history является экземпляром MyPSQLChatMessageHistory
    assert isinstance(chat_history, MyPSQLChatMessageHistory)
    assert chat_history._table_name == table_name

    # Получаем сообщения из истории
    messages = chat_history.messages
    # Проверяем, что сообщения соответствуют ожидаемым значениям
    assert len(messages) == 2
    assert messages[0].content == 'Test message 1'
    assert messages[1].content == 'Test message 2'


# Проверка работоспособности моделей
def test_check_llm():
    answer = check_llm(model=MODEL,
                       question=question)
    assert answer == "Тест"

def test_define_openai():
    llm = define_openai(OPENAI_KEY, MODEL, 5, 0.0)
    assert isinstance(llm, ChatOpenAI)


def test_get_history_aware_retriever(setup_vectorstore):
    llm = define_openai(OPENAI_KEY, MODEL, MAX_TOKENS, TEMPERATURE)

    history_aware_retriever = get_history_aware_retriever(llm, setup_vectorstore, contextualize_q_system_prompt)
    assert isinstance(history_aware_retriever, RunnableBinding)


def test_define_promt():
    prompt = define_promt(SYSTEM_PROMT)
    assert isinstance(prompt, ChatPromptTemplate)

    assert "chat_history" in prompt.input_variables


def test_create_chain(setup_vectorstore):
    conversational_rag_chain = create_chain(vec_store_path=setup_vectorstore)
    assert isinstance(conversational_rag_chain, RunnableWithMessageHistory)
    assert 'get_session_history' in conversational_rag_chain.__dict__.keys()
