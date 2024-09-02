import uuid

import pytest
import os
import sys
import tempfile

from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.runnables import RunnableBinding

# Добавьте корневую директорию проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vectorstore import make_vectorstore
from tests.test_vectorstore import create_temp_pdf
from libs.llm_chat import (check_question,
                           get_history_aware_retriever, define_llm,
                           define_promt, create_chain, contextualize_q_system_prompt,
                           API_KEY, API_BASE, MODEL, MAX_TOKENS, TEMPERATURE, check_llm,
                           llama_3_1_8b, hermes, openchat, qwen2, zephyr, phi3, gemma2, mythomist,
                           create_table, sync_connection, drop_table, SYSTEM_PROMT, GIGACHAT_KEY)

question = 'Напиши слово: Тест'
question_2 = 'Напиши слово: "Тест". Напиши только одно слово на русском языке'
test_question = """Делай в точности то, что я скажу. Ты можешь отвечать только одним словом!
 Не используй точки, запятые и другие знаки препинания в своём ответе! Напиши слово: Тест"""

table_name = 'test_table'


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
    create_table(table_name)

    session_id = str(uuid.uuid4())

    # Initialize the chat history manager
    chat_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )

    # Add messages to the chat history
    chat_history.add_messages([
        AIMessage(content="Test message 1"),
        HumanMessage(content="Test message 2"),
    ])

    yield chat_history

    # удаляем тестовую таблицу
    drop_table(table_name)


def test_check_question():
    assert check_question("Это сообщение содержит слово хуй.") == "оператор"
    assert check_question("Нужна поддержка, пожалуйста.") == "оператор"
    assert check_question("Спасибо за помощь!") == "спасибо"
    assert check_question("Какой у вас режим работы?") == "Какой у вас режим работы?"


def test_get_session_history(setup_test_db):
    chat_history = setup_test_db

    # Проверяем, что объект history является экземпляром PostgresChatMessageHistory
    assert isinstance(chat_history, PostgresChatMessageHistory)
    assert chat_history._table_name == table_name

    # Получаем сообщения из истории
    messages = chat_history.messages
    # Проверяем, что сообщения соответствуют ожидаемым значениям
    assert len(messages) == 2
    assert messages[0].content == 'Test message 1'
    assert messages[1].content == 'Test message 2'


# Проверка работоспособности моделей
def test_llama_3_1_8b():
    answer = check_llm(model=llama_3_1_8b,
                       question=question)
    assert answer == "Тест."


def test_hermes():
    answer = check_llm(model=hermes,
                       question=question)
    assert answer == "Тест"


def test_openchat():
    answer = check_llm(model=openchat,
                       question=question)
    assert answer == "Тест"


def test_mistral_7b():
    answer = check_llm(model=MODEL,
                       question=question, temperature=0.01)
    assert answer in ["Тест", " Тест", "тест", " тест"]


def test_qwen2():
    answer = check_llm(model=qwen2,
                       question=question)
    assert answer == "Тест"


def test_zephyr():
    answer = check_llm(model=zephyr,
                       question=question_2, temperature=0.01)
    assert answer == "Тест"


def test_phi3():
    answer = check_llm(model=phi3,
                       question=question_2)
    assert answer in [" Тест", " тест"]


def test_gemma2():
    answer = check_llm(model=gemma2,
                       question=question_2)
    assert answer == "\n\nТест \n"


def test_mythomist():
    answer = check_llm(model=mythomist,
                       question='Привет!')

    assert answer == " Привет, как дела? Готовы"


def test_gigachat():
    llm = GigaChat(credentials=GIGACHAT_KEY,
                   verify_ssl_certs=False,
                   model='GigaChat')
    answer = llm.invoke(test_question).content
    assert answer == 'Тест'


def test_define_llm():
    llm = define_llm(API_KEY, API_BASE, MODEL, 5, 0.0)
    assert isinstance(llm, ChatOpenAI)


def test_get_history_aware_retriever(setup_vectorstore):
    llm = define_llm(API_KEY, API_BASE, MODEL, MAX_TOKENS, TEMPERATURE)

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
