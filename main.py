from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from cachetools import TTLCache

from libs.llm_chat import (create_chain, check_question, get_session_history,
                           vec_store_save_path, bk_path, check_and_make_vectorstore)
version = '0.3.0'
description = f"""
## Версии
### 0.0.4
- Увеличено покрытия кода тестами
- Настроено базовое ci/cd в github actions
- Добавлена простая html страничка для работы с ботом
### 0.1.0
- Добавлены 9 llm моделей
- Убрана модель без памяти 
- Изменён сплитер текста на деление по символам (т.е. вопросам)
### 0.2.0
- Переход на GPT4-O mini
- В vectorestore уходит только последний запрос пользователя
- В память боту попадают только 3 последних вопроса и ответа
### {version}
- Переписан поиск по БЗ и истории запросов
- Добавлен вызов оператора после 5 вопроса гостя и 10 мин таймер для сброса сессии
"""

# Создание экземпляра FastAPI
app = FastAPI(
    title="Чат-бот API Жизньмарт",
    version=version,
    description=description)


# Модель данных для запроса
class QuestionRequest(BaseModel):
    user_id: str
    question: str


class HistoryRequest(BaseModel):
    user_id: str


# проверка наличия векторстора с базой знаний
check_and_make_vectorstore(bk_path, vec_store_save_path)

# инициализация чат-бота
chain = create_chain()

chat_start_answer      = 'Приветствую! Спешим на помощь💚 Какой у Вас вопрос?'
chat_end_answer        = 'Всегда готовы помочь! Желаем Вам всего самого доброго! 💚'
operator_switch_answer = 'Перевожу на оператора...'

# Создаем кэш с неограниченным размером и временем жизни записей 10 минут
SESSION_TIMEOUT_SECONDS = 10 * 60  # 10 минут
user_message_count = TTLCache(maxsize=10000, ttl=SESSION_TIMEOUT_SECONDS)
MAX_MESSAGES_BEFORE_OPERATOR = 5


def fast_answer(question: str):
    if question == 'оператор':
        # добавить перевод на оператора в JSONResponse
        return JSONResponse(content={"response": operator_switch_answer, 'operator': 1})
    elif question == 'привет':
        return JSONResponse(content={"response": chat_start_answer, 'operator': 0})
    elif question == "спасибо":
        # добавить код окончания диалога в JSONResponse
        return JSONResponse(content={"response": chat_end_answer, 'operator': 0})
    else:
        return None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Открываем файл index.html в режиме чтения с кодировкой utf-8
    """
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            # Читаем содержимое файла
            content = file.read()
        # Возвращаем содержимое файла в ответе
        return HTMLResponse(content=content)
    except Exception as e:
        # Обработка ошибок (например, файл не найден)
        return HTMLResponse(content=f"Не удалось загрузить страницу\n Ошибка {e}", status_code=500)


@app.post("/ask_bot")
async def ask_bot(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель mistralai/mistral-7b-instruct:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Увеличиваем счётчик сообщений пользователя
    count = user_message_count.get(user_id, 0)
    count += 1
    user_message_count[user_id] = count

    # Проверяем, если достигнуто 5 сообщений, переводим на оператора
    if count > MAX_MESSAGES_BEFORE_OPERATOR:
        # Сбрасываем счётчик
        user_message_count.pop(user_id, None)
        return JSONResponse(content={"response": operator_switch_answer, 'operator': 1})

    # Отправка запроса модели
    try:

        response_content = chain.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer'].replace('\n', ''), 'operator': 0})

    except Exception as e:
        print(e)
        return JSONResponse(content={"response": operator_switch_answer, 'operator': 1})


@app.post("/get_history")
async def get_history(request: HistoryRequest):
    user_id = request.user_id
    session_history = get_session_history(user_id)

    # Преобразование сообщений в список словарей
    messages = []
    for message in session_history.messages:
        if isinstance(message, HumanMessage):
            messages.append({"Human": message.content})
        elif isinstance(message, AIMessage):
            messages.append({"AI": message.content})

    return JSONResponse(content={'user_id': user_id, "response": messages})


# Эндпоинт для получения версии приложения
@app.get("/version")
async def get_version():
    return {"version": version}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
