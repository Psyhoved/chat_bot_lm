import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from libs.llm_chat import (create_chain, check_question, get_session_history, MODEL,
                           llama_3_1_8b, hermes, openchat, capybara, qwen2, zephyr, phi3, gemma2, mythomist,
                           vec_store_save_path)

description = """
## Версии
### 0.0.3
- Добавлено сохранение истории диалогов в базу данных SQLlite.
- Исправлен баг с неадекватным ответом на приветствие
- Увеличено покрытия кода тестами
### 0.0.4
- Увеличено покрытия кода тестами
- Настроено базовое ci/cd в github actions
- Добавлена простая html страничка для работы с ботом
### 0.1.0
- Добавлены 9 llm моделей
- Убрана модель без памяти 
- Изменён сплитер текста на деление по символам (т.е. вопросам)
"""

# Создание экземпляра FastAPI
app = FastAPI(
    title="Чат-бот API Жизньмарт",
    version="0.1.0",
    description=description)


# Модель данных для запроса
class QuestionRequest(BaseModel):
    user_id: str
    question: str


class HistoryRequest(BaseModel):
    user_id: str


# проверка наличия векторстора с базой знаний
bk_path = "База знаний Чат-Бот ЖМ 09.24.pdf"

if not os.path.exists(vec_store_save_path):
    from vectorstore import make_vectorstore

    make_vectorstore(bk_path, vec_store_save_path)

del bk_path, vec_store_save_path

# инициализация чат-ботов

chain = create_chain(model=MODEL)

chain_llama_3_1_8b = create_chain(model=llama_3_1_8b)
chain_hermes       = create_chain(model=hermes)
chain_zephyr       = create_chain(model=zephyr)
chain_openchat     = create_chain(model=openchat)
chain_phi3         = create_chain(model=phi3)
chain_gemma2       = create_chain(model=gemma2)
chain_qwen2        = create_chain(model=qwen2)
chain_capybara     = create_chain(model=capybara)
chain_mythomist    = create_chain(model=mythomist)

chat_start_answer      = 'Приветствую! Спешим на помощь💚 Какой у Вас вопрос?'
chat_end_answer        = 'Всегда готовы помочь! Желаем Вам всего самого доброго! 💚'
operator_switch_answer = 'Перевожу на оператора...'


def fast_answer(question: str):
    if question == 'оператор':
        # добавить перевод на оператора в JSONResponse
        return JSONResponse(content={"response": operator_switch_answer})
    elif question == 'привет':
        return JSONResponse(content={"response": chat_start_answer})
    elif question == "спасибо":
        # добавить код окончания диалога в JSONResponse
        return JSONResponse(content={"response": chat_end_answer})
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


@app.post("/ask_mistral_7b")
async def ask_mistral_7b(request: QuestionRequest):
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

    # Отправка запроса модели
    try:

        response_content = chain.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_llama_3_1_8b")
async def ask_llama_3_1_8b(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель meta-llama/llama-3.1-8b-instruct:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_llama_3_1_8b.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_hermes")
async def ask_hermes(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель nousresearch/hermes-3-llama-3.1-405b.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_hermes.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_openchat")
async def ask_openchat(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель openchat/openchat-7b:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_openchat.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_qwen2")
async def ask_qwen2(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель qwen/qwen-2-7b-instruct:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_qwen2.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_zephyr")
async def ask_zephyr(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель huggingfaceh4/zephyr-7b-beta:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_zephyr.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_phi3")
async def ask_phi3(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель microsoft/phi-3-medium-128k-instruct:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_phi3.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_gemma2")
async def ask_gemma2(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель google/gemma-2-9b-it:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_gemma2.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_mythomist")
async def ask_mythomist(request: QuestionRequest):
    """
    Общение с ботом с учётом истории сообщений с пользователем, модель gryphe/mythomist-7b:free.
    В промт идёт вся прошлая переписка.

    :param request:

    :return:
    """
    user_id = request.user_id
    question = check_question(request.question)

    if fast_answer(question):
        return fast_answer(question)

    # Отправка запроса модели
    try:

        response_content = chain_mythomist.invoke({"input": question}, config={"configurable": {"session_id": user_id}})

        return JSONResponse(content={"response": response_content['answer']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
