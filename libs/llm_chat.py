import re
import os
from dotenv import load_dotenv
from pathlib import Path
import psycopg

from langchain_core.runnables import RunnableBinding
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from libs.lib import MyPSQLChatMessageHistory
from vectorstore import load_vectorstore

load_dotenv()

vec_store_save_path = "FAISS_store_10_24.db"
bk_path = "База знаний Чат-Бот ЖМ 10.24.pdf"
# Определение корневого каталога проекта
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
VEC_STORE_LOAD_PATH = Path.joinpath(PROJECT_ROOT, vec_store_save_path)
OPENAI_KEY = os.environ.get('OPENAI_KEY')
HF_KEY = os.environ.get('HF_KEY')
MODEL = 'gpt-4o-mini'
MAX_TOKENS = 500
TEMPERATURE = 1
HISTORY_LIMIT = 6

PSQL_USERNAME = os.environ.get("PSQL_USERNAME")
PSQL_PASSWORD = os.environ.get("PSQL_PASSWORD")
PGSQL_HOST = os.environ.get("PGSQL_HOST")
PGSQL_DATABASE = os.environ.get("PGSQL_DATABASE")
CONN_STR2BD = f"dbname={PGSQL_DATABASE} user={PSQL_USERNAME} password={PSQL_PASSWORD} host={PGSQL_HOST}"
sync_connection = psycopg.connect(CONN_STR2BD)
TABLE = 'chat_bot_history'

SYSTEM_PROMT = """Ты - чат-бот консультант, и работаешь в чате службы поддержки сети магазинов хороших продуктов "Жизньмарт",
    твоя функция - стараться ответить на любой вопрос клиента про работу магазинов "Жизьмарт".
    Ты говоришь только на чистом, грамотном русском языке без ошибок, уважительно и на "Вы"!!! 
    Если вопрос не касается контекста, то вежливо и дружелюбно расскажи про Жизьмарт.

    {context}

    Используй только этот контекст, чтобы ответить на последний вопрос.
    Твой ответ должен быть полным и точно соответствовать тому, что написано в context.
    Если ответа нет в контексте, просто позитивно поддержи диалог на тему Жизньмарта!
    """

contextualize_q_system_prompt = """find the documents closest to the question in meaning"""


def check_llm(model, question, max_tokens=10, temperature=0.0):
    llm = define_openai(OPENAI_KEY, model, max_tokens, temperature)
    answer = llm.invoke(question).content

    return answer


def check_and_make_vectorstore(path_kb: str, vec_store_load_path):
    if not os.path.exists(vec_store_load_path):
        from vectorstore import make_vectorstore
        make_vectorstore(path_kb, vec_store_load_path)


def check_question(message: str) -> str:
    """
    Проверяет сообщение на наличие матерных слов, запросов к оператору и благодарностей.

    Args:
        message (str): Входящее сообщение пользователя.

    Returns:
        str: "оператор" если обнаружены матерные слова или запрос к оператору,
         "спасибо" если сообщение является благодарностью, иначе возвращает исходное сообщение.
    """

    # Примерный список матерных слов на русском языке (закройте ушки)
    curse_words = [
        'хуй', 'хуя', 'пизда', 'ебать', 'ебаный', 'блядь', 'сука', 'пидор', 'гондон', 'мудак', 'сука', 'мразь', 'говно',
        'дерьмо', 'охуел', 'ебанулся', 'дурак', 'заебал'
    ]

    # Расширенный список слов для вызова оператора
    operator_words = [
        'оператор', 'оператора', 'поддержка', 'help', 'support', 'позовите', 'позови', 'assistance', 'customer service',
        'саппорт', 'свяжите', 'соедините', 'роспотребнадзор', 'сотрудник', 'сотрудника', "отзыв", "приборы", "деньги", "кофе"
        "кислое", "кислый", "кислая", "кислые", "испорчено", "испорчены", "испорчено", "испорчен", "тухлое", "тухлый",
        "тухлая", "невкусно","невкусное", "невкусный", "невкусная", "камень", "стекло", "постороннее", "посторонний","пластик", "пленка",
        "мошка", "муха", "таракан", "жук", "живность", "насекомое", "кость", "нитка", "сломан зуб",  "волос", "волосы",
        "прокуратура", "прокуратуру", "суд", "твердый предмет", "твердое", "твердая", "сухо", "сухая", "сухое", "сухой",
        "переварено", "переварен", "переварена", "сырое", "сырая", "сырой", "пересолено", "пересолен","пересолена",
        "непропеченная",  "непропеченное",  "непропеченный", "отравление", "плохое самочувствие", "несвежий",
        "несвежая", "несвежее", "обслуживание", "сервис", "нагрубили","нагрубил","нагрубила",
        "отказали", "плесень", "плесенью", "плесени","грязно", "грязь", "перепутали", "просрок","просрочка",
        "денежные средства", "воняет", "вонь", "задержка", "отвратительные", "отвратительно", "отвратительная","отвратительный",
        "клиентоориентированность", "недолив", "опоздание", "долго", "ожидание", "диарея",  "нить", "разочарование",
        "ужасный", "ужасное", "ужасная", "плохо приготовлено", "обматерили", "обматерил", "остро", "острое", "острый", "острая",
        "кислятина", "тухлятина", "срок",  "курьер", "разлито", "разлили", "разлил", "обжегся", "обжег", "кипяток",
        "вилка", "ложка", "нож", "негатив", "не привезли", "нагрубили", "облился", "облили", "вес", "недовес",
        "расстроен", "расстроена", "зуб", "зубы", "хрустит", "сайт", "приложение", "баг", "ошибка", "ошибку"
    ]

    thanks_words = [
        'спасибо', "благодарю", "спасибо за помощь", "благодарствую", "чао", "пока", "досвидания", "до свидания", 'спс',
        'всего доброго'
    ]

    hello_words = [
        'хай', 'привет', 'добрый день', 'добрый вечер', 'здравствуйте', 'здорова', 'доброе утро'
    ]

    # Приводим сообщение к нижнему регистру для унификации поиска
    message_lower = message.lower()

    # Проверяем на наличие матерных слов
    for curse_word in curse_words:
        if re.search(r'\b' + re.escape(curse_word) + r'\b', message_lower):
            return "оператор"

    # Проверяем на наличие слов для обращения к оператору
    for operator_word in operator_words:
        if re.search(r'\b' + re.escape(operator_word) + r'\b', message_lower):
            return "оператор"

    # Проверяем на окончание диалога
    if message_lower in thanks_words or message_lower in [word + '!' for word in thanks_words]:
        return "спасибо"

    # Проверяем на приветствие
    if message_lower in hello_words or message_lower in [word + '!' for word in hello_words]:
        return "привет"

    return message


def get_session_history(session_id: int) -> BaseChatMessageHistory:
    """
    Возвращает историю сообщений для заданной сессии.

    Args:
        session_id (str): Идентификатор сессии.

    Returns:
        BaseChatMessageHistory: Объект истории сообщений чата.
    """
    # Получаем всю историю для данной сессии
    history = MyPSQLChatMessageHistory(table_name=TABLE,
                                       session_id=session_id,
                                       sync_connection=sync_connection)

    return history


def get_history_aware_retriever(llm: ChatOpenAI, vec_store_path: str | Path,
                                contextualize_q_system_prompt: str) -> RunnableBinding:
    """
    Создает и возвращает ретривер, учитывающий историю чата.

    Args:
        llm (ChatOpenAI): Языковая модель.
        vec_store_path(str): путь к векторному хранилищу базы знаний
        contextualize_q_system_prompt: промт для работы с  историей сообщений
    Returns:
        HistoryAwareRetriever: Ретривер, который учитывает историю чата.
    """
    # проверка наличия векторстора с базой знаний
    check_and_make_vectorstore(bk_path, vec_store_path)

    retriever = load_vectorstore(vec_store_path).as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            # MessagesPlaceholder("chat_history"), включает историю сообщений в запрос к ретриверу
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def define_openai(api_key: str, model: str, max_tokens: int, temperature: float) -> ChatOpenAI:
    """
        Определяет и возвращает языковую модель.

    Args:
        api_key (str): API ключ для доступа к модели.
        model (str): Имя модели.
        max_tokens (int): Максимальное количество токенов для генерации.
        temperature (float): Температура генерации текста.

    Returns:
        ChatOpenAI: Языковая модель.
    """

    llm = ChatOpenAI(openai_api_key=api_key,
                     model_name=model,
                     max_tokens=max_tokens,
                     temperature=temperature)
    return llm


def define_promt(system_prompt: str) -> ChatPromptTemplate:
    """
    Определяет и возвращает системный промт.

    Returns:
        ChatPromptTemplate: Системный промт.
    """
    # Если клиент доволен ответом на вопрос, например, говорит "спасибо", скажи "спасибо" и попрощайся.

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", n_messages=HISTORY_LIMIT),
            ("human", "{input}"),
        ]
    )

    return prompt


def create_chain(vec_store_path: str | Path = VEC_STORE_LOAD_PATH, model: str = MODEL,
                 api_key: str = OPENAI_KEY, system_prompt: str = SYSTEM_PROMT,
                 story_prompt: str = contextualize_q_system_prompt,
                 max_tokens=MAX_TOKENS, temperature=TEMPERATURE) -> RunnableWithMessageHistory:
    """
        Создает и возвращает цепочку обработки запросов с учетом истории чата.

    Returns:
        RunnableWithMessageHistory: Цепочка обработки запросов с учетом истории чата.
    """
    llm = define_openai(api_key, model, max_tokens, temperature)

    prompt = define_promt(system_prompt)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    history_aware_retriever = get_history_aware_retriever(llm, vec_store_path, story_prompt)

    chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    # Create a chain
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
