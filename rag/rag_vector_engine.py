import os
import json
import hashlib
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
DATA_SOURCE = os.path.join(BASE_DIR, "data")
CACHE_FILE = os.path.join(BASE_DIR, "llm_cache.json")

# --- 1. ЛОГИКА КЭШИРОВАНИЯ И ОЧИСТКИ ---

def generate_cache_key(query, history):
    """Создает уникальный хэш для пары Вопрос + История"""
    full_string = f"{history}|{query}".strip().lower()
    return hashlib.md5(full_string.encode()).hexdigest()

def get_from_cache(query, history=""):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                key = generate_cache_key(query, history)
                if key in cache:
                    cache[key]["use_count"] += 1
                    cache[key]["last_access"] = time.time()
                    # Сохраняем обновленную статистику использования
                    with open(CACHE_FILE, 'w', encoding='utf-8') as fw:
                        json.dump(cache, fw, ensure_ascii=False, indent=4)
                    return cache[key]["answer"]
        except: return None
    return None

def save_to_cache(query, history, response):
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except: pass
    
    key = generate_cache_key(query, history)
    cache[key] = {
        "question": query,
        "answer": response,
        "timestamp": time.time(),
        "last_access": time.time(),
        "use_count": 1
    }

    # ОЧИСТКА: Если больше 500 записей, удаляем ту, которую реже всего запрашивали
    if len(cache) > 500:
        # Сортируем по количеству использований и времени последнего доступа
        sorted_keys = sorted(cache.keys(), key=lambda k: (cache[k]['use_count'], cache[k]['last_access']))
        del cache[sorted_keys[0]]

    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# --- 2. РАБОТА С БАЗОЙ (RAG) ---

def ingest_docs():
    """Загрузка документов (Твой код + поддержка PDF)"""
    try:
        txt_loader = DirectoryLoader(DATA_SOURCE, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        pdf_loader = DirectoryLoader(DATA_SOURCE, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = txt_loader.load() + pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=DB_PATH)
        return len(chunks)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return 0

def get_context(query):
    """Поиск в базе памяти (k=5 для оценки Ragas > 70)"""
    if os.path.exists(DB_PATH):
        try:
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings())
            docs = vectorstore.similarity_search_with_relevance_scores(query, k=5)
            # Порог релевантности 0.4-0.5
            return "\n---\n".join([d[0].page_content for d in docs if d[1] > 0.4])
        except: return ""
    return ""

# --- 3. ГЛАВНАЯ ЦЕПОЧКА (Текст / Зрение / Слух) ---

def get_rag_response(query, history=""):
    # ПРИОРИТЕТ 1: КЭШ
    cached_res = get_from_cache(query, history)
    if cached_res:
        return cached_res

    # ПРИОРИТЕТ 2: БАЗА ПАМЯТИ
    context = get_context(query)
    
    # ПРИОРИТЕТ 3: ИНТЕРНЕТ (Tavily)
    search = TavilySearchResults(k=3)
    trigger_words = ["курс", "сегодня", "погода", "цена", "сейчас", "новости"]
    needs_web = not context or any(word in query.lower() for word in trigger_words)
    
    web_info = ""
    if needs_web:
        try:
            web_info = search.run(query)
        except: web_info = "[Веб-поиск недоступен]"

    # ПРОМПТ (Требования + История 20 сообщений)
    system_prompt = (
        f"Ты — строгий ассистент Марго по Шри-Ланке. Бюджет: $2500 на 3 месяца. "
        f"Обязательно наличие рабочего стола. Твой ответ должен быть точным. "
        "Сначала используй контекст из файлов, если его нет — интернет. "
        "Учитывай историю диалога для связности ответов."
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"ИСТОРИЯ ДИАЛОГА (до 20 сообщений):\n{history}\n\nКОНТЕКСТ ИЗ ФАЙЛОВ:\n{context}\n\nИНТЕРНЕТ:\n{web_info}\n\nВОПРОС: {query}"}
    ]
    
    response = llm.invoke(messages).content
    
    # КЭШИРУЕМ РЕЗУЛЬТАТ (для экономии финансов)
    save_to_cache(query, history, response)
        
    return response