import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

def test_all():
    print("--- 1. Проверка OpenAI API ---")
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        res = llm.invoke("Привет, ты тут?")
        print(f"ОК: OpenAI ответил: {res.content[:20]}...")
    except Exception as e:
        print(f"ОШИБКА OpenAI: {e}")

    print("\n--- 2. Проверка Базы (RAG) ---")
    db_path = "/root/lanka_bot/data/chroma_db"
    if os.path.exists(db_path):
        try:
            vectorstore = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())
            docs = vectorstore.similarity_search("бюджет", k=1)
            print(f"ОК: База нашла документов: {len(docs)}")
        except Exception as e:
            print(f"ОШИБКА RAG: {e}")
    else:
        print("ОШИБКА: Папка базы не найдена!")

    print("\n--- 3. Проверка Поиска (Tavily) ---")
    try:
        search = TavilySearchResults(k=1)
        res = search.run("курс рупии к доллару сегодня")
        print(f"ОК: Поиск работает, получены данные.")
    except Exception as e:
        print(f"ОШИБКА Поиска: {e}")

if __name__ == "__main__":
    test_all()
