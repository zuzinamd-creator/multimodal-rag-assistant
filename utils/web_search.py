import os
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

load_dotenv()

async def search_internet(query):
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key: return "Инфо из сети недоступно."
        
        # Используем асинхронный клиент
        tavily = AsyncTavilyClient(api_key=api_key)
        response = await tavily.search(query=query, search_depth="advanced", max_results=3, include_answer=True)
        
        return response.get('answer', 'Нет точного ответа в сети.')
    except Exception as e:
        print(f"Ошибка Tavily: {e}")
        return "Ошибка поиска."
