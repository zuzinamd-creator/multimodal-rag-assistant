import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Мы добавили параметр history, чтобы бот помнил, о чем говорили ранее
def get_response(user_text, history=None):
    try:
        messages = [
            {
                "role": "system", 
                "content": (
                    "Ты — умный ассистент Маргариты. Твоя база знаний сосредоточена на Шри-Ланке, "
                    "но ты должен быть внимателен к контексту. Если пользователь задает общий вопрос "
                    "(например, 'где лучший серфинг?'), прежде чем давать ответ по Шри-Ланке, "
                    "уточни: 'Вы имеете в виду Шри-Ланку или другие места в мире?'. "
                    "Если вопрос 'А в мире?', соотноси его с предыдущей темой обсуждения (например, серфингом)."
                )
            }
        ]
        
        # Добавляем историю, если она есть
        if history:
            messages.extend(history)
        
        # Добавляем текущий вопрос
        messages.append({"role": "user", "content": str(user_text)})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Ошибка OpenAI: {e}"
