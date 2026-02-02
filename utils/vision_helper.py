import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, travel_plan="План путешествия не предоставлен"):
    """
    Анализирует изображение и сопоставляет его с планом поездки.
    """
    try:
        base64_image = encode_image(image_path)
        
        # Системный промпт
        instruction = f"""Ты — экспертный гид по Шри-Ланке. 
        1. Опознай достопримечательность на фото. 
        2. Сверь её с планом поездки пользователя, который приведен ниже:
        ---
        {travel_plan}
        ---
        3. Если место есть в плане, обязательно скажи, в какой день пользователь там будет и что у него там запланировано. 
        Если места нет в плане, просто кратко опиши его."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Опознай это место и проверь по моему плану."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка зрения: {e}"
