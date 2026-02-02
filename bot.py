import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from rag.rag_vector_engine import get_rag_response, ingest_docs
from bot_memory import add_to_history, user_histories
from utils.voice_handler import transcribe_voice
from utils.vision_helper import analyze_image

load_dotenv()

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()

if not os.path.exists("data"):
    os.makedirs("data")

# Вспомогательная функция, чтобы не дублировать код для всех типов ввода
async def process_user_request(user_id, query_text):
    # 1. Берем историю ДО добавления текущего вопроса (для чистоты кэша)
    history_list = user_histories.get(user_id, [])[-20:]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])
    
    # 2. Единая точка входа: Кэш -> База -> Интернет
    response = get_rag_response(query_text, history=history_text)
    
    # 3. Сохраняем в память
    add_to_history(user_id, "user", query_text)
    add_to_history(user_id, "assistant", response)
    
    return response

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет, Марго! Я твой умный ассистент. Кэш включен, база готова, бюджет $2500 помню! 🌴")

@dp.message(Command("ingest"))
async def reload_kb(message: types.Message):
    await message.answer("Обновляю базу знаний...")
    count = ingest_docs()
    await message.answer(f"Готово! Загружено фрагментов: {count}")

@dp.message(F.content_type == "voice")
async def handle_voice(message: types.Message):
    user_id = message.from_user.id
    file_id = message.voice.file_id
    file_path = f"data/voice_{file_id}.ogg"
    
    try:
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, file_path)
        text = await transcribe_voice(file_path)
        
        if text:
            # Используем общую логику (Кэш -> База -> Интернет)
            response = await process_user_request(user_id, text)
            await message.answer(f"🎤 [Голос]: {text}\n\n{response}")
        else:
            await message.answer("Марго, не смогла разобрать голос. Повтори, пожалуйста.")
    except Exception as e:
        await message.answer(f"Ошибка голоса: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@dp.message(F.content_type == "photo")
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    photo = message.photo[-1]
    file_path = f"data/img_{photo.file_id}.jpg"
    
    try:
        file = await bot.get_file(photo.file_id)
        await bot.download_file(file.file_path, file_path)
        
        # 1. Зрение анализирует ЧТО на картинке
        image_description = analyze_image(file_path, "Опиши кратко, что на фото.")
        
        # 2. Отправляем описание в общую цепочку RAG (чтобы проверить по базе и кэшу)
        # Например: "На фото договор аренды. Проверь его по моим правилам."
        response = await process_user_request(user_id, f"Пользователь прислал фото. На нем: {image_description}")
        
        await message.answer(f"📸 [Анализ фото]:\n{response}")
    except Exception as e:
        await message.answer(f"Ошибка зрения: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@dp.message()
async def handle_text(message: types.Message):
    response = await process_user_request(message.from_user.id, message.text)
    await message.answer(response)

async def main():
    print("Бот Марго запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())