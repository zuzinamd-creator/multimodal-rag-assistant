import asyncio
import os
import pandas as pd
from rag.rag_vector_engine import get_rag_response, get_context
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from datasets import Dataset

# Отключаем лишние логи
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Тестовые вопросы на основе твоих данных
questions = [
    "Какой общий бюджет у Марго на поездку?",
    "Какие требования у Марго к жилью в Полхене?",
    "Когда Марго планирует быть в Сигирии и что там делать?"
]

# Правильные ответы для сравнения (Ground Truth)
ground_truths = [
    "Общий бюджет на 3 месяца составляет $2500 за все.",
    "Обязательно наличие полноценного рабочего стола (не кофейного) и кондиционера.",
    "26 февраля — подъем на Львиную скалу, 27 февраля — Пидурангала и сафари."
]

async def run_eval():
    print("Собираем ответы от бота и контекст для оценки...")
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": ground_truths
    }

    for q in questions:
        # Получаем ответ через нашу новую цепочку (без истории для чистоты теста)
        ans = get_rag_response(q, history="") 
        # Получаем контекст, который реально видит база (наш k=5)
        ctx = get_context(q)
        
        data["question"].append(q)
        data["answer"].append(ans)
        data["contexts"].append([ctx] if ctx else ["Контекст не найден"])
        print(f"Обработан вопрос: {q}")

    dataset = Dataset.from_dict(data)
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
    
    print("\nЗапускаем расчет метрик Ragas (Faithfulness и Relevancy)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm
    )
    
    print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ ---")
    print(result)
    
    # Сохраняем отчет для куратора
    df = result.to_pandas()
    df.to_csv("ragas_report.csv", index=False, encoding='utf-8-sig')
    print("\nМарго, подробный отчет сохранен в ragas_report.csv")

if __name__ == "__main__":
    asyncio.run(run_eval())