import os

def query_kb(query):
    file_path = "rag/plan.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "План поездки пока не загружен."
