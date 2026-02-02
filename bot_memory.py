from collections import defaultdict

# Хранилище истории: {user_id: [список сообщений]}
user_histories = defaultdict(list)
MAX_HISTORY = 20

def get_history_text(user_id):
    history = user_histories[user_id]
    if not history:
        return ""
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

def add_to_history(user_id, role, content):
    user_histories[user_id].append({"role": role, "content": content})
    if len(user_histories[user_id]) > MAX_HISTORY:
        user_histories[user_id].pop(0)
