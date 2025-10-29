import requests
import time
from src.config import GIGACHAT_AUTH_KEY, GIGACHAT_CLIENT_SECRET, AUTH_URL

# Глобальные переменные для кэширования
_CACHED_TOKEN = None
_CACHED_TOKEN_TS = 0


def get_token():
    global _CACHED_TOKEN, _CACHED_TOKEN_TS

    now = time.time()
    if _CACHED_TOKEN is not None and now - _CACHED_TOKEN_TS <= 25 * 60:
        return _CACHED_TOKEN  # используем кэш

    # Запрашиваем новый токен
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': GIGACHAT_CLIENT_SECRET,
        'Authorization': f'Basic {GIGACHAT_AUTH_KEY}'
    }
    payload = {'scope': 'GIGACHAT_API_PERS'}

    try:
        response = requests.post(AUTH_URL, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        access_token = response.json().get('access_token')
        if access_token:
            print("✅ Токен обновлён.")
            _CACHED_TOKEN = access_token
            _CACHED_TOKEN_TS = now
            return access_token
        else:
            print("❌ Токен не получен.")
            return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None