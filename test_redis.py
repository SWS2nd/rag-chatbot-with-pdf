import redis
import json
from dotenv import load_dotenv
from urllib.parse import urlparse
import os

# .env 불러오기
load_dotenv()

# URL 파싱
REDIS_URL = os.getenv("REDIS_URL")
redis_url = urlparse(REDIS_URL)

REDIS_HOST = redis_url.hostname
REDIS_PORT = redis_url.port
REDIS_DB = int(redis_url.path.lstrip("/")) if redis_url.path else 0
REDIS_PASSWORD = redis_url.password

# Redis 접속
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# 사용자 입력
session_id = input("확인하고 싶은 세션 ID를 입력하세요: ")
redis_key = f"message_store:{session_id}"

# Redis에 해당 키가 있는지 확인
if r.exists(redis_key):
    messages = r.lrange(redis_key, 0, -1)
    print(f"\n세션 '{session_id}' 메시지 기록:\n")
    
    for i, msg in enumerate(messages, 1):
        try:
            # 이미 decode_responses=True라서 bytes -> str 필요 없음
            msg_obj = json.loads(msg)
            
            msg_type = msg_obj.get("type", "unknown")
            content = msg_obj.get("data", {}).get("content", "")
            
            # 메시지 타입에 따라 출력 색상/구분
            if msg_type == "ai":
                print(f"{i}. 🤖 AI 답변:\n{content}\n{'-'*50}")
            elif msg_type == "human":
                print(f"{i}. 👤 유저 질문:\n{content}\n{'-'*50}")
            else:
                print(f"{i}. {msg_type}:\n{content}\n{'-'*50}")
                
        except json.JSONDecodeError:
            print(f"{i}. ⚠️ JSON 디코딩 실패:\n{msg}\n{'-'*50}")
else:
    print(f"키 '{redis_key}'는 Redis에 존재하지 않습니다. 새 세션을 생성하지 않았습니다.")
