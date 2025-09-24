import time
import os, getpass, base64, uuid, tempfile
from typing import Dict, List, Any, Optional
import hashlib # 새 pdf 파일 판단 시 사용
import streamlit as st
import redis

from langchain_upstage import ChatUpstage, UpstageEmbeddings 
from langchain_chroma import Chroma # 로컬 테스트용 크로마 벡터스토어
import chromadb # 로컬 테스트용 크로마db
from langchain_community.vectorstores import Redis # 배포용 레디스 벡터스토어
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage

from langsmith import Client

# 🔹 추가: utils_rag.py에서 Streamlit용 함수와 핸들러 가져오기
# Streamlit용 유틸 함수
from utils_rag import init_conversation, print_conversation, StreamHandler

# 🔹 추가: 로컬 테스트용 in-memory 히스토리
from langchain.memory import ChatMessageHistory  
# 🔹 추가: Redis 기반 채팅 기록 (LangChain RedisChatMessageHistory 사용)
from langchain_community.chat_message_histories import RedisChatMessageHistory

from dotenv import load_dotenv


# ----------------------------
# 환경변수 로딩
# ----------------------------
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
# .env 파일 내용 상황에 맞게 주석 처리 필요
# "chroma"=로컬, "redis"=배포
# 로컬 테스트: VECTORSTORE="chroma" → Chroma + ChatMessageHistory
# 배포: VECTORSTORE="redis" → Redis + RedisChatMessageHistory
VECTORSTORE = os.getenv("VECTORSTORE")

# 불러온 값 확인 (디버깅용, 실제 코드에선 print는 지워도 됨)
print("UPSTAGE_API_KEY:", os.getenv("UPSTAGE_API_KEY"))
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
print("VECTORSTORE:", VECTORSTORE)

# ----------------------------
# LangSmith 인증 테스트
# ----------------------------
# LangSmith client 설정
client = Client()  # env로부터 자동 설정
# 프로젝트 목록이 보이면 인증 OK
projects = list(client.list_projects())[:3]
print("LangSmith projects (sample):", [p.name for p in projects] or "(none)")

# ----------------------------
# LLM + 기본 프롬프트 설정
# ----------------------------
# Upstage 모델: 기본값(서버 디폴트) 사용 — 필요 시 model="solar-pro-2" 등으로 지정 가능
llm = ChatUpstage(temperature=0.2)  # 기본 테스트 권장: temperature=0.2
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 한국어로 정확하고 간결하게 답하는 시니어 AI 멘토야."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

if "chain_test_done" not in st.session_state:
    chain_test_result = chain.invoke({"question": "Streamlit의 placeholder 용도 한 줄로 설명해줘."})
    print(chain_test_result)
    st.session_state.chain_test_done = True

# 🔹 주석 처리 부분: 사용자가 직접 세션을 입력하는 것으로 변경함
# 기존 로컬 세션 초기화 부분(UUID 생성 및 file_cache)
# 세션 상태 초기화
#if "id" not in st.session_state:
#    st.session_state.id = uuid.uuid4()
#    st.session_state.file_cache = {} 
# 세션 ID 설정
#session_id = st.session_state.id
#client = None

# ----------------------------
# 전역 세션 관리 (기본: '1'로 고정 테스트 원하면 여기 수정)
# ----------------------------
# 세션 ID 초기화
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "1"
session_id = st.session_state["session_id"]

# file_cache 초기화
if "file_cache" not in st.session_state:
    st.session_state["file_cache"] = {}

# ----------------------------
# 메시지 히스토리 초기화(로컬/Redis 모드)
# ----------------------------
# Redis 대화 기록 관리 함수
def get_redis_message_history(session_id: str):
    """Redis 기반 메시지 기록 객체 생성"""
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

if VECTORSTORE == "redis":
    if "redis_history" not in st.session_state:
        st.session_state["redis_history"] = get_redis_message_history(session_id)
    message_history = st.session_state["redis_history"]
else:
    if "memory_history" not in st.session_state:
        st.session_state["memory_history"] = ChatMessageHistory()
    message_history = st.session_state["memory_history"]
    
# ----------------------------
# PDF 파일 디스플레이 함수
# ----------------------------
def display_pdf(file):
    st.markdown("## PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    file.seek(0)
    st.markdown(
        f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100vh" type="application/pdf"></iframe>""",
        unsafe_allow_html=True
    )

# ----------------------------
# 사이드바: 세션 변경 / 파일 업로드 / 대화 초기화
# ----------------------------
with st.sidebar:
    session_id_input = st.text_input("Session ID를 입력하세요", value=session_id)
    # 🔹 중요: 세션 ID가 바뀌면 이전 모든 상태 초기화 필요
    # 사용자가 입력한 session id가 기본 session id 값과 다를 경우
    # 입력한 session id로 세션 상태 변경 후 해당 세션으로 메시지 객체 갱신
    if session_id_input != session_id:
        st.session_state["session_id"] = session_id_input
        session_id = session_id_input
        
        # 🔹 RAG 체인 초기화
        st.session_state.pop("rag_chain", None)
        # 🔹 messages 초기화
        st.session_state["messages"] = []
        # 🔹 파일 캐시 초기화
        st.session_state["file_cache"] = {}
        # 🔹 이전 파일 참조 초기화
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("last_file_key", None)
        # 🔹 이전 vectorstore 객체 제거 (로컬 Chroma 참조도 제거)
        st.session_state.pop("vectorstore", None)
        # 🔹 로컬 모드: 메모리 히스토리 초기화
        if VECTORSTORE == "chroma":
            st.session_state["memory_history"] = ChatMessageHistory()
        # 🔹 배포 모드: Redis 히스토리 초기화
        if VECTORSTORE == "redis":
            st.session_state["redis_history"] = get_redis_message_history(session_id)
    
    # 명시적 대화 기록 초기화 버튼 (채팅을 나눈 히스토리만 삭제하여 업로드 한 pdf는 그대로 기억하고 있도록 함)
    if st.button("채팅 초기화 🗑️"):
        # 🔹 히스토리 초기화
        st.session_state["messages"] = []
        if VECTORSTORE == "chroma":
            st.session_state["memory_history"] = ChatMessageHistory()
        if VECTORSTORE == "redis":
            st.session_state["redis_history"] = get_redis_message_history(session_id)
        st.rerun()
    
    # PDF 파일 업로드 처리
    st.header(f"📂Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    
    # 파일 해시 생성 함수(제목이 같아도 내용이 달라지는 pdf 파일까지 구분하기 위함)
    def get_file_hash(file) -> str:
        """파일 내용을 읽어서 SHA256 해시 반환"""
        file.seek(0)
        content = file.read()
        file.seek(0)  # 파일 포인터 복원
        return hashlib.sha256(content).hexdigest()
    
    if uploaded_file is not None:
        print(uploaded_file)
        # 🔹 새 PDF 판단용 key: 세션id + 파일이름 + 파일 해시
        file_hash = get_file_hash(uploaded_file)
        file_key = f"{session_id}-{uploaded_file.name}-{file_hash}"
        
        # 🔹 이전 파일과 다르면 초기화
        if st.session_state.get("last_file_key") != file_key:
            st.session_state.pop("rag_chain", None)
            st.session_state["messages"] = []
            st.session_state["file_cache"] = {}
            st.session_state.pop("uploaded_file", None)
            st.session_state.pop("last_file_key", None)
            st.session_state.pop("vectorstore", None)
            
            # 🔹 로컬 모드: 메모리 히스토리 초기화
            if VECTORSTORE == "chroma":
                st.session_state["memory_history"] = ChatMessageHistory()
            # 🔹 배포 모드: Redis 히스토리 초기화
            if VECTORSTORE == "redis":
                st.session_state["redis_history"] = get_redis_message_history(session_id)
        
        # 🔹 session_state에 새로 업로드한 파일 참조
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["last_file_key"] = file_key
        
        # 🔹 PDF 인덱싱 처리
        # 인덱싱은 file_cache에 없을 때만 수행(중복 방지)
        if file_key not in st.session_state["file_cache"]:
            st.write("Indexing your document...")
            try:
                # 임시 디렉토리 생성 및 파일 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    print("file path:", file_path)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # PDF 로더 생성 및 문서 분할                
                    if os.path.exists(temp_dir):
                        print("temp_dir:", temp_dir)
                        loader = PyPDFLoader(file_path)
                    # 파일 경로 확인 및 에러 처리
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    # 페이지 로드 및 벡터 스토어 생성
                    pages = loader.load_and_split()
                    
                    # 🔹 추가: 사이드바에서 페이지 미리보기 출력(업로드 잘 되었는지 확인용)
                    for i, p in enumerate(pages[:3]):
                        st.write(f"Page {i} content preview:", p.page_content[:200])
                        
                    # 다시 한번 더 이전에 생성된 vectorstore가 있으면 제거(메모리 참조 해제)
                    st.session_state.pop("vectorstore", None)
                        
                    # ----------------------------
                    # 🔹 VECTORSTORE 생성 분기 처리(로컬/배포)
                    # ----------------------------
                    if VECTORSTORE == "chroma":
                        st.info("Using Chroma (로컬 모드)")
                        # ✅ 완전 새로 생성되는 in-memory DB(이전 업로드 했던 pdf 기억 문제 해결)
                        chroma_client = chromadb.Client()
                        # ✅ 매번 새 collection(이전 업로드 했던 pdf 기억 문제 해결)
                        col_name = f"pdf_{uuid.uuid4().hex}"
                        # 크로마 벡터 스토어 생성
                        # 크로마에서 자연어를 벡터로, 벡터를 자연어로 처리해준다
                        # 텍스트를 추출할 수 있는 PDF 파일이어야 함
                        # 스캔된 PDF 파일인 경우 OCR을 이용해서 추출한 텍스트를 pages에 저장 후 넘겨 줘야 함
                        vectorstore = Chroma.from_documents(
                            pages,
                            UpstageEmbeddings(model="solar-embedding-1-large"),
                            client=chroma_client,
                            collection_name=col_name
                        )
                    elif VECTORSTORE == "redis":
                        st.info("Using Redis (배포 모드)")
                        # 💥 Redis index 이름: 세션ID + 파일 해시로 고정 (uuid 제거 -> 동일한 pdf면 그대로 사용하도록 함)
                        idx_name = f"pdf_index_{session_id}_{file_hash}"
                        
                        # 💥 이전 Redis index 삭제: 세션 변경이나 PDF 변경 시 누적 방지
                        prev_idx = st.session_state.get("last_index_name")
                        if prev_idx and prev_idx != idx_name:
                            try:
                                Redis.delete_index(prev_idx) # 💥 기존 index 삭제
                                print(f"Deleted previous Redis index: {prev_idx}")
                            except Exception as e:
                                print("이전 Redis 인덱스 삭제 실패:", e)
                        
                        print("Redis index name:", idx_name)
                        # 레디스 벡터 스토어 생성                        
                        vectorstore = Redis.from_documents(
                            pages,
                            UpstageEmbeddings(model="solar-embedding-1-large"),
                            redis_url=REDIS_URL,
                            index_name=idx_name
                        )
                        
                        # 💥 session_state에 새 index_name 저장
                        st.session_state["last_index_name"] = idx_name
                        
                        # 💥 Redis 메시지 히스토리 초기화 (PDF 교체 시 이전 대화 제거)
                        if "redis_history" not in st.session_state:
                            st.session_state["redis_history"] = get_redis_message_history(session_id)
                        message_history = st.session_state["redis_history"]
                    else:
                        st.error(f"지원하지 않는 VECTORSTORE 값: {VECTORSTORE}")
                        st.stop()
                    
                    # vectorstore를 세션_state에 저장하여 같은 세션에서 재사용 가능
                    st.session_state["vectorstore"] = vectorstore
                    
                    # 리트리버 생성
                    retriever = vectorstore.as_retriever(k=3) # 검색 범위를 3개로 확장 (수정)

                    # ----------------------------
                    # RAG 체인 구성
                    # ----------------------------
                    # 챗봇 생성
                    # ChatUpstage 객체 생성
                    api_key = os.getenv("UPSTAGE_API_KEY")
                    chat = ChatUpstage(api_key=api_key)

                    # 질문 재구성 프롬프트(히스토리 참고용)
                    # 히스토리 기반 리트리버 생성
                    contextualize_q_system_prompt = (
                        "이전 대화 내용과 최신 사용자 질문이 있을 때, "
                        "이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. "
                        "이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. "
                        "질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요. "
                        "특히 '페이지 번호'가 명시되어 있으면 해당 페이지 텍스트만 참조하도록 하세요."
                    )
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    # 히스토리 기반 리트리버 생성
                    history_aware_retriever = create_history_aware_retriever(
                        chat, retriever, contextualize_q_prompt
                    )

                    # 질문 답변 체인 생성
                    qa_system_prompt = (
                        "당신은 PDF에서 검색된 내용만 사용하여 질문에 답하는 보조원입니다. "
                        "PDF 내용 외에는 어떤 지식도 사용하지 마세요. "
                        "질문에 답하기 위해 검색된 내용을 반드시 사용하세요. "
                        "답변을 모르면 모른다고 말하세요. "
                        "답변은 10문장 내외로 간결하지만 핵심과 증거를 포함하세요. "
                        "질문에 특정 페이지가 명시되어 있으면 반드시 해당 페이지를 참조하세요. "
                        "## 답변 예시\n📍답변 내용:\n📍증거:\n{context}")
                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", qa_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
                    
                    # rag chain
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    # session_state에 저장
                    st.session_state["rag_chain"] = rag_chain
                    # 🔹 추가: file_cache 업데이트 (중복 방지용)
                    st.session_state["file_cache"][file_key] = True
                    
                    # 🔹 추가: Redis 모드(배포 모드)면 업로드한 PDF 파일명 저장
                    if VECTORSTORE == "redis":
                        message_history.add_user_message(f"업로드 파일: {uploaded_file.name}")
                    
                    # PDF 파일 디스플레이
                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
                    
                    # RAG 디버깅용: 페이지 내용 확인 (추가)
                    for i, p in enumerate(pages[:3]):
                        print(f"Page {i} content preview:", p.page_content[:200])
            except Exception as e:
                    st.error(f"An error occuered : {e}")
                    st.stop()

# ----------------------------
# 메인 UI
# ----------------------------
# 페이지 표시 및 타이틀 입력
st.set_page_config(page_title="Upload Text PDF And Chat",page_icon="📝")
st.title("🧑‍🚀 Askument")

# 🔹 주석 처리 부분:
# utils_rag.py 파일에 정의한 init_conversation()과 print_conversation() 메서드로 대체(코드를 더 짧고 재사용 가능하게 분리 및 StreamHandler 등과 함께 사용할 때 더 편리.) 
# 메세지 초기화
#if "messages" not in st.session_state:
#    st.session_state.messages = []
# 기존 메세지 표시
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]): # role = user, assistant
#        st.markdown(message["content"])

# 🔹 추가: utils_rag.py 파일에 정의한 메서드 활용(바로 위 주석 코드 대체) - 세션 초기화 및 기존 메시지 출력
init_conversation()
print_conversation()

# 기록하는 대화의 최대갯수 설정
MAX_MESSAGES_BEFORE_DELETION = 12

# 유저 입력 처리
if prompt := st.chat_input("질문하세요!"):
    # 🔹 추가: PDF 업로드가 안 된 상태라면 안내 메시지
    if "uploaded_file" not in st.session_state:
        st.toast("먼저 PDF 파일을 업로드해야 질문할 수 있습니다.")
    else:
        rag_chain = st.session_state["rag_chain"]
        message_history = st.session_state.get("redis_history") if VECTORSTORE == "redis" else st.session_state["memory_history"]
        
        # 세션 메시지 길이 제한
        if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
            # 이 부분에서 세션의 Max크기(지금은 12개)를 넘어가면 2개를 지우는 이유는 입력, 출력 2개이기 때문에 2개를 지우는 것! 기억!
            del st.session_state.messages[0:2]
            
        #  유저 메시지 세션 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 유저 메시지 UI 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        # Redis에 유저 질문 저장
        if VECTORSTORE == "redis":
            message_history.add_user_message(prompt)

        # AI 응답처리
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # 여기를 빈칸으로 만들었다가
            full_response = ""
            try:
                result = rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": st.session_state.messages
                })
                answer = result.get("answer", "답변을 생성하지 못했습니다.")
            except Exception as e:
                # invoke 자체가 실패했을 때
                answer = "답변을 생성하지 못했습니다."
                print("에러 발생:", e)

            # 실시간 토큰 스트리밍처럼 보이는 효과 구현을 위한 부분
            # 한 단어씩 message_placeholder에 표시됨
            for chunk in answer.split(" "):
                # 디버깅시
                # print("모델의 출력값", result["answer"])
                # print(chunk)
                full_response += chunk + " "
                time.sleep(0.05)
                # 이 부분에서 message_placeholder를 채우는 부분
                message_placeholder.markdown(full_response+ "▌")
                
            # 최종 답변 표시
            message_placeholder.markdown(full_response)
            
            # 🔹 주석 추가: AI 메시지 세션 저장
            # from langchain_core.messages import ChatMessage 클래스를 사용하여
            # messages를 객체로 활용하지 않았다. 딕셔너리로 사용했다!
            # 따라서, utis_rag.py 파일의 message 출력 코드를 그에 맞게 수정해줌!
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response, 
                "context": result.get("context", "검색된 문서 없음")
            })
            # Redis에 AI 답변 저장
            if VECTORSTORE == "redis":
                message_history.add_ai_message(full_response)

            # 검색된 context 확인 (추가)
            # 참고한 자료, 문맥 표시
            with st.expander("참고한 부분"):
                st.write(result.get("context", "검색된 문서 없음"))