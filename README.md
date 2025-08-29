# RAG Chatbot with PDF

이 프로젝트는 사용자가 업로드한 PDF를 기반으로 질문에 답하는 Streamlit 기반 RAG(Retrieval-Augmented Generation) 챗봇입니다.  
LangChain, Chroma, Upstage LLM을 활용하여 PDF 내용을 검색하고, 그 내용을 근거로 답변을 생성합니다.

## 주요 기능
- PDF 업로드 및 페이지별 내용 확인
- PDF 내용 기반 질문 답변 (검색 증강 생성)
- 질문에 특정 페이지가 명시되면 해당 페이지 참조
- 채팅 내역 저장 및 관리
- Streamlit UI로 실시간 대화

## 제한 사항
- 현재는 텍스트 기반 PDF만 지원하며, 스캔된 이미지 PDF는 처리할 수 없음
- 스캔 PDF 지원을 위해서는 OCR(예: Tesseract, AWS Textract 등) 통합이 필요

## 향후 발전 방향
- 이미지 기반 PDF 처리 및 OCR 연동
- 멀티 PDF 동시 검색 및 비교
- 질문 답변 품질 향상을 위한 문맥 추적 개선
- PDF 내 그림, 표 등 구조화된 정보 활용

## 요구 사항
- Python 3.11.x (다른 버전은 호환성 미확인)
- pip 최신 버전 권장

## 설치 및 실행

1. 리포지토리 클론 및 가상환경 생성
```bash
git clone <리포지토리 URL>
cd rag-chatbot-with-pdf
# 가상환경의 파이썬 버전 꼭 체크!
python -m venv <your venv>
python --version
# 가상환경 활성화 명령 Windows
<your venv>\Scripts\activate
# 가상환경 활성화 명령 macOS/Linux
source <your venv>/bin/activate
```
2. pip 업그레이드
```bash
python -m pip install --upgrade pip
```
3. 의존성 설치
```bash
pip install -r requirements.txt
```
4. 환경변수 설정
포함되어 있는 .env.example 파일명을 .env로 수정 후
UPSTAGE, LANGCHAIN에서 생성한 your API key를 입력

5. Streamlit 실행
```bash
streamlit run solar_rag.py
```
## 사용방법
1. 사이드바에서 PDF 업로드
2. 업로드 후 챗 인터페이스에서 질문 입력
3. 챗봇이 PDF 내용을 기반으로 답변 제공
4. 필요시 채팅 초기화 가능

## 라이선스
MIT License