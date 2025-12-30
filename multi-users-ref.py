"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇
Supabase를 활용한 멀티유저 로그인 및 세션 저장/로드 기능
"""

import os
import streamlit as st
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import logging
import re
import uuid
import json
import hashlib

# LangChain 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Supabase 임포트
from supabase import create_client, Client

# Anthropic, Google 임포트
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================
# Supabase 클라이언트 초기화
# ============================================
def init_supabase_client() -> Optional[Client]:
    """Supabase 클라이언트를 초기화합니다."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        return None
    
    try:
        client = create_client(supabase_url, supabase_key)
        return client
    except Exception as e:
        logger.error(f"Supabase 클라이언트 초기화 실패: {e}")
        return None

# ============================================
# 사용자 인증 함수
# ============================================
def hash_password(password: str) -> str:
    """비밀번호를 해시화합니다."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str) -> Tuple[bool, str]:
    """새 사용자를 등록합니다."""
    if not st.session_state.supabase_client:
        return False, "Supabase 연결이 필요합니다."
    
    try:
        # 비밀번호 해시화
        password_hash = hash_password(password)
        
        # users 테이블에 사용자 추가
        response = st.session_state.supabase_client.table("users").insert({
            "username": username,
            "password_hash": password_hash
        }).execute()
        
        if response.data:
            return True, "회원가입이 완료되었습니다."
        else:
            return False, "회원가입에 실패했습니다."
    except Exception as e:
        error_msg = str(e)
        if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
            return False, "이미 존재하는 사용자명입니다."
        return False, f"회원가입 실패: {error_msg[:200]}"

def login_user(username: str, password: str) -> Tuple[bool, str]:
    """사용자 로그인을 처리합니다."""
    if not st.session_state.supabase_client:
        return False, "Supabase 연결이 필요합니다."
    
    try:
        # 사용자 조회
        response = st.session_state.supabase_client.table("users").select("*").eq("username", username).execute()
        
        if not response.data or len(response.data) == 0:
            return False, "사용자명 또는 비밀번호가 올바르지 않습니다."
        
        user = response.data[0]
        password_hash = hash_password(password)
        
        if user["password_hash"] == password_hash:
            return True, user["id"]
        else:
            return False, "사용자명 또는 비밀번호가 올바르지 않습니다."
    except Exception as e:
        return False, f"로그인 실패: {str(e)[:200]}"

# ============================================
# 구분선 및 취소선 제거 함수
# ============================================
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    # 취소선 마크다운 제거
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # 여러 줄에 걸친 구분선 제거
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    # 단독 라인의 구분선 제거
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    # 연속된 빈 줄 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============================================
# LLM 모델 선택 함수
# ============================================
def get_llm(model_name: str, temperature: float = 0.7):
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    # API 키는 session_state에서 가져옴
    if model_name == "gpt-5.1":
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API 키를 입력해주세요.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, streaming=True, api_key=api_key)
    elif model_name == "claude-sonnet-4-5":
        if not ANTHROPIC_AVAILABLE:
            st.error("langchain_anthropic이 설치되지 않았습니다.")
            st.stop()
        api_key = st.session_state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Anthropic API 키를 입력해주세요.")
            st.stop()
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature, streaming=True, api_key=api_key)
    elif model_name == "gemini-3-pro-preview":
        if not GOOGLE_AVAILABLE:
            st.error("langchain_google_genai가 설치되지 않았습니다.")
            st.stop()
        api_key = st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Gemini API 키를 입력해주세요.")
            st.stop()
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", 
            google_api_key=api_key, 
            temperature=temperature
        )
    else:
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API 키를 입력해주세요.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, streaming=True, api_key=api_key)

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 세션 상태 초기화
# ============================================
if "supabase_client" not in st.session_state:
    st.session_state.supabase_client = init_supabase_client()

if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None

if "current_username" not in st.session_state:
    st.session_state.current_username = None

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore_initialized" not in st.session_state:
    st.session_state.vectorstore_initialized = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

if "embeddings" not in st.session_state:
    # OpenAI API 키가 있으면 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            st.session_state.embeddings = OpenAIEmbeddings(api_key=api_key)
        except:
            st.session_state.embeddings = None
    else:
        st.session_state.embeddings = None

# API 키 초기화
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")

# ============================================
# Supabase 헬퍼 함수
# ============================================
def get_all_sessions() -> List[Dict]:
    """현재 사용자의 모든 세션을 가져옵니다."""
    if not st.session_state.supabase_client or not st.session_state.current_user_id:
        return []
    try:
        response = st.session_state.supabase_client.table("sessions").select("*").eq("user_id", st.session_state.current_user_id).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"세션 목록 가져오기 실패: {e}")
        return []

def create_session(title: str) -> Tuple[Optional[str], Optional[str]]:
    """새 세션을 생성하고 ID를 반환합니다. (session_id, error_message)"""
    if not st.session_state.supabase_client:
        return None, "Supabase 클라이언트가 초기화되지 않았습니다."
    
    if not st.session_state.current_user_id:
        return None, "로그인이 필요합니다."
    
    # title 검증 및 정리
    if not title or not isinstance(title, str):
        title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    if len(title) > 500:
        title = title[:500]
    
    title = str(title).strip()
    if not title:
        title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    try:
        session_data = {
            "title": str(title).strip(),
            "user_id": st.session_state.current_user_id
        }
        
        if not session_data["title"]:
            session_data["title"] = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        response = st.session_state.supabase_client.table("sessions").insert(session_data).execute()
        
        if response.data and len(response.data) > 0:
            session_id = response.data[0].get("id")
            if session_id:
                return str(session_id), None
            else:
                return None, "세션 생성 응답에 ID가 없습니다."
        else:
            return None, "세션 생성 실패: 응답 데이터가 없습니다."
    except Exception as e:
        error_msg = str(e)
        logger.error(f"세션 생성 실패: {error_msg}")
        return None, f"세션 생성 실패: {error_msg[:300]}"

def save_messages(session_id: str, messages: List[Dict]) -> Optional[str]:
    """메시지를 저장합니다. 성공 시 None, 실패 시 에러 메시지 반환."""
    if not st.session_state.supabase_client:
        return "Supabase 클라이언트가 초기화되지 않았습니다."
    if not messages:
        return "저장할 메시지가 없습니다."
    
    if not session_id:
        return "세션 ID가 없습니다."
    
    try:
        uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return f"세션 ID가 올바른 UUID 형식이 아닙니다: {session_id[:50]}"
    
    try:
        # 기존 메시지 삭제 (덮어쓰기)
        st.session_state.supabase_client.table("messages").delete().eq("session_id", session_id).execute()
        
        # 새 메시지 저장
        message_data = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "").strip().lower()
            if role not in ["user", "assistant"]:
                role = "user"
            
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            
            if len(content) > 1000000:
                content = content[:1000000] + "\n\n[내용이 너무 길어 일부가 잘렸습니다]"
            
            if content is None:
                content = ""
            
            message_data.append({
                "session_id": str(session_id),
                "role": role,
                "content": content,
                "message_order": int(idx)
            })
        
        if message_data:
            chunk_size = 500
            for i in range(0, len(message_data), chunk_size):
                batch = message_data[i:i + chunk_size]
                st.session_state.supabase_client.table("messages").insert(batch).execute()
        return None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"메시지 저장 실패: {error_msg}")
        return f"메시지 저장 실패: {error_msg[:300]}"

def load_messages(session_id: str) -> List[Dict]:
    """세션의 메시지를 로드합니다."""
    if not st.session_state.supabase_client:
        return []
    try:
        response = st.session_state.supabase_client.table("messages").select("*").eq("session_id", session_id).order("message_order").execute()
        if response.data:
            return [{"role": msg["role"], "content": msg["content"]} for msg in response.data]
    except Exception as e:
        logger.error(f"메시지 로드 실패: {e}")
    return []

def save_vector_documents(session_id: str, documents: List[Document], file_name: str):
    """벡터 문서를 저장합니다."""
    if not st.session_state.supabase_client or not documents:
        return
    
    # embeddings가 없으면 생성
    if not st.session_state.embeddings:
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                st.session_state.embeddings = OpenAIEmbeddings(api_key=api_key)
            except Exception as e:
                st.error(f"OpenAI API 키가 올바르지 않습니다: {e}")
                return
        else:
            st.error("OpenAI API 키를 입력해주세요.")
            return
    
    try:
        # 기존 문서 삭제 (같은 파일명)
        st.session_state.supabase_client.table("vector_documents").delete().eq("session_id", session_id).eq("file_name", file_name).execute()
        
        # 임베딩 생성
        texts = [doc.page_content for doc in documents]
        embeddings_list = st.session_state.embeddings.embed_documents(texts)
        
        # 문서 저장
        doc_data = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
            doc_data.append({
                "session_id": session_id,
                "file_name": file_name,
                "chunk_text": doc.page_content,
                "chunk_index": idx,
                "metadata": json.dumps(doc.metadata) if doc.metadata else "{}",
                "embedding": embedding
            })
        
        # 배치로 저장 (500개씩)
        chunk_size = 500
        for i in range(0, len(doc_data), chunk_size):
            batch = doc_data[i:i + chunk_size]
            st.session_state.supabase_client.table("vector_documents").insert(batch).execute()
    except Exception as e:
        logger.error(f"벡터 문서 저장 실패: {e}")

def load_vector_documents(session_id: str) -> List[Document]:
    """세션의 벡터 문서를 로드합니다."""
    if not st.session_state.supabase_client:
        return []
    try:
        response = st.session_state.supabase_client.table("vector_documents").select("*").eq("session_id", session_id).order("chunk_index").execute()
        if response.data:
            documents = []
            for row in response.data:
                metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
                documents.append(Document(
                    page_content=row["chunk_text"],
                    metadata=metadata
                ))
            return documents
    except Exception as e:
        logger.error(f"벡터 문서 로드 실패: {e}")
    return []

def search_vector_documents(session_id: str, query: str, k: int = 5) -> List[Document]:
    """벡터 검색을 수행합니다."""
    if not st.session_state.supabase_client:
        return []
    
    # embeddings가 없으면 생성
    if not st.session_state.embeddings:
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                st.session_state.embeddings = OpenAIEmbeddings(api_key=api_key)
            except Exception as e:
                logger.error(f"Embeddings 초기화 실패: {e}")
                return []
        else:
            return []
    
    try:
        # 쿼리 임베딩 생성
        query_embedding = st.session_state.embeddings.embed_query(query)
        
        # RPC 함수 호출
        response = st.session_state.supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id
            }
        ).execute()
        
        if response.data:
            documents = []
            for row in response.data:
                metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
                documents.append(Document(
                    page_content=row["content"],
                    metadata=metadata
                ))
            return documents
    except Exception as e:
        logger.error(f"벡터 검색 실패: {e}")
    return []

def delete_session(session_id: str):
    """세션을 삭제합니다."""
    if not st.session_state.supabase_client:
        return
    try:
        st.session_state.supabase_client.table("sessions").delete().eq("id", session_id).execute()
    except Exception as e:
        logger.error(f"세션 삭제 실패: {e}")

def get_session_files(session_id: str) -> List[str]:
    """세션의 파일 목록을 가져옵니다."""
    if not st.session_state.supabase_client:
        return []
    try:
        response = st.session_state.supabase_client.table("vector_documents").select("file_name").eq("session_id", session_id).execute()
        if response.data:
            return list(set([row["file_name"] for row in response.data]))
    except Exception as e:
        logger.error(f"파일 목록 가져오기 실패: {e}")
    return []

def generate_session_title(first_question: str, first_answer: str) -> str:
    """첫 질문과 답변을 기반으로 세션 제목을 생성합니다."""
    try:
        llm = get_llm(st.session_state.llm_model, temperature=0.7)
        prompt = f"""다음 질문과 답변을 요약하여 간결한 세션 제목을 만들어주세요.

질문: {first_question}

답변: {first_answer[:500]}

요구사항:
- 제목은 최대 30자 이내로 작성
- 질문의 핵심 주제를 반영
- 한글로 작성
- 설명 없이 제목만 반환

제목:"""
        response = llm.invoke(prompt)
        title = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if len(title) > 30:
            title = title[:30]
        return title
    except Exception as e:
        logger.error(f"세션 제목 생성 실패: {e}")
        return f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

# ============================================
# CSS 스타일
# ============================================
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# 제목 영역
# ============================================
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_title, col_empty = st.columns([4, 1])

with col_title:
    st.markdown("""
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">PDF 기반</span> 
            <span style="color: #ffd700;">멀티유저</span>
            <span style="color: #ff69b4;">멀티세션</span>
            <span style="color: #1f77b4;">RAG 챗봇</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

with col_empty:
    st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 사이드바
# ============================================
with st.sidebar:
    # API 키 입력 (상단)
    st.title("🔑 API 키 설정")
    st.markdown("---")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        help="OpenAI API 키를 입력하세요"
    )
    st.session_state.openai_api_key = openai_key
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        # embeddings 업데이트
        try:
            st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_key)
        except Exception as e:
            logger.warning(f"Embeddings 초기화 실패: {e}")
    
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.anthropic_api_key,
        type="password",
        help="Anthropic API 키를 입력하세요"
    )
    st.session_state.anthropic_api_key = anthropic_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    
    gemini_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Google Gemini API 키를 입력하세요"
    )
    st.session_state.gemini_api_key = gemini_key
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    
    st.markdown("---")
    
    # 로그인/회원가입
    st.title("👤 사용자 인증")
    st.markdown("---")
    
    if not st.session_state.current_user_id:
        # 로그인되지 않은 상태
        tab1, tab2 = st.tabs(["로그인", "회원가입"])
        
        with tab1:
            login_username = st.text_input("사용자명", key="login_username")
            login_password = st.text_input("비밀번호", type="password", key="login_password")
            
            if st.button("로그인", use_container_width=True):
                if login_username and login_password:
                    success, result = login_user(login_username, login_password)
                    if success:
                        st.session_state.current_user_id = result
                        st.session_state.current_username = login_username
                        st.success("로그인 성공!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.error("사용자명과 비밀번호를 입력해주세요.")
        
        with tab2:
            reg_username = st.text_input("사용자명", key="reg_username")
            reg_password = st.text_input("비밀번호", type="password", key="reg_password")
            reg_password_confirm = st.text_input("비밀번호 확인", type="password", key="reg_password_confirm")
            
            if st.button("회원가입", use_container_width=True):
                if reg_username and reg_password:
                    if reg_password != reg_password_confirm:
                        st.error("비밀번호가 일치하지 않습니다.")
                    else:
                        success, message = register_user(reg_username, reg_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.error("모든 필드를 입력해주세요.")
    else:
        # 로그인된 상태
        st.success(f"로그인: {st.session_state.current_username}")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.current_user_id = None
            st.session_state.current_username = None
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.session_state.vectorstore_initialized = False
            st.session_state.processed_files = []
            st.rerun()
    
    st.markdown("---")
    
    # Supabase 연결 확인
    if not st.session_state.supabase_client:
        st.error("⚠️ Supabase 연결 실패")
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
        
        if not supabase_url:
            st.error("❌ SUPABASE_URL이 설정되지 않았습니다.")
        if not supabase_key:
            st.error("❌ SUPABASE_ANON_KEY가 설정되지 않았습니다.")
    
    # 로그인된 경우에만 세션 관리 및 기타 기능 표시
    if st.session_state.current_user_id:
        # LLM 모델 선택
        st.markdown('<h2 style="color: #1f77b4;">LLM 모델 선택</h2>', unsafe_allow_html=True)
        all_models = ["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"]
        selected_model = st.radio(
            "사용할 언어모델을 선택하세요",
            options=all_models,
            index=all_models.index(st.session_state.llm_model) if st.session_state.llm_model in all_models else 0,
            key='llm_model_radio'
        )
        st.session_state.llm_model = selected_model
        
        st.markdown("---")
        
        # 세션 관리
        st.markdown('<h2 style="color: #ffd700;">세션 관리</h2>', unsafe_allow_html=True)
        
        # 세션 목록 가져오기
        sessions = get_all_sessions()
        session_options = ["새 세션"] + [f"{s['title']} ({s['id'][:8]}...)" for s in sessions]
        session_ids = [None] + [s['id'] for s in sessions]
        
        if "last_selected_session_idx" not in st.session_state:
            st.session_state.last_selected_session_idx = 0
        
        selected_session_idx = st.selectbox(
            "세션 선택",
            range(len(session_options)),
            format_func=lambda x: session_options[x],
            key="session_selectbox",
            index=st.session_state.last_selected_session_idx
        )
        
        # 세션 선택이 변경되면 자동으로 로드
        if selected_session_idx != st.session_state.last_selected_session_idx:
            st.session_state.last_selected_session_idx = selected_session_idx
            if selected_session_idx > 0:
                selected_session_id = session_ids[selected_session_idx]
                st.session_state.current_session_id = selected_session_id
                
                # 메시지 로드
                messages = load_messages(selected_session_id)
                st.session_state.chat_history = messages
                
                # 벡터 문서 로드
                documents = load_vector_documents(selected_session_id)
                if documents:
                    st.session_state.vectorstore_initialized = True
                    st.session_state.processed_files = get_session_files(selected_session_id)
                else:
                    st.session_state.vectorstore_initialized = False
                    st.session_state.processed_files = []
                
                st.rerun()
        
        # 세션 로드 버튼
        if st.button("📂 세션 로드", use_container_width=True):
            if selected_session_idx > 0:
                selected_session_id = session_ids[selected_session_idx]
                st.session_state.current_session_id = selected_session_id
                
                messages = load_messages(selected_session_id)
                st.session_state.chat_history = messages
                
                documents = load_vector_documents(selected_session_id)
                if documents:
                    st.session_state.vectorstore_initialized = True
                    st.session_state.processed_files = get_session_files(selected_session_id)
                else:
                    st.session_state.vectorstore_initialized = False
                    st.session_state.processed_files = []
                
                st.success(f"세션이 로드되었습니다: {sessions[selected_session_idx-1]['title']}")
                st.rerun()
            else:
                st.warning("세션을 선택해주세요.")
        
        # 세션 저장 버튼
        if st.button("💾 세션 저장", use_container_width=True):
            if not st.session_state.supabase_client:
                st.error("⚠️ Supabase 연결이 필요합니다.")
            elif len(st.session_state.chat_history) >= 2:
                with st.spinner("세션 저장 중..."):
                    first_question = st.session_state.chat_history[0]["content"] if st.session_state.chat_history[0]["role"] == "user" else ""
                    first_answer = st.session_state.chat_history[1]["content"] if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[1]["role"] == "assistant" else ""
                    
                    if first_question and first_answer:
                        try:
                            title = generate_session_title(first_question, first_answer)
                        except Exception as e:
                            logger.warning(f"세션 제목 생성 실패: {e}")
                            title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    else:
                        title = f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    
                    if st.session_state.current_session_id:
                        session_id = st.session_state.current_session_id
                        error_msg = None
                        try:
                            uuid.UUID(str(session_id))
                        except (ValueError, TypeError):
                            st.warning(f"현재 세션 ID가 올바르지 않습니다. 새 세션을 생성합니다.")
                            session_id, error_msg = create_session(title)
                            if session_id:
                                st.session_state.current_session_id = session_id
                    else:
                        session_id, error_msg = create_session(title)
                        if session_id:
                            st.session_state.current_session_id = session_id
                    
                    if error_msg:
                        st.error(f"세션 생성 실패: {error_msg}")
                    elif session_id:
                        try:
                            uuid.UUID(str(session_id))
                        except (ValueError, TypeError):
                            st.error(f"세션 ID 형식 오류: {session_id[:50]}")
                            st.stop()
                        
                        save_error = save_messages(session_id, st.session_state.chat_history)
                        
                        if save_error:
                            st.error(f"메시지 저장 실패: {save_error}")
                        else:
                            st.success(f"✅ 세션이 저장되었습니다: {title}")
                            st.rerun()
                    else:
                        st.error("세션 저장에 실패했습니다.")
            else:
                st.warning("저장할 대화가 없습니다. 최소 1개의 질문과 답변이 필요합니다.")
        
        # 세션 삭제 버튼
        if st.button("🗑️ 세션 삭제", use_container_width=True):
            if st.session_state.current_session_id:
                delete_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.session_state.chat_history = []
                st.session_state.vectorstore_initialized = False
                st.session_state.processed_files = []
                st.success("세션이 삭제되었습니다.")
                st.rerun()
            else:
                st.warning("삭제할 세션이 없습니다.")
        
        # 화면 초기화 버튼
        if st.button("🔄 화면 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.session_state.vectorstore_initialized = False
            st.session_state.processed_files = []
            st.rerun()
        
        # VectorDB 파일 목록 보기
        if st.button("📋 VectorDB", use_container_width=True):
            if st.session_state.current_session_id:
                files = get_session_files(st.session_state.current_session_id)
                if files:
                    st.info("**현재 세션의 파일 목록:**")
                    for file in files:
                        st.write(f"- {file}")
                else:
                    st.info("현재 세션에 저장된 파일이 없습니다.")
            else:
                st.warning("세션을 먼저 로드해주세요.")
        
        st.markdown("---")
        
        # PDF 파일 업로드
        st.markdown('<h2 style="color: #ff69b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files and st.button("파일 처리하기"):
            if not st.session_state.current_session_id:
                session_id, _ = create_session(f"세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                if session_id:
                    st.session_state.current_session_id = session_id
                else:
                    st.error("세션 생성에 실패했습니다.")
                    st.stop()
            
            if st.session_state.current_session_id:
                with st.spinner("PDF 파일을 처리하는 중..."):
                    all_docs = []
                    new_files = []
                    
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = uploaded_file.name
                            all_docs.extend(docs)
                            new_files.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"파일 {uploaded_file.name} 처리 중 오류: {str(e)}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    
                    if all_docs:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        file_chunks = {}
                        for chunk in chunks:
                            file_name = chunk.metadata.get("source", "unknown.pdf")
                            if file_name not in file_chunks:
                                file_chunks[file_name] = []
                            file_chunks[file_name].append(chunk)
                        
                        for file_name, file_chunk_list in file_chunks.items():
                            save_vector_documents(st.session_state.current_session_id, file_chunk_list, file_name)
                        
                        st.session_state.processed_files.extend(new_files)
                        st.session_state.vectorstore_initialized = True
                        st.success(f"✅ {len(chunks)}개의 문서 청크가 저장되었습니다!")
                        st.rerun()
                    else:
                        st.warning("처리할 문서가 없습니다.")
    else:
        st.info("로그인 후 사용 가능한 기능입니다.")

# ============================================
# 메인 채팅 인터페이스
# ============================================
if not st.session_state.current_user_id:
    st.info("👆 사이드바에서 로그인해주세요.")
else:
    # 대화 기록 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 답변 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # RAG 사용 여부 확인
                use_rag = st.session_state.vectorstore_initialized and st.session_state.current_session_id
                
                if use_rag:
                    # RAG 검색
                    retrieved_docs = search_vector_documents(st.session_state.current_session_id, prompt, k=5)
                    
                    if retrieved_docs:
                        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])
                        
                        system_prompt = f"""너는 매우 친절한 선생님이야. 답변은 매우 쉽게 중학생 레벨에서 이해할 수 있도록 해줘. 
그러나 내용은 생략하는 것 없이 모두 답을 해줘. 모르면 모른다고 답해줘. 말투는 존대말 한글로 해줘.

다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context_text}

질문: {prompt}

답변 형식:
- 답변은 반드시 제목과 본문으로 구분하여 작성하세요
- 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
- 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
- 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
- 본문은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요"""
                    else:
                        system_prompt = f"""질문에 답변해주세요.

질문: {prompt}

답변 형식:
- 답변은 반드시 제목과 본문으로 구분하여 작성하세요
- 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
- 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
- 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
- 본문은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요"""
                else:
                    system_prompt = f"""당신은 유능한 AI 어시스턴트입니다. 반드시 한국어로 답변해주세요.

질문: {prompt}

답변 형식:
- 답변은 반드시 제목과 본문으로 구분하여 작성하세요
- 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
- 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
- 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
- 본문은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요"""
                
                # LLM으로 답변 생성 (스트리밍)
                llm = get_llm(st.session_state.llm_model, temperature=1)
                
                # 스트리밍 모드로 답변 생성
                if hasattr(llm, 'stream'):
                    for chunk in llm.stream(system_prompt):
                        if hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        else:
                            chunk_text = str(chunk)
                        full_response += chunk_text
                        cleaned_response = remove_separators(full_response)
                        message_placeholder.markdown(cleaned_response)
                else:
                    response = llm.invoke(system_prompt)
                    full_response = response.content if hasattr(response, 'content') else str(response)
                    cleaned_response = remove_separators(full_response)
                    message_placeholder.markdown(cleaned_response)
                
                # 답변 정리
                full_response = remove_separators(full_response)
                
                # 다음 질문 3개 생성
                try:
                    next_questions_prompt = f"""
질문자가 한 질문: {prompt}

생성된 답변:
{full_response}

위 질문과 답변 내용을 검토하여, 질문자가 다음에 할 수 있는 중요한 3가지 질문을 생성해주세요.

요구사항:
- 답변 내용을 더 깊이 이해하기 위한 후속 질문
- 답변에서 언급된 내용을 구체화하거나 확장하는 질문
- 관련된 다른 주제나 관점을 탐색할 수 있는 질문
- 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
- 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성

형식:
질문1
질문2
질문3

참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
"""
                    next_questions_response = llm.invoke(next_questions_prompt)
                    next_questions_text = next_questions_response.content if hasattr(next_questions_response, 'content') else str(next_questions_response)
                    next_questions = [q.strip() for q in next_questions_text.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                    next_questions = next_questions[:3]
                    
                    if next_questions:
                        full_response += "\n\n"
                        full_response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                        for i, question in enumerate(next_questions, 1):
                            full_response += f"{i}. {question}\n\n"
                        
                        message_placeholder.markdown(full_response)
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # 자동 저장 (첫 질문과 답변이 있으면)
                if len(st.session_state.chat_history) == 2 and not st.session_state.current_session_id:
                    first_question = st.session_state.chat_history[0]["content"]
                    first_answer = st.session_state.chat_history[1]["content"]
                    title = generate_session_title(first_question, first_answer)
                    session_id, _ = create_session(title)
                    if session_id:
                        st.session_state.current_session_id = session_id
                        save_messages(session_id, st.session_state.chat_history)
                elif st.session_state.current_session_id:
                    # 기존 세션에 메시지 저장 (자동 저장)
                    save_messages(st.session_state.current_session_id, st.session_state.chat_history)
            
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                logger.error(f"답변 생성 오류: {e}")

