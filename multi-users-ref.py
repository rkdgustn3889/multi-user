import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Any, List, Dict
from datetime import datetime
import logging
import re
import json
from supabase import create_client, Client
from langchain_core.documents import Document
import hashlib

# 환경 변수 로드 (로컬 개발용)
load_dotenv()

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"multi_users_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# HTTP 요청 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# Supabase 클라이언트 초기화
@st.cache_resource
def init_supabase():
    """Supabase 클라이언트를 초기화합니다."""
    # Streamlit Cloud secrets 또는 환경변수에서 읽기
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("SUPABASE_URL과 SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY)가 환경변수에 설정되어 있지 않습니다.")
        st.stop()
    
    return create_client(supabase_url, supabase_key)

supabase: Client = init_supabase()

# 비밀번호 해시 함수 (간단한 해시, 프로덕션에서는 bcrypt 등 사용 권장)
def hash_password(password: str) -> str:
    """비밀번호를 해시합니다."""
    return hashlib.sha256(password.encode()).hexdigest()

# 사용자 인증 함수
def authenticate_user(login_id: str, password: str) -> Dict:
    """사용자 인증"""
    try:
        hashed_password = hash_password(password)
        response = supabase.table("users").select("*").eq("login_id", login_id).eq("password", hashed_password).execute()
        
        if response.data and len(response.data) > 0:
            return {"success": True, "user": response.data[0]}
        else:
            return {"success": False, "message": "로그인 ID 또는 비밀번호가 올바르지 않습니다."}
    except Exception as e:
        logger.error(f"인증 중 오류: {e}")
        return {"success": False, "message": f"인증 중 오류가 발생했습니다: {str(e)}"}

# 사용자 등록 함수
def register_user(login_id: str, password: str) -> Dict:
    """새 사용자 등록"""
    try:
        # 중복 확인
        existing = supabase.table("users").select("*").eq("login_id", login_id).execute()
        if existing.data and len(existing.data) > 0:
            return {"success": False, "message": "이미 존재하는 로그인 ID입니다."}
        
        # 새 사용자 생성
        hashed_password = hash_password(password)
        response = supabase.table("users").insert({
            "login_id": login_id,
            "password": hashed_password
        }).execute()
        
        if response.data and len(response.data) > 0:
            return {"success": True, "user": response.data[0]}
        else:
            return {"success": False, "message": "사용자 등록에 실패했습니다."}
    except Exception as e:
        logger.error(f"사용자 등록 중 오류: {e}")
        return {"success": False, "message": f"사용자 등록 중 오류가 발생했습니다: {str(e)}"}

# 구분선 및 취소선 제거 함수
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    # 취소선 마크다운 제거 (~~텍스트~~ -> 텍스트)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # 여러 줄에 걸친 구분선 제거 (공백 포함)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    # 단독 라인의 구분선 제거
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    # 연속된 빈 줄 정리 (최대 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7, api_keys: Dict = None) -> Any:
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    api_keys = api_keys or {}
    
    if model_name == "gpt-5.1":
        openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API 키가 설정되어 있지 않습니다.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, api_key=openai_key)
    elif model_name == "claude-sonnet-4-5":
        from langchain_anthropic import ChatAnthropic
        anthropic_key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            st.error("Anthropic API 키가 설정되어 있지 않습니다.")
            st.stop()
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature, api_key=anthropic_key)
    elif model_name == "gemini-3-pro-preview":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_key = api_keys.get("gemini") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            st.error("Google API 키가 설정되어 있지 않습니다.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=gemini_key, temperature=temperature)
    else:
        # 기본값: gpt-5.1
        openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API 키가 설정되어 있지 않습니다.")
            st.stop()
        return ChatOpenAI(model="gpt-5.1", temperature=temperature, api_key=openai_key)

# Supabase를 사용한 벡터 스토어 클래스
class SupabaseVectorStore:
    """Supabase를 벡터 스토어로 사용하는 클래스"""
    
    def __init__(self, session_id: str, embeddings: OpenAIEmbeddings):
        self.session_id = session_id
        self.embeddings = embeddings
        self.supabase = supabase
    
    def add_documents(self, documents: List[Document], file_name: str):
        """문서를 벡터화하여 Supabase에 저장"""
        try:
            # 문서를 임베딩
            texts = [doc.page_content for doc in documents]
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # 각 문서를 Supabase에 저장
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
                # embedding이 리스트인지 확인하고, PostgreSQL vector 형식으로 변환
                if isinstance(embedding, list):
                    # 리스트를 그대로 전달 (Supabase가 자동으로 vector로 변환)
                    embedding_value = embedding
                else:
                    embedding_value = list(embedding) if hasattr(embedding, '__iter__') else embedding
                
                data = {
                    "session_id": str(self.session_id),
                    "file_name": file_name,
                    "chunk_index": idx,
                    "content": doc.page_content,
                    "metadata": json.dumps(doc.metadata, ensure_ascii=False),
                    "embedding": embedding_value
                }
                self.supabase.table("document_embeddings").insert(data).execute()
            
            logger.info(f"문서 {len(documents)}개를 Supabase에 저장했습니다.")
        except Exception as e:
            logger.error(f"문서 저장 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """유사도 검색"""
        try:
            # 쿼리 임베딩
            query_embedding = self.embeddings.embed_query(query)
            
            # Supabase에서 벡터 유사도 검색 (cosine distance)
            # PostgreSQL의 vector 확장 기능 사용
            try:
                # RPC 함수 파라미터 순서: query_embedding, match_count, session_id
                rpc_params = {
                    "query_embedding": list(query_embedding) if isinstance(query_embedding, (list, tuple)) else query_embedding,
                    "match_count": k
                }
                if self.session_id:
                    rpc_params["session_id"] = str(self.session_id)
                
                response = self.supabase.rpc("match_documents", rpc_params).execute()
                
                # 결과를 Document 객체로 변환
                documents = []
                if response.data:
                    for row in response.data:
                        metadata_str = row.get("metadata", "{}")
                        if isinstance(metadata_str, str):
                            metadata = json.loads(metadata_str)
                        else:
                            metadata = metadata_str
                        doc = Document(
                            page_content=row.get("content", ""),
                            metadata=metadata
                        )
                        documents.append(doc)
                
                return documents
            except Exception as rpc_error:
                logger.warning(f"RPC 함수 호출 실패, 대체 검색 사용: {rpc_error}")
                return self._fallback_search(query, k)
        except Exception as e:
            logger.error(f"유사도 검색 중 오류: {e}")
            # RPC 함수가 없으면 직접 SQL 쿼리
            return self._fallback_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[Document]:
        """RPC 함수가 없을 때 사용하는 대체 검색 방법"""
        try:
            import numpy as np
            
            # 쿼리 임베딩
            query_embedding = self.embeddings.embed_query(query)
            # numpy 배열로 변환
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=float)
            
            # 모든 문서를 가져와서 Python에서 유사도 계산
            response = self.supabase.table("document_embeddings")\
                .select("*")\
                .eq("session_id", str(self.session_id))\
                .execute()
            
            if not response.data:
                return []
            
            # 코사인 유사도 계산
            documents_with_scores = []
            for row in response.data:
                embedding = row.get("embedding")
                
                # embedding이 문자열인 경우 파싱
                if isinstance(embedding, str):
                    try:
                        # JSON 배열 문자열 파싱
                        import ast
                        embedding = ast.literal_eval(embedding)
                    except:
                        try:
                            # JSON 파싱 시도
                            embedding = json.loads(embedding)
                        except:
                            logger.warning(f"임베딩 파싱 실패: {type(embedding)}")
                            continue
                
                # embedding이 리스트/배열인지 확인
                if embedding and (isinstance(embedding, (list, tuple)) or hasattr(embedding, '__len__')):
                    try:
                        # numpy 배열로 변환
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding, dtype=float)
                        
                        if len(embedding) > 0 and len(embedding) == len(query_embedding):
                            # 코사인 유사도 계산
                            query_norm = np.linalg.norm(query_embedding)
                            embed_norm = np.linalg.norm(embedding)
                            if query_norm > 0 and embed_norm > 0:
                                similarity = np.dot(query_embedding, embedding) / (query_norm * embed_norm)
                            else:
                                similarity = 0.0
                            
                            # 메타데이터 파싱
                            metadata_str = row.get("metadata", "{}")
                            if isinstance(metadata_str, str):
                                try:
                                    metadata = json.loads(metadata_str)
                                except:
                                    metadata = {}
                            else:
                                metadata = metadata_str
                            
                            # Document 생성
                            doc = Document(
                                page_content=row.get("content", ""),
                                metadata=metadata
                            )
                            documents_with_scores.append((doc, similarity))
                        else:
                            continue
                    except Exception as e:
                        logger.warning(f"임베딩 처리 중 오류: {e}")
                        continue
                else:
                    continue
            
            # 유사도 순으로 정렬
            documents_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 반환
            return [doc for doc, _ in documents_with_scores[:k]]
        except ImportError:
                logger.error("numpy가 설치되어 있지 않습니다. pip install numpy를 실행해주세요.")
                # numpy 없이 간단한 거리 계산
                documents = []
                for row in response.data[:k]:
                    metadata_str = row.get("metadata", "{}")
                    if isinstance(metadata_str, str):
                        try:
                            metadata = json.loads(metadata_str)
                        except:
                            metadata = {}
                    else:
                        metadata = metadata_str
                    doc = Document(
                        page_content=row.get("content", ""),
                        metadata=metadata
                    )
                    documents.append(doc)
                return documents
        except Exception as e:
            logger.error(f"대체 검색 중 오류: {e}")
            return []

# 세션 관리 함수
def save_session(session_id: str = None, user_id: str = None) -> str:
    """현재 세션을 Supabase에 저장"""
    try:
        # 세션 제목 자동 생성 (첫 번째 질문과 답변 기반)
        title = "새 세션"
        if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
            first_question = st.session_state.chat_history[0].get("content", "")
            first_answer = st.session_state.chat_history[1].get("content", "")
            
            if first_question and first_answer:
                try:
                    api_keys = st.session_state.get("api_keys", {})
                    llm = get_llm(st.session_state.llm_model, temperature=0.7, api_keys=api_keys)
                    title_prompt = f"""
                    다음 질문과 답변을 요약하여 세션 제목을 만들어주세요.
                    제목은 최대 30자 이내로 간결하고 명확하게 작성해주세요.
                    
                    질문: {first_question[:200]}
                    답변: {first_answer[:300]}
                    
                    제목만 출력하세요 (설명 없이):
                    """
                    title_response = llm.invoke(title_prompt)
                    if hasattr(title_response, 'content'):
                        title = title_response.content.strip()
                    else:
                        title = str(title_response).strip()
                    # 제목이 너무 길면 자르기
                    if len(title) > 50:
                        title = title[:50]
                except Exception as e:
                    logger.warning(f"세션 제목 생성 실패: {e}")
                    title = first_question[:30] if first_question else "새 세션"
        
        # 세션 데이터 준비
        current_user_id = user_id or st.session_state.get("user_id")
        session_data = {
            "title": title,
            "chat_history": json.dumps(st.session_state.chat_history, ensure_ascii=False),
            "processed_files": st.session_state.processed_files,
            "llm_model": st.session_state.llm_model,
            "use_rag": st.session_state.use_rag,
            "search_model": st.session_state.search_model
        }
        
        # user_id가 있으면 추가 (NULL 허용이므로 없어도 괜찮음)
        if current_user_id:
            session_data["user_id"] = current_user_id
        
        if session_id:
            # 기존 세션 업데이트
            session_data["id"] = session_id
            response = supabase.table("sessions").update(session_data).eq("id", session_id).execute()
        else:
            # 새 세션 생성
            response = supabase.table("sessions").insert(session_data).execute()
            session_id = response.data[0]["id"]
        
        logger.info(f"세션 저장 완료: {session_id}")
        return session_id
    except Exception as e:
        logger.error(f"세션 저장 중 오류: {e}")
        st.error(f"세션 저장 중 오류가 발생했습니다: {e}")
        return None

def load_session(session_id: str):
    """Supabase에서 세션을 로드"""
    try:
        response = supabase.table("sessions").select("*").eq("id", session_id).execute()
        
        if not response.data:
            st.error("세션을 찾을 수 없습니다.")
            return False
        
        session = response.data[0]
        
        # 세션 상태 복원
        st.session_state.chat_history = json.loads(session.get("chat_history", "[]"))
        st.session_state.processed_files = session.get("processed_files", [])
        st.session_state.llm_model = session.get("llm_model", "gpt-5.1")
        st.session_state.use_rag = session.get("use_rag", False)
        st.session_state.search_model = session.get("search_model", "사용 안 함")
        st.session_state.current_session_id = session_id
        
        # 벡터 스토어 복원
        if st.session_state.processed_files:
            api_keys = st.session_state.get("api_keys", {})
            openai_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
            if not openai_key:
                st.error("OpenAI API 키가 설정되어 있지 않습니다.")
                return False
            embeddings = OpenAIEmbeddings(api_key=openai_key)
            st.session_state.vectorstore = SupabaseVectorStore(session_id, embeddings)
            
            # 검색기 생성
            class VectorRetriever:
                def __init__(self, vectorstore: SupabaseVectorStore, k: int = 10):
                    self.vectorstore = vectorstore
                    self.k = k
                
                def invoke(self, query: str):
                    return self.vectorstore.similarity_search(query, self.k)
            
            st.session_state.retriever = VectorRetriever(st.session_state.vectorstore, k=10)
        
        logger.info(f"세션 로드 완료: {session_id}")
        return True
    except Exception as e:
        logger.error(f"세션 로드 중 오류: {e}")
        st.error(f"세션 로드 중 오류가 발생했습니다: {e}")
        return False

def delete_session(session_id: str):
    """세션 삭제"""
    try:
        # 세션과 관련된 벡터 데이터도 함께 삭제됨 (CASCADE)
        supabase.table("sessions").delete().eq("id", session_id).execute()
        logger.info(f"세션 삭제 완료: {session_id}")
        return True
    except Exception as e:
        logger.error(f"세션 삭제 중 오류: {e}")
        st.error(f"세션 삭제 중 오류가 발생했습니다: {e}")
        return False

def get_all_sessions(user_id: str = None) -> List[Dict]:
    """모든 세션 목록 가져오기 (사용자별 필터링)"""
    try:
        query = supabase.table("sessions").select("id, title, created_at, updated_at")
        
        if user_id:
            query = query.eq("user_id", user_id)
        else:
            # 현재 로그인한 사용자의 세션만 가져오기
            if "user_id" in st.session_state:
                query = query.eq("user_id", st.session_state.user_id)
        
        response = query.order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        logger.error(f"세션 목록 조회 중 오류: {e}")
        return []

def get_vector_db_files(session_id: str = None) -> List[str]:
    """벡터 데이터베이스에 있는 파일명 목록 가져오기"""
    try:
        query = supabase.table("document_embeddings").select("file_name").order("file_name")
        
        if session_id:
            query = query.eq("session_id", session_id)
        else:
            if "current_session_id" in st.session_state:
                query = query.eq("session_id", st.session_state.current_session_id)
        
        response = query.execute()
        
        # 중복 제거
        files = list(set([row["file_name"] for row in response.data]))
        return files
    except Exception as e:
        logger.error(f"파일 목록 조회 중 오류: {e}")
        return []

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 멀티유저 멀티세션 RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "login_id" not in st.session_state:
    st.session_state.login_id = None

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "search_model" not in st.session_state:
    st.session_state.search_model = "사용 안 함"

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
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

# 로그인 페이지
if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; font-weight: bold; margin: 0; line-height: 1.2;">
            <span style="color: #1f77b4;">PDF 기반</span> 
            <span style="color: #ffd700;">멀티유저</span>
            <span style="color: #ff69b4;">멀티세션</span>
            <span style="color: #1f77b4;">RAG 챗봇</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["로그인", "회원가입"])
    
    with tab1:
        st.markdown('<h2 style="color: #1f77b4;">로그인</h2>', unsafe_allow_html=True)
        login_id = st.text_input("로그인 ID", key="login_input")
        password = st.text_input("비밀번호", type="password", key="password_input")
        
        if st.button("로그인", use_container_width=True):
            if login_id and password:
                result = authenticate_user(login_id, password)
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user_id = result["user"]["id"]
                    st.session_state.login_id = result["user"]["login_id"]
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error(result["message"])
            else:
                st.warning("로그인 ID와 비밀번호를 입력해주세요.")
    
    with tab2:
        st.markdown('<h2 style="color: #ff69b4;">회원가입</h2>', unsafe_allow_html=True)
        new_login_id = st.text_input("로그인 ID", key="register_login_input")
        new_password = st.text_input("비밀번호", type="password", key="register_password_input")
        confirm_password = st.text_input("비밀번호 확인", type="password", key="confirm_password_input")
        
        if st.button("회원가입", use_container_width=True):
            if new_login_id and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("비밀번호가 일치하지 않습니다.")
                else:
                    result = register_user(new_login_id, new_password)
                    if result["success"]:
                        st.success("회원가입 성공! 로그인 탭에서 로그인해주세요.")
                    else:
                        st.error(result["message"])
            else:
                st.warning("모든 필드를 입력해주세요.")
    
    st.stop()

# 메인 애플리케이션
# 제목 영역 (상단에 배치)
st.markdown("""
<div style="margin-top: -3rem; margin-bottom: 1rem;">
""", unsafe_allow_html=True)

col_title, col_user = st.columns([4, 1])

with col_title:
    # 제목 (더 크게)
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

with col_user:
    # 사용자 정보 및 로그아웃
    st.markdown(f"**로그인:** {st.session_state.login_id}")
    if st.button("로그아웃", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.login_id = None
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.current_session_id = None
        st.session_state.api_keys = {}
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# 성공/에러 메시지 표시 (PDF 처리 후)
if "show_success_message" in st.session_state and st.session_state.show_success_message:
    st.success(st.session_state.show_success_message)
    # 메시지 표시 후 플래그 제거 (한 번만 표시)
    del st.session_state.show_success_message

if "show_error_message" in st.session_state and st.session_state.show_error_message:
    st.error(st.session_state.show_error_message)
    # 메시지 표시 후 플래그 제거 (한 번만 표시)
    del st.session_state.show_error_message

# 처리된 파일이 있으면 상태 표시
if st.session_state.processed_files:
    st.info(f"📄 처리된 파일: {len(st.session_state.processed_files)}개 | RAG 사용 가능")
    if st.session_state.retriever:
        st.success("✅ 벡터 검색 준비 완료! 질문을 입력하세요.")
else:
    st.markdown("모델을 선택하고 PDF 파일을 업로드해주세요.")

# 사이드바 설정
with st.sidebar:
    # 0. API 키 입력 (상단)
    st.markdown('<h2 style="color: #1f77b4;">0. API 키 설정</h2>', unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_keys.get("openai", ""), key="openai_key_input")
    anthropic_key = st.text_input("Anthropic API Key", type="password", value=st.session_state.api_keys.get("anthropic", ""), key="anthropic_key_input")
    gemini_key = st.text_input("Google (Gemini) API Key", type="password", value=st.session_state.api_keys.get("gemini", ""), key="gemini_key_input")
    
    if st.button("API 키 저장", use_container_width=True):
        st.session_state.api_keys = {
            "openai": openai_key,
            "anthropic": anthropic_key,
            "gemini": gemini_key
        }
        st.success("API 키가 저장되었습니다!")
    
    st.markdown("---")
    
    # 1. LLM 모델 선택
    st.markdown('<h2 style="color: #1f77b4;">1. LLM 모델 선택</h2>', unsafe_allow_html=True)
    all_models = ["gpt-5.1", "claude-sonnet-4-5", "gemini-3-pro-preview"]
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = all_models[0]
    
    try:
        current_index = all_models.index(st.session_state.llm_model)
    except ValueError:
        current_index = 0
    
    selected_model = st.radio(
        "사용할 언어모델을 선택하세요",
        options=all_models,
        index=current_index,
        key='llm_model_radio'
    )
    st.session_state.llm_model = selected_model

    # 2. RAG 선택
    st.markdown('<h2 style="color: #ff69b4;">2. RAG (PDF 검색)</h2>', unsafe_allow_html=True)
    use_rag = st.radio(
        "RAG를 사용하시겠습니까?",
        [
            "사용 안 함",
            "RAG 사용"
        ],
        index=0 if not st.session_state.use_rag else 1
    )
    st.session_state.use_rag = (use_rag == "RAG 사용")

    # 3. PDF 파일 업로드
    st.markdown('<h2 style="color: #d62728;">3. PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            try:
                # 임시 파일 생성 및 처리
                temp_dir = tempfile.TemporaryDirectory()
                
                all_docs = []
                new_files = []
                
                # 각 파일 처리
                for uploaded_file in uploaded_files:
                    # 이미 처리된 파일 스킵
                    if uploaded_file.name in st.session_state.processed_files:
                        continue
                        
                    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                    
                    # 업로드된 파일을 임시 파일로 저장
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # PDF 로더 생성 및 문서 로드
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    
                    # 메타데이터에 파일 이름 추가
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name
                    
                    all_docs.extend(documents)
                    new_files.append(uploaded_file.name)
            
                if not all_docs:
                    st.warning("모든 파일이 이미 처리되었습니다.")
                    # 이미 처리된 파일이 있으면 retriever 재생성 시도
                    if st.session_state.processed_files and st.session_state.current_session_id:
                        try:
                            openai_key = st.session_state.api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
                            if not openai_key:
                                st.error("OpenAI API 키가 설정되어 있지 않습니다.")
                            else:
                                embeddings = OpenAIEmbeddings(api_key=openai_key)
                                if st.session_state.vectorstore is None:
                                    st.session_state.vectorstore = SupabaseVectorStore(
                                        st.session_state.current_session_id,
                                        embeddings
                                    )
                                
                                class VectorRetriever:
                                    def __init__(self, vectorstore: SupabaseVectorStore, k: int = 10):
                                        self.vectorstore = vectorstore
                                        self.k = k
                                    
                                    def invoke(self, query: str):
                                        return self.vectorstore.similarity_search(query, self.k)
                                
                                st.session_state.retriever = VectorRetriever(st.session_state.vectorstore, k=10)
                                logger.info("기존 파일로 retriever 재생성 완료")
                        except Exception as e:
                            logger.error(f"Retriever 재생성 실패: {e}")
                else:
                    with st.spinner(f"PDF 파일 {len(new_files)}개를 처리 중입니다... (텍스트 분할 중)"):
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        logger.info(f"총 {len(chunks)}개의 청크 생성됨")
                    
                    with st.spinner("임베딩 생성 및 벡터 DB 저장 중..."):
                        try:
                            # 임베딩 생성
                            openai_key = st.session_state.api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
                            if not openai_key:
                                st.error("OpenAI API 키가 설정되어 있지 않습니다.")
                                raise ValueError("OpenAI API 키가 필요합니다.")
                            
                            embeddings = OpenAIEmbeddings(api_key=openai_key)
                            
                            # 세션 ID 확인 또는 생성
                            if not st.session_state.current_session_id:
                                # 새 세션이면 먼저 저장
                                st.session_state.current_session_id = save_session(user_id=st.session_state.user_id)
                                logger.info(f"새 세션 생성: {st.session_state.current_session_id}")
                            
                            # Supabase 벡터 스토어 생성 또는 가져오기
                            if st.session_state.vectorstore is None:
                                st.session_state.vectorstore = SupabaseVectorStore(
                                    st.session_state.current_session_id,
                                    embeddings
                                )
                                logger.info("벡터 스토어 생성 완료")
                            
                            # 각 파일별로 문서 저장
                            file_chunks = {}
                            for chunk in chunks:
                                file_name = chunk.metadata.get("source", "unknown")
                                if file_name not in file_chunks:
                                    file_chunks[file_name] = []
                                file_chunks[file_name].append(chunk)
                            
                            # 파일별로 벡터 스토어에 추가
                            total_chunks = 0
                            for file_name, file_chunk_list in file_chunks.items():
                                logger.info(f"파일 {file_name} 처리 중: {len(file_chunk_list)}개 청크")
                                st.session_state.vectorstore.add_documents(file_chunk_list, file_name)
                                total_chunks += len(file_chunk_list)
                            
                            # 검색기 생성
                            class VectorRetriever:
                                def __init__(self, vectorstore: SupabaseVectorStore, k: int = 10):
                                    self.vectorstore = vectorstore
                                    self.k = k
                                
                                def invoke(self, query: str):
                                    return self.vectorstore.similarity_search(query, self.k)
                            
                            st.session_state.retriever = VectorRetriever(st.session_state.vectorstore, k=10)
                            logger.info(f"Retriever 생성 완료: {st.session_state.retriever is not None}")
                            
                            # 처리된 파일 목록 업데이트
                            st.session_state.processed_files.extend(new_files)
                            logger.info(f"처리된 파일 목록: {st.session_state.processed_files}")
                            
                            # 자동 저장
                            session_saved = save_session(st.session_state.current_session_id, st.session_state.user_id)
                            if session_saved:
                                logger.info("세션 저장 완료")
                            else:
                                logger.warning("세션 저장 실패")
                            
                            # 임시 디렉토리 정리
                            try:
                                temp_dir.cleanup()
                            except:
                                pass
                            
                            # 성공 메시지 (사이드바와 메인 영역 모두에 표시)
                            success_msg = f"✅ {len(new_files)}개 파일 처리가 완료되었습니다! (총 {total_chunks}개 청크)"
                            st.success(success_msg)
                            logger.info(success_msg)
                            logger.info(f"상태 확인 - retriever: {st.session_state.retriever is not None}, processed_files: {len(st.session_state.processed_files)}")
                            
                            # 메인 영역에도 성공 메시지 표시를 위한 플래그 설정
                            st.session_state.show_success_message = success_msg
                            st.rerun()
                        except Exception as e:
                            logger.error(f"파일 처리 중 오류: {e}", exc_info=True)
                            raise
                        
            except Exception as e:
                # 임시 디렉토리 정리
                try:
                    if 'temp_dir' in locals():
                        temp_dir.cleanup()
                except:
                    pass
                
                error_msg = f"파일 처리 중 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                logger.error(f"PDF 파일 처리 오류: {e}", exc_info=True)
                import traceback
                logger.error(traceback.format_exc())
                
                # 에러 메시지도 메인 영역에 표시
                st.session_state.show_error_message = error_msg
                st.rerun()

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

    # 세션 관리 섹션
    st.markdown('<h2 style="color: #1f77b4;">4. 세션 관리</h2>', unsafe_allow_html=True)
    
    # 세션 목록 가져오기
    sessions = get_all_sessions(st.session_state.user_id)
    session_titles = {s["id"]: s["title"] for s in sessions}
    
    if sessions:
        # 세션 선택 드롭다운
        selected_session_id = st.selectbox(
            "세션 선택",
            options=[None] + [s["id"] for s in sessions],
            format_func=lambda x: "새 세션" if x is None else session_titles.get(x, "알 수 없음"),
            key="session_selector"
        )
        
        # 세션 선택 시 자동 로드
        if selected_session_id and selected_session_id != st.session_state.get("selected_session_id"):
            st.session_state.selected_session_id = selected_session_id
            if load_session(selected_session_id):
                st.success("세션을 로드했습니다.")
                st.rerun()
    else:
        st.info("저장된 세션이 없습니다.")
    
    # 세션 관리 버튼
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("세션저장", use_container_width=True):
            if st.session_state.chat_history:
                session_id = save_session(st.session_state.current_session_id, st.session_state.user_id)
                if session_id:
                    st.session_state.current_session_id = session_id
                    st.success("세션이 저장되었습니다!")
                    st.rerun()
            else:
                st.warning("저장할 대화 내용이 없습니다.")
    
    with col2:
        if st.button("세션로드", use_container_width=True):
            if "selected_session_id" in st.session_state and st.session_state.selected_session_id:
                if load_session(st.session_state.selected_session_id):
                    st.success("세션을 로드했습니다.")
                    st.rerun()
            else:
                st.warning("로드할 세션을 선택해주세요.")
    
    if st.button("세션삭제", use_container_width=True):
        if st.session_state.current_session_id:
            if delete_session(st.session_state.current_session_id):
                st.success("세션이 삭제되었습니다.")
                # 상태 초기화
                st.session_state.current_session_id = None
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.session_state.vectorstore = None
                st.session_state.retriever = None
                st.rerun()
        else:
            st.warning("삭제할 세션이 없습니다.")
    
    if st.button("화면초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
    
    if st.button("vectordb", use_container_width=True):
        files = get_vector_db_files()
        if files:
            st.markdown('<h3 style="color: #ffd700;">벡터 DB 파일 목록</h3>', unsafe_allow_html=True)
            for file in files:
                st.write(f"- {file}")
        else:
            st.info("벡터 DB에 저장된 파일이 없습니다.")
    
    # 현재 설정 표시
    st.markdown('<h3 style="color: #1f77b4;">현재 설정</h3>', unsafe_allow_html=True)
    st.text(f"사용자: {st.session_state.login_id}")
    st.text(f"모델: {st.session_state.llm_model}")
    st.text(f"RAG: {'사용' if st.session_state.use_rag else '사용 안 함'}")
    if st.session_state.processed_files:
        st.text(f"처리된 파일: {len(st.session_state.processed_files)}개")
        st.text(f"대화 기록: {len(st.session_state.chat_history)}개")
    if st.session_state.current_session_id:
        st.text(f"세션 ID: {st.session_state.current_session_id[:8]}...")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # RAG 사용이 선택되었고 PDF 파일이 있는 경우
    if st.session_state.use_rag and st.session_state.retriever is not None:
        with st.spinner("PDF 기반 RAG 답변을 생성 중입니다..."):
            try:
                # RAG 검색
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                
                if not retrieved_docs:
                    response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                else:
                    # 상위 3개 문서만 사용
                    top_docs = retrieved_docs[:3]
                    
                    # 컨텍스트 구성
                    context_text = ""
                    max_context_length = 8000
                    current_length = 0
                    
                    for i, doc in enumerate(top_docs):
                        doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                        if current_length + len(doc_text) > max_context_length:
                            break
                        context_text += doc_text
                        current_length += len(doc_text)
                    
                    # 시스템 프롬프트 구성
                    system_prompt = f"""
                    질문: {prompt}
                    
                    관련 문서:
                    {context_text}
                    
                    위 문서 내용을 고려하여 질문에 답변해주세요.
                    
                    답변 형식:
                    - 답변은 반드시 제목과 본문으로 구분하여 작성하세요
                    - 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요 (최대 20자 이내 권장)
                    - 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
                    - 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
                    - 본문은 서술형으로 작성하되 존대말을 사용하세요
                    - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                    
                    주의사항:
                    - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                    - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                    - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                    - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                    - 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
                    - 마크다운 구분선이나 선을 그리는 기호를 절대 사용하지 마세요
                    - 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
                    - 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요
                    """
                    
                    # LLM으로 답변 생성 (스트리밍 모드)
                    llm = get_llm(st.session_state.llm_model, temperature=1, api_keys=st.session_state.api_keys)
                    
                    response = ""
                    with st.chat_message("assistant"):
                        stream_placeholder = st.empty()
                        # 스트리밍으로 답변 생성
                        for chunk in llm.stream(system_prompt):
                            if hasattr(chunk, 'content'):
                                chunk_text = chunk.content
                            else:
                                chunk_text = str(chunk)
                            response += chunk_text
                            # 실시간으로 표시 (구분선 제거 포함)
                            cleaned_response = remove_separators(response)
                            stream_placeholder.markdown(cleaned_response)
                    
                    # 답변에서 구분선 제거
                    response = remove_separators(response)
                
                    # 다음 질문 3개 생성
                    next_questions_prompt = f"""
                    질문자가 한 질문: {prompt}
                    
                    생성된 답변:
                    {response}
                    
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
                    
                    try:
                        next_questions_response = llm.invoke(next_questions_prompt)
                        if hasattr(next_questions_response, 'content'):
                            next_questions_text = next_questions_response.content
                        else:
                            next_questions_text = str(next_questions_response)
                        
                        # 질문들을 리스트로 파싱
                        next_questions = [q.strip() for q in next_questions_text.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                        # 최대 3개만 선택
                        next_questions = next_questions[:3]
                        
                        # 답변 끝에 다음 질문 추가
                        if next_questions:
                            response += "\n\n"
                            response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                            for i, question in enumerate(next_questions, 1):
                                response += f"{i}. {question}\n\n"
                            # 다음 질문 추가 후 다시 표시
                            with st.chat_message("assistant"):
                                st.markdown(response)
                    except Exception as e:
                        # 다음 질문 생성 실패 시 무시하고 원래 답변만 표시
                        logger.warning(f"다음 질문 생성 실패: {e}")
                    
                    # 대화 기록에 추가
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # 대화 맥락 메모리에 추가
                    st.session_state.conversation_memory.append(f"사용자: {prompt}")
                    st.session_state.conversation_memory.append(f"AI: {response}")
                    if len(st.session_state.conversation_memory) > 100:
                        st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                    
                    # 자동 저장
                    if st.session_state.current_session_id:
                        save_session(st.session_state.current_session_id, st.session_state.user_id)
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"오류가 발생했습니다: {str(e)}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
                logger.error(f"RAG 답변 생성 오류: {e}")

    # RAG 사용이 선택되지 않았거나 PDF 파일이 없는 경우
    else:
        if st.session_state.use_rag and st.session_state.retriever is None:
            # 디버깅 정보 로깅
            logger.warning(f"RAG 선택되었으나 retriever가 None입니다.")
            logger.warning(f"상태 확인 - use_rag: {st.session_state.use_rag}, retriever: {st.session_state.retriever}, processed_files: {st.session_state.processed_files}, vectorstore: {st.session_state.vectorstore is not None}")
            
            with st.chat_message("assistant"):
                if st.session_state.processed_files:
                    st.warning(f"파일은 처리되었지만 검색기가 준비되지 않았습니다. 처리된 파일: {st.session_state.processed_files}")
                    st.info("💡 파일을 다시 처리하거나 페이지를 새로고침해주세요.")
                else:
                    st.warning("RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요.")
            st.session_state.chat_history.append({"role": "assistant", "content": "RAG를 사용하려면 먼저 PDF 파일을 업로드하고 처리해주세요."})
            logger.warning("RAG 선택되었으나 PDF 파일이 없음")
        else:
            try:
                llm = get_llm(st.session_state.llm_model, temperature=1, api_keys=st.session_state.api_keys)
                direct_prompt = f"""당신은 유능한 AI 어시스턴트입니다. 반드시 한국어로 답변해주세요.

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
- 취소선(~~텍스트~~)을 사용하지 마세요. 삭제된 내용을 표시하지 마세요
- 수정된 내용을 표시할 때 취소선이나 선을 그어서 표시하지 마세요"""
                
                response = ""
                with st.chat_message("assistant"):
                    stream_placeholder = st.empty()
                    # 스트리밍으로 답변 생성
                    for chunk in llm.stream(direct_prompt):
                        if hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        else:
                            chunk_text = str(chunk)
                        response += chunk_text
                        # 실시간으로 표시 (구분선 제거 포함)
                        cleaned_response = remove_separators(response)
                        stream_placeholder.markdown(cleaned_response)
                
                # 답변에서 구분선 제거
                response = remove_separators(response)
                
                # 다음 질문 3개 생성
                try:
                    next_questions_prompt = f"""
                    질문자가 한 질문: {prompt}
                    
                    생성된 답변:
                    {response}
                    
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
                    if hasattr(next_questions_response, 'content'):
                        next_questions_text = next_questions_response.content
                    else:
                        next_questions_text = str(next_questions_response)
                    
                    next_questions = [q.strip() for q in next_questions_text.strip().split('\n') if q.strip() and not q.strip().startswith('#')]
                    next_questions = next_questions[:3]
                    
                    if next_questions:
                        response += "\n\n"
                        response += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                        for i, question in enumerate(next_questions, 1):
                            response += f"{i}. {question}\n\n"
                        # 다음 질문 추가 후 다시 표시
                        with st.chat_message("assistant"):
                            st.markdown(response)
                except Exception as e:
                    logger.warning(f"다음 질문 생성 실패: {e}")
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 자동 저장
                if st.session_state.current_session_id:
                    save_session(st.session_state.current_session_id, st.session_state.user_id)
                    
            except Exception as e:
                error_message = f"LLM 생성 중 오류 발생: {e}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                logger.error(f"LLM 답변 생성 오류: {e}")

