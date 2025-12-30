-- multi-users-ref 데이터베이스 스키마
-- Supabase에서 실행할 SQL 스크립트

-- 1. 사용자 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    login_id TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL, -- 실제 운영에서는 해시된 비밀번호 저장 권장
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. 세션 테이블 생성 (user_id 추가)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chat_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    processed_files TEXT[] DEFAULT ARRAY[]::TEXT[],
    llm_model TEXT,
    use_rag BOOLEAN DEFAULT false,
    search_model TEXT
);

-- 3. 벡터 임베딩을 위한 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 4. 문서 임베딩 테이블 생성
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536), -- OpenAI text-embedding-ada-002는 1536 차원
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. 인덱스 생성 (벡터 검색 성능 향상)
CREATE INDEX IF NOT EXISTS document_embeddings_session_id_idx ON document_embeddings(session_id);
CREATE INDEX IF NOT EXISTS document_embeddings_file_name_idx ON document_embeddings(file_name);
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx ON document_embeddings USING ivfflat (embedding vector_cosine_ops);

-- 6. 세션 업데이트 시간 자동 갱신 트리거 함수
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 7. 세션 테이블 업데이트 트리거 생성
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 8. 사용자 테이블 업데이트 트리거 생성
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 9. 세션 조회를 위한 인덱스
CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_created_at_idx ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS sessions_title_idx ON sessions(title);

-- 10. 사용자 조회를 위한 인덱스
CREATE INDEX IF NOT EXISTS users_login_id_idx ON users(login_id);

-- 11. 벡터 유사도 검색을 위한 RPC 함수 생성
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    session_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        document_embeddings.id,
        document_embeddings.content,
        document_embeddings.metadata,
        1 - (document_embeddings.embedding <=> query_embedding) AS similarity
    FROM document_embeddings
    WHERE (session_id IS NULL OR document_embeddings.session_id = match_documents.session_id)
    ORDER BY document_embeddings.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 12. RLS (Row Level Security) 정책 설정
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- 모든 사용자가 자신의 데이터에 접근 가능하도록 정책 생성
-- 프로덕션에서는 더 엄격한 정책이 필요할 수 있습니다
CREATE POLICY "Allow all operations on users" ON users
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow all operations on sessions" ON sessions
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow all operations on document_embeddings" ON document_embeddings
    FOR ALL
    USING (true)
    WITH CHECK (true);

