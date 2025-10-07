CREATE TABLE IF NOT EXISTS user_files (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NULL,
    tenant_id TEXT NULL,
    original_name TEXT NOT NULL,
    mime TEXT NOT NULL,
    size BIGINT NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    indexed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_user_files_user_thread
    ON user_files (user_id, thread_id);

CREATE INDEX IF NOT EXISTS idx_user_files_user_created_at
    ON user_files (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_files_user_sha_name
    ON user_files (user_id, sha256, original_name);
