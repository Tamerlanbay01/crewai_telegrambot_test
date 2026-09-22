-- PostgreSQL migration for immutable S3/MinIO-backed skill packages.
-- Package object contents live in object storage; this column records the
-- deterministic SHA-256 metadata needed for idempotency and verification.

ALTER TABLE skills
    ADD COLUMN IF NOT EXISTS package_checksum VARCHAR(64);

DO $$ BEGIN
    ALTER TABLE skills
        ADD CONSTRAINT ck_skills_package_checksum_sha256
        CHECK (package_checksum IS NULL OR package_checksum ~ '^[0-9a-f]{64}$');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
