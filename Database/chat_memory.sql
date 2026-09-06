-- CREATE TABLE IF NOT EXISTS public.chat_memory (
--     chat_id TEXT PRIMARY KEY,
--     messages JSONB NOT NULL,
--     updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
-- );

-- SELECT current_user, current_database();

-- SELECT
--     grantee,
--     privilege_type
-- FROM information_schema.role_table_grants
-- WHERE table_schema = 'public'
--   AND table_name = 'chat_memory';

-- GRANT USAGE ON SCHEMA public TO postgres;

-- GRANT SELECT, INSERT, UPDATE, DELETE
-- ON TABLE public.chat_memory
-- TO postgres;

-- SELECT
--     table_schema,
--     table_name
-- FROM information_schema.tables
-- WHERE table_name = 'chat_memory';

-- SELECT * FROM public.chat_memory;
-- INSERT INTO public.chat_memory (chat_id, messages)
-- VALUES ('test_chat', '[]'::jsonb);
-- SELECT * FROM public.chat_memory;

-- DELETE FROM public.chat_memory
-- WHERE chat_id = 'test_chat';

-- CREATE USER app_chat_writer WITH PASSWORD 'ChatWriter123';

-- GRANT USAGE ON SCHEMA public TO app_chat_writer;

-- GRANT SELECT, INSERT, UPDATE, DELETE
-- ON TABLE public.chat_memory
-- TO app_chat_writer;

-- SET ROLE app_chat_writer;

-- SELECT current_user, current_database();

-- SELECT *
-- FROM public.chat_memory
-- LIMIT 5;

-- RESET ROLE;