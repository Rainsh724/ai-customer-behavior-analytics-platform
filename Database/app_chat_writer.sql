-- SELECT current_user, current_database();
-- CREATE USER app_chat_writer
-- WITH PASSWORD 'ChatWriter123';
-- SELECT usename
-- FROM pg_user
-- WHERE usename = 'app_chat_writer';
-- SELECT current_database();
-- GRANT CONNECT ON DATABASE postgres TO app_chat_writer;

---- in cmd:
---- "C:\Program Files\PostgreSQL\18\bin\psql.exe" -h localhost -p 5432 -U app_chat_writer -d postgres
---- Password:ChatWriter123


-- SELECT has_database_privilege(
--     'app_chat_writer',
--     'postgres',
--     'CONNECT'
-- );
-- CREATE TABLE test_chat_connection (
--     id SERIAL PRIMARY KEY,
--     message TEXT
-- );
-- INSERT INTO test_chat_connection (message)
-- VALUES ('connection test');
-- SELECT * FROM test_chat_connection;
-- DROP TABLE test_chat_connection;