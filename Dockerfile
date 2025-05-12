FROM pgvector/pgvector:pg16

# Optional: Set environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=gpnpostgres@2025
ENV POSTGRES_DB=postgres

# Optional: Copy your custom SQL or config files
# COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432