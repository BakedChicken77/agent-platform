chmod +x codex_environment_setup_scripts/install_pgvector.sh
./codex_environment_setup_scripts/install_pgvector.sh
cp .env.example .env
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
uv lock
uv sync --group dev --group client

