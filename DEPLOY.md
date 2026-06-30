# Deploy no Coolify

Guia para rodar o front (FastAPI + dashboard) no Coolify com **volume permanente**
para a base SQLite e os arquivos JSON (`raw_requests`, `raw_responses`, `traces`,
`metadata`, `exports`).

O app guarda tudo em `FIFA_DATA_DIR` (padrão `/data`) e a base em `FIFA_DB_PATH`
(padrão `/data/fifa_forecasts.db`). Basta montar um volume em `/data` e os dados
sobrevivem a redeploys.

---

## Opção A — Docker Compose (recomendada, volume já declarado)

O `docker-compose.yml` do repositório já declara o volume `fifa-data` montado em
`/data`. O Coolify cria e gerencia esse volume automaticamente.

1. **New Resource → Docker Compose** (ou *Application* com Build Pack =
   `Docker Compose`).
2. Conecte o repositório Git (público ou privado) e selecione a branch.
3. **Compose file**: `docker-compose.yml` (raiz do projeto).
4. **Environment Variables** — adicione as chaves de API (o resto já vem do
   compose):
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=...
   XAI_API_KEY=...
   ```
   > O Coolify injeta essas variáveis; o `env_file: .env` do compose é opcional
   > e pode ser ignorado quando você define as variáveis pela UI.
5. **Domain / Port**: exponha a porta **8000**.
6. **Deploy**.

O volume `fifa-data` aparece em **Storages** e persiste entre deploys.

---

## Opção B — Build por Dockerfile + Persistent Storage manual

1. **New Resource → Application → Public/Private Repository**.
2. **Build Pack**: `Dockerfile`.
3. **Port (Ports Exposes)**: `8000`.
4. **Environment Variables**:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=...
   XAI_API_KEY=...
   FIFA_DATA_DIR=/data
   FIFA_DB_PATH=/data/fifa_forecasts.db
   ```
5. **Persistent Storage** (aba **Storages** do recurso) → **+ Add**:
   - **Name**: `fifa-data`
   - **Mount Path**: `/data`
   - (deixe *Source/Host Path* vazio para usar um volume Docker nomeado,
     gerenciado pelo Coolify)
6. **Deploy**.

> Importante: sem esse mapeamento de `/data`, a base e os JSON ficam dentro do
> container e **são apagados a cada redeploy**. O passo 5 é o que garante a
> persistência.

---

## Primeira inicialização

- O volume começa **vazio**: o app cria a base SQLite e as pastas de artefatos
  automaticamente no primeiro acesso. Não há migração manual.
- Se quiser levar os dados que já existem localmente (`fifa_forecasts.db` +
  `data/`), copie-os para dentro do volume depois do primeiro deploy. Pelo
  terminal do Coolify (aba **Terminal** / **Execute Command**):
  ```sh
  # exemplo: subir um db existente para o volume
  # (faça o upload do arquivo para o servidor e copie para /data)
  cp /caminho/fifa_forecasts.db /data/fifa_forecasts.db
  ```

---

## Verificação pós-deploy

- Abra a URL pública → o dashboard deve carregar.
- `GET /api/moments` → `["pre_match","halftime","post_match"]`
- `GET /api/models` → lista dos 8 modelos.
- Rode um teste rápido pela aba **Rodar** com 1 modelo + 1 partida e veja os logs
  ao vivo.

---

## Backup

Tudo que importa está no volume `/data`:

```
/data/fifa_forecasts.db            # base principal
/data/raw_requests/  raw_responses/  traces/  metadata/   # arquivo JSON por execução
/data/exports/                     # planilha Excel
```

Para backup, basta copiar o conteúdo de `/data` (ou snapshot do volume no
Coolify / no host).
