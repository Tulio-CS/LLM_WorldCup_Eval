# Deploying to Coolify

This deploys a **web dashboard** (FastAPI on port 8000) backed by a **persistent
volume** for the dataset. From the dashboard you can view stats / the
quality-variability report / results, download the Excel workbooks, and trigger
`run` / `fetch` / `report` as background jobs. Coolify maps your domain to port
8000 and uses `/health` for health checks. Provider API keys are set as
**environment variables** in Coolify, never in git.

> The underlying CLI is unchanged — Coolify **Scheduled Tasks** can still exec
> `python -m fifa_forecast ...` inside the same container for cron automation.

## 1. Put the code in a Git repository

Secrets are git-ignored (`.env`, the DB, `data/`), so it's safe to push. Make
sure the fixtures CSV **is** committed.

```bash
git add -A
git status                 # confirm .env / data/ / *.db are NOT listed
git commit -m "Containerize + web dashboard for Coolify"
git remote add origin <your-git-remote>   # GitHub / GitLab / Gitea
git push -u origin main
```

## 2. Create the resource in Coolify

1. **+ New → Resource → Docker Compose**, pointed at your repo + branch.
2. Coolify reads [`docker-compose.yml`](docker-compose.yml) + [`Dockerfile`](Dockerfile).
3. **Set a domain** for the `fifa-forecast` service and point it at **port 8000**.
   Health check path: `/health`.

## 3. Set environment variables

In the resource's **Environment Variables** tab (mark secrets as secret):

```
DASHBOARD_PASSWORD=<choose-a-strong-password>   # enables the Run/Fetch buttons + locks the UI
DASHBOARD_USER=admin                            # optional, defaults to admin
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...
FOOTBALL_DATA_API_KEY=...                        # only if you use fetch
```

> **Auth model:** with `DASHBOARD_PASSWORD` set, the whole dashboard requires
> HTTP Basic login and the action buttons work. **Without it, the dashboard is
> read-only** and the money-spending endpoints are disabled — so a public URL
> can never trigger paid API calls by accident. `/health` is always open.

Leave `FIFA_DATA_DIR` / `FIFA_DATABASE_PATH` / `FIFA_MATCHES_CSV` as set in the
compose file.

## 4. Persistent storage

The named volume `fifa-data` (mounted at `/data`) is persisted by Coolify across
deploys. It holds the SQLite DB (`/data/fifa_forecasts.db`), the raw
request/response/trace archive, the Excel exports, and the matches CSV
(`/data/matches.csv`, seeded on first boot).

> Carry over existing local data (optional): after the first deploy, upload your
> local `fifa_forecasts.db` into the volume (Coolify file browser, or
> `docker cp fifa_forecasts.db <container>:/data/`).

## 5. Deploy & open

Hit **Deploy**, wait for the health check to go green, then open your domain.
You'll see the dataset stats, action buttons (if a password is set), the current
job + live log, and download links.

## 6. Automate with Scheduled Tasks (optional)

Add **Scheduled Tasks** on the resource — each is a cron expression + a command
run inside the container:

| Schedule | Command | Purpose |
|---|---|---|
| `0 * * * *` | `python -m fifa_forecast fetch --results --no-export` | pull finished scores hourly |
| `0 6 * * *` | `python -m fifa_forecast fetch --fixtures` | add next-phase games daily |
| (match days) | `python -m fifa_forecast run --date 2026-06-27` | collect forecasts |
| `30 6 * * *` | `python -m fifa_forecast report` | rebuild the analysis workbook |

> ⚠️ **`run` spends real money** (LLM API calls). Schedule it deliberately and run
> `estimate` first. `fetch` / `report` / `add-results` are cheap or free. The
> dashboard runs one background job at a time.

## 7. Updating the code

Push to git and redeploy. The `fifa-data` volume persists, so collected data
survives across deploys.

## Local test (optional)

```bash
pip install -r requirements.txt
DASHBOARD_PASSWORD=test uvicorn fifa_forecast.web:app --port 8000
# open http://localhost:8000  (login: admin / test)
```
