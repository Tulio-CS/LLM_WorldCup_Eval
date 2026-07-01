# Rodando modelos locais (Qwen, Mistral, Gemma) via Ollama

Modelos abertos rodam localmente pelo **Ollama**, que expõe uma API
OpenAI-compatível. Por isso eles reusam o mesmo adapter dos modelos OpenAI/Grok
— sem custo de API (`pricing = None`, `api_cost` fica NULL).

Já estão no catálogo (`config.py`), habilitados e selecionáveis no dashboard:

| key             | tag Ollama (model_id) | reps | tamanho aprox. (Q4) |
|-----------------|-----------------------|------|---------------------|
| `qwen-local`    | `qwen2.5:7b`          | 2    | ~4.7 GB             |
| `mistral-local` | `mistral:7b`          | 2    | ~4.4 GB             |
| `gemma-local`   | `gemma2:9b`           | 2    | ~5.4 GB             |

## Realidade da sua máquina (Ryzen 7 5800X, RX 570 4 GB, 16 GB RAM)

- **Sem GPU útil.** A Radeon RX 570 (gfx803) não é suportada pelo ROCm do Ollama
  no Windows → tudo roda em **CPU**. O 5800X (8c/16t) dá conta, mas devagar
  (~5–15 tokens/s).
- **RAM é o gargalo.** Um modelo 7B Q4 ocupa ~5 GB. Com 16 GB totais, **feche
  apps** antes de rodar (você tinha só ~3 GB livres). Rode **um modelo por vez**
  (o Ollama carrega/descarrega sozinho entre chamadas).
- Se ficar apertado ou lento, troque para tags menores no `config.py`:
  `qwen2.5:3b`, `gemma2:2b`, `mistral:7b` (não tem menor oficial) — ou até
  `llama3.2:3b`. É só mudar o `model_id`.

## 1. Instalar o Ollama (uma vez)

Baixe em <https://ollama.com/download/windows> e instale. Depois, no PowerShell:

```powershell
ollama --version          # confirma instalação
```

O instalador já sobe o servidor em `http://localhost:11434` e o mantém rodando.
Se precisar subir na mão: `ollama serve`.

## 2. Baixar os modelos (uma vez, ~15 GB no total)

```powershell
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull gemma2:9b
```

Teste rápido de que respondem:

```powershell
ollama run qwen2.5:7b "responda só: ok"
```

## 3. Rodar as previsões

Nada de chave de API — o Ollama ignora (usamos um placeholder). Garanta que o
servidor está de pé e rode só os modelos locais:

```powershell
# um jogo, pré-jogo, só os 3 locais
python -m fifa_forecast run --match-id 1 --moment pre_match `
  --model qwen-local --model mistral-local --model gemma-local
```

Pelo **dashboard**: aba Run → marque só `qwen-local / mistral-local /
gemma-local`, escolha jogos/momentos → Run. A estimativa mostra o nº de chamadas
(custo = 0).

Dicas:
- Comece com **1 jogo** para medir o tempo antes de disparar os 62.
- Se algo falhar no meio (timeout, servidor caiu), rode de novo com
  `--retry-errors` — refaz só o que deu erro/faltou, mantém os sucessos.
- Endpoint diferente (outra porta ou outra máquina na rede):
  `set OLLAMA_BASE_URL=http://192.168.0.10:11434/v1` antes de rodar.

## 4. Juntar as bases depois (`merge`)

Se você rodar os locais **na mesma base** (`fifa_forecasts.db`), não precisa de
nada — o `run_id` é único por modelo, então local e nuvem convivem na mesma
tabela.

Só precisa de merge se os locais foram parar em **outro arquivo .db** (ex.: você
rodou em outra máquina). Aí:

```powershell
# traz as linhas de outra base para a base principal (mantém o que já existe)
python -m fifa_forecast merge --from caminho/para/outra.db

# sobrescrever linhas com o mesmo run_id (ex.: refez uma que tinha errado):
python -m fifa_forecast merge --from outra.db --replace

# várias de uma vez:
python -m fifa_forecast merge --from maquina_A.db --from maquina_B.db
```

O merge é **idempotente**: por padrão usa `INSERT OR IGNORE` (não duplica, não
apaga o que já tem); `--replace` troca as linhas de mesmo `run_id`. Copia tanto
`forecast_runs` quanto `match_results`, e só as colunas presentes nas duas bases
(uma base antiga sem colunas novas ainda funde sem erro). Ao final reconstrói o
Excel (use `--no-export` para pular).

Depois de juntar, os locais entram normalmente em `report`, `evaluate` e nas
comparações do dashboard, lado a lado com os modelos de nuvem.

## 5. Cenário: app no Coolify, Ollama na sua máquina

O servidor do Coolify **não alcança** o Ollama da sua casa (está atrás do NAT,
sem IP público), então a integração é **de baixo para cima**: você roda a coleta
local na sua máquina e **envia o resultado** para o banco do Coolify. Não tente
expor o Ollama na internet para o Coolify chamar — é frágil, inseguro e lento.

**Passo a passo:**

1. Na sua máquina, clone/atualize o repositório (para ter o mesmo
   `fifa_world_cup_2026_future_matches.csv`, senão os `match_id` não batem) e
   suba o Ollama com os modelos baixados (seções 1–2).

2. Rode os locais gravando numa **base separada** (não mexe no banco de
   produção), apontando `FIFA_DB_PATH` para um arquivo local:

   ```powershell
   $env:FIFA_DB_PATH = "local_runs.db"
   python -m fifa_forecast run --moment pre_match `
     --model qwen-local --model mistral-local --model gemma-local --no-export
   ```

   Isso gera `local_runs.db` só com as previsões locais.

3. Abra o dashboard no Coolify → aba **Downloads** → **Import local runs** →
   escolha `local_runs.db` → **Upload & merge**. O servidor funde as linhas no
   banco de produção por `run_id` (`INSERT OR IGNORE` — nunca sobrescreve a
   nuvem; marque *replace* só se quiser trocar linhas de mesmo id).

   Requer `DASHBOARD_PASSWORD` setado no Coolify (o upload é uma ação protegida).

   Equivalente por linha de comando (útil para automatizar):

   ```powershell
   curl -u admin:SUA_SENHA -F "file=@local_runs.db" -F "replace=false" `
     https://SEU-APP.coolify.host/actions/merge-upload
   ```

4. Pronto — os locais aparecem no dashboard (Overview, Compare, forecasts) junto
   com os de nuvem. Rode de novo quando tiver mais jogos: o merge é idempotente,
   reenviar o mesmo `local_runs.db` não duplica nada.

> Alternativa sem upload: baixe o banco de produção do volume do Coolify para a
> sua máquina, rode `merge --from local_runs.db` localmente e devolva o arquivo
> ao volume. Funciona, mas é mais trabalhoso que o botão de upload.
