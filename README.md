# Digital TaiBIFer RAG

臺灣生物多樣性知識檢索與問答後端。將文件切分、轉換為向量並存入 PostgreSQL，再透過語意檢索與生成模型提供附來源的回答。

## 功能與架構

- Manifest 驅動的文件匯入，可預覽切分結果與預估 Token 用量。
- PDF、Markdown、TXT 解析與 Recursive Chunking，保留標題階層、頁碼及來源行號。
- 文件版本管理、有效版本切換與重複索引略過。
- 向量搜尋 API，以及含引用來源、檢索摘要與模型用量的單輪問答 API。
- FAQ 分流工具，將問答素材與檢索正文分開保存。

```text
文件 → Parser → Chunker → Embedding → PostgreSQL / pgvector
                                             ↑
問題 → Embedding → 向量檢索 ────────────────────┘
                         ↓
                Prompt → Chat Model → 回答與引用來源
```

技術：Python 3.12、FastAPI、Pydantic、SQLAlchemy、Alembic、PostgreSQL 17、pgvector、OpenAI API、Docker Compose。

## 快速開始

需要 Docker Engine 與 Docker Compose，以及可使用所設定模型的 OpenAI API 金鑰。

```sh
cp .env.example .env
```

編輯 `.env`，設定 `POSTGRES_PASSWORD`、`OPENAI_API_KEY` 與模型相關參數，再執行：

```sh
docker compose up --build -d --wait
docker compose exec api alembic upgrade head
curl --fail http://localhost:8000/health
```

- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc
- OpenAPI：http://localhost:8000/openapi.json

`/health` 僅檢查 API 能否回應。未設定 API 金鑰仍可啟動服務與查看文件，模型相關 API 會回傳 `503`。

主要設定如下，完整項目見 [.env.example](.env.example)：

| 設定 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | Embedding 與回答生成使用的金鑰 |
| `EMBEDDING_MODEL` | 預設 `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | 目前資料表要求 1536 維 |
| `CHAT_MODEL` | 回答模型，需具備對應的 API 使用權限 |
| `CHAT_MAX_INPUT_TOKENS` / `CHAT_MAX_OUTPUT_TOKENS` | 回答生成的輸入與輸出預算 |
| `API_PORT` / `POSTGRES_PORT` / `ADMINER_PORT` | 主機連接埠，預設 8000／5432／8080 |

更換 Embedding 模型需重新索引文件；資料庫向量維度變更需搭配 Migration。

## 準備與匯入文件

原始文件放在專案外，整理後的 Manifest 與文件放在 Git 忽略的 `.local/knowledge/`。準備方式與 Manifest 欄位見 [知識來源說明](knowledge/README.md)，格式範例見 [manifest.example.yaml](knowledge/manifest.example.yaml)。範例清冊不包含可直接匯入的文件。

目前整理工具會分離 Markdown 的 FAQ 區塊；PDF、TXT 與 CSV 先列入待確認清冊。PDF／TXT 雖有 Parser，仍需確認內容並準備符合 Manifest 規則的輸入；CSV 尚無匯入策略。

完成資料準備後，先預覽，再匯入：

```sh
# 僅解析、切分並估算 Token，不呼叫模型或連線資料庫。
docker compose run --rm ingest --dry-run

# 匯入 Manifest 中所有 prepared 文件。
docker compose run --rm ingest

# 僅匯入指定文件，將 DOCUMENT_ID 換成 Manifest 中的 ID。
docker compose run --rm ingest --document-id DOCUMENT_ID
```

CLI 預設讀取容器中的 `/knowledge/manifest.yaml`，對應主機 `.local/knowledge/manifest.yaml`。標準輸出是 JSON 報告，執行進度輸出至標準錯誤。相同有效輸入再次匯入會回報 `unchanged`，略過 Embedding 呼叫。

文件匯入會將選取的文字片段傳送至 OpenAI；查詢會傳送問題，問答則另傳送選取的知識片段。私有原文、實際 Manifest、FAQ 素材與執行報告不隨程式碼發布。

## API

### 搜尋知識片段

```sh
curl --fail-with-body http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"TaiBIF 與 GBIF 有什麼關係？","top_k":5}'
```

回傳 Chunk 內容、文件版本、來源資訊、餘弦距離與相似度，以及問題 Embedding 用量。`score = 1 - distance`，代表向量相似度，不是答案正確率。

### 知識庫問答

```sh
curl --fail-with-body http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"TaiBIF 與 GBIF 有什麼關係？","top_k":5}'
```

回傳回答、引用來源、引用狀態、檢索摘要，以及 Embedding／生成模型用量。沒有可用上下文時會略過回答模型。完整請求與回應結構可在 Swagger UI 查看。

目前為單輪問答，不自動保存搜尋結果或對話。引用檢查驗證標記與來源的對應關係，尚未驗證來源是否支持回答內容；`grounding_checked` 為 `false`。檢索品質與回答準確度尚待評估。

## 開發與測試

程式碼複製於 Docker Image 中；修改後需重新建置：

```sh
docker compose up --build -d --wait
docker compose exec api alembic upgrade head
docker compose exec api python -m pytest -q
```

測試預設不呼叫真實 OpenAI API。資料庫整合測試使用 PostgreSQL，部分測試會建立並清理獨立測試資料庫，因此需具備建立資料庫的權限；請使用本機開發環境。

```text
backend/
  app/
    api/             # HTTP 路由與對外資料格式
    core/            # 設定與資料庫資源
    ingestion/       # 解析、切分、Embedding 與匯入 CLI
    knowledge/       # 文件準備與 Manifest
    models/          # SQLAlchemy 資料模型
    rag/             # 檢索、Prompt、生成與引用檢查
    repositories/    # 資料庫存取
  alembic/           # 資料庫 Migration
  tests/
docker/              # PostgreSQL 初始化
knowledge/           # 知識來源說明與範例清冊
scripts/             # CLI 入口
```

## 資料庫管理與日常操作

可選用 Adminer 查看 `documents`、`document_versions` 與 `document_chunks`：

```sh
docker compose up -d --wait adminer
```

開啟 http://localhost:8080，系統選 PostgreSQL、伺服器填 `db`，其餘帳密與資料庫名稱依 `.env` 設定。

```sh
docker compose logs --tail=50 api db
docker compose down
```

`docker compose down` 保留資料庫 Volume；加上 `-v` 會刪除資料。現有 Compose 設定供本機開發使用，服務連接埠綁定 `127.0.0.1`。
