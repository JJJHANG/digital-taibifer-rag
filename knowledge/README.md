# 私有知識清冊與 FAQ 分流（S1-03）

真實文件、標題清單、來源路徑、Manifest、FAQ 與評估答案都只存在 `.local/` 或原始文件資料夾，不提交 Git。這個目錄只放操作文件與虛構的 `manifest.example.yaml`。

## 資料流

```text
外部原始目錄（唯讀、保留原檔）
  ├─ Markdown → 排除 FAQ → .local/knowledge/retrieval/
  │                     └→ .local/knowledge/evaluation/faq/
  └─ CSV / PDF / TXT → 只登錄清冊，待確認處理策略

.local/knowledge/manifest.yaml  → 每份來源的 ID、用途、路徑與待確認資訊
.local/knowledge/inventory.json → 不含內容的數量統計
```

**後續 ingestion 只能讀 manifest 中 `ingestion_status: prepared` 且 `path` 非空的項目**，以 manifest 所在目錄解析相對路徑。不可以遞迴掃描 `.local/knowledge/`，也不能直接匯入 `source_root` 的原始 Markdown，否則 FAQ 會混回檢索內容。

## 重新準備資料

先從專案根目錄建立 `.local`，將以下範例的來源與專案路徑換成本機的絕對路徑：

```sh
mkdir -p .local
docker compose build api
docker compose run --rm --no-deps \
  --user "$(id -u):$(id -g)" \
  --volume /absolute/private-source:/absolute/private-source:ro \
  --volume /absolute/project/.local:/private-data \
  api python -m app.knowledge.prepare \
  --source /absolute/private-source \
  --output /private-data/knowledge
```

使用相同的來源絕對路徑作為容器掛載位置，才能讓 Manifest 中的 `source_root` 對應主機原始目錄。輸出的 `path` 與 `evaluation_path` 則是可搬移的相對路徑。

工具不覆寫既有輸出，以免清掉人工註記。再次整理時使用新的輸出名稱，例如 `/private-data/knowledge-v2`，再人工確認要使用哪份 snapshot。原始目錄內不允許建立輸出；遇到未閉合 front matter、code fence 或未歸類的 `**Q:` 問答時停止，不產生半套 snapshot。

整理工具不是 S1-04 Parser，也不是 S1-07 Ingestion CLI；它不執行 Chunking、Embedding 或資料庫寫入。

## Manifest 欄位

| 欄位 | 用途 |
| --- | --- |
| `id` | 依來源相對路徑產生的穩定識別碼；修改內容不改 ID，改檔名則會改 ID。這與 DB 自動生成的主鍵不同 |
| `source_path` | 原始檔相對於 `source_root` 的路徑，保留來源對照 |
| `path` | 實際可供後續索引的衍生檔路徑；未納入索引的項目為 null |
| `title` | 先以檔名作為草稿標題，尚未驗證 front matter 或正文標題 |
| `source_type` | 原始格式。CSV 雖可列入清冊，但尚不支援寫入 DB 的 source_type |
| `language` / `category` | 待人工確認，不憑內容自動猜測 |
| `source_url` / `organization` / `published_at` | 待人工確認的來源資訊；內文連結不自動當成文件來源 |
| `license` | 預設 review_required，不把「可閱讀」推定為「已授權公開」 |
| `visibility` | private，與是否允許本機索引是不同事情 |
| `ingestion_status` | prepared / deferred_csv_strategy / deferred_content_review |
| `evaluation_path` | FAQ 原文素材的位置，絕不可送入索引 |
| `evaluation_status` | 有 FAQ 時為 needs_human_review，不能直接算成已確認的 evaluation dataset |
| `faq_line_ranges` | FAQ 在原始 Markdown 的一基底、含頭尾行號範圍；不是衍生檔的行號 |
| `faq_question_markers` | 偵測到的粗略 Q 標記數量，不是已通過人工確認的題數 |

`prepared` 只表示完成了 FAQ 分流，不表示授權與來源資訊均已確認。S1-04 Parser 處理衍生檔時，也需保留來源對照，不能將衍生檔行號誤報成原始文件行號。

CSV 目前只讀取表頭與資料列數，沒有把每列直接當成文件。正式處理前需確認內容欄位、內部編輯註記、空白列、術語鍵值，以及一列要如何轉成可檢索文字。
PDF / TXT 也先停在待確認，避免同一份知識中重複出現的 FAQ 或摘要影響評估。

## FAQ 為什麼要分開

如果把同一組「問題＋答案」放進向量庫，再用原問題測試，檢索可能只是找回完整答案，導致測試表現過度樂觀。這是資料洩漏的一種形式。

但正文本來就應包含回答所需的知識，不需要把所有相關事實一起刪除。這裡隔離的是已整理的題目與標準答案，讓測試檢查系統是否能從正文找到依據並回答。

這並不保證 FAQ 與其他文件完全沒有語意或文字重複，也不能保證模型預訓練時從未看過相關內容。

到 S3-01 時再從素材挑題，逐題確認：

1. 問題清楚，答案正確且沒有過時。
2. 排除 FAQ 後，索引正文仍有足夠證據；否則補正文或標成應拒答題。
3. 註記支持答案的 document / section；Chunk 建立後再補標準 Chunk ID。
4. 區分回答品質評估（事實、引用、拒答）與檢索評估（Recall@K、MRR）。只有答案文字，還不足以計算檢索指標。

若根據這些題目反覆調整參數，這組題目就成為開發集；之後仍應保留一組未用來調參的測試題。

## Git 保護範圍

專案 `.gitignore` 排除 `.local/`、`knowledge/raw/` 與 `evaluation/private/`；工具也在每個輸出 snapshot 寫入忽略全部檔案的 `.gitignore`。
原始文件不會複製進可提交的目錄，真實 FAQ 不會放進程式測試 fixture。已經被 Git 追蹤的檔案不會因新增 ignore 規則自動解除追蹤；本步沒有新增任何追蹤或 commit。
