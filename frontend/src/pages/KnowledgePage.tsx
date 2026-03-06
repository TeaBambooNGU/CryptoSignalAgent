import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ChangeEvent, useMemo, useState } from "react";
import {
  createKnowledgeDocument,
  deleteKnowledgeDocument,
  listKnowledgeDocuments,
  uploadKnowledgeDocument,
} from "../api/client";

const DOC_TYPE_OPTIONS = [
  { value: "whitepaper", label: "Whitepaper" },
  { value: "research_report", label: "Research Report" },
  { value: "governance_proposal", label: "Governance Proposal" },
  { value: "methodology", label: "Methodology" },
  { value: "internal_memo", label: "Internal Memo" },
  { value: "event_note", label: "Event Note" },
];

const FILE_SUPPORT_MATRIX = [
  { label: ".md / .txt", detail: "当前可自动读取文本并预填正文" },
  { label: ".pdf / .docx", detail: "当前可通过 `/v1/knowledge/upload` 直传并解析入库" },
  { label: "Metadata", detail: "标题、来源、标签、语言、symbols 全量跟随入库" },
];

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function buildFingerprint(file: File): string {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

async function readTextFile(file: File): Promise<string> {
  return await file.text();
}

export function KnowledgePage() {
  const queryClient = useQueryClient();
  const [userId, setUserId] = useState("u001");
  const [taskId, setTaskId] = useState("");
  const [kbId, setKbId] = useState("macro-research");
  const [title, setTitle] = useState("Bitcoin Liquidity Map Q1");
  const [primarySymbol, setPrimarySymbol] = useState("BTC");
  const [symbolsInput, setSymbolsInput] = useState("BTC,ETH");
  const [source, setSource] = useState("messari");
  const [docType, setDocType] = useState("research_report");
  const [language, setLanguage] = useState("zh-CN");
  const [tagsInput, setTagsInput] = useState("liquidity,etf,macro");
  const [publishedAt, setPublishedAt] = useState(new Date().toISOString());
  const [text, setText] = useState("这里粘贴长期有效的背景资料正文，后续将进入 knowledge evidence 召回。");
  const [extraMetadataJson, setExtraMetadataJson] = useState('{"author":"Research Desk"}');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");

  const parsedSymbols = useMemo(
    () => symbolsInput.split(",").map((item) => item.trim()).filter(Boolean),
    [symbolsInput],
  );

  const parsedTags = useMemo(
    () => tagsInput.split(",").map((item) => item.trim()).filter(Boolean),
    [tagsInput],
  );

  const extraMetadata = useMemo(() => {
    try {
      return JSON.parse(extraMetadataJson) as Record<string, unknown>;
    } catch {
      return null;
    }
  }, [extraMetadataJson]);

  const metadataPreview = useMemo(() => {
    return {
      title,
      doc_type: docType,
      tags: parsedTags,
      language,
      kb_id: kbId,
      symbols: parsedSymbols,
      file_name: selectedFile?.name,
      checksum: selectedFile ? buildFingerprint(selectedFile) : undefined,
      uploaded_by: userId,
      ingest_mode: "knowledge_manual",
      ...(extraMetadata ?? {}),
    } satisfies Record<string, unknown>;
  }, [docType, extraMetadata, kbId, language, parsedSymbols, parsedTags, selectedFile, title, userId]);

  const payloadPreview = useMemo(() => {
    const documentPayload = {
      user_id: userId,
      task_id: taskId || undefined,
      document: {
        title,
        source,
        doc_type: docType,
        symbols: parsedSymbols,
        tags: parsedTags,
        text,
        kb_id: kbId,
        language,
        published_at: publishedAt,
        metadata: metadataPreview,
      },
    };
    if (!selectedFile) {
      return documentPayload;
    }
    return {
      ...documentPayload,
      upload_file: selectedFile.name,
      content_type: selectedFile.type || "application/octet-stream",
    };
  }, [
    docType,
    kbId,
    language,
    metadataPreview,
    parsedSymbols,
    parsedTags,
    publishedAt,
    selectedFile,
    source,
    taskId,
    text,
    title,
    userId,
  ]);

  const documentsQuery = useQuery({
    queryKey: ["knowledge-documents"],
    queryFn: listKnowledgeDocuments,
  });

  const mutation = useMutation({
    mutationFn: async () => {
      if (selectedFile) {
        const formData = new FormData();
        formData.append("user_id", userId);
        formData.append("title", title);
        formData.append("source", source);
        formData.append("doc_type", docType);
        formData.append("symbols", parsedSymbols.join(","));
        formData.append("tags", parsedTags.join(","));
        formData.append("kb_id", kbId);
        formData.append("language", language);
        formData.append("published_at", publishedAt);
        formData.append("metadata_json", JSON.stringify(metadataPreview));
        formData.append("file", selectedFile);
        return uploadKnowledgeDocument(formData);
      }

      return createKnowledgeDocument({
        user_id: userId,
        task_id: taskId || undefined,
        document: {
          title,
          source,
          doc_type: docType,
          symbols: parsedSymbols,
          tags: parsedTags,
          text,
          kb_id: kbId,
          language,
          published_at: publishedAt,
          metadata: metadataPreview,
        },
      });
    },
    onSuccess: (response) => {
      setMessage(
        `知识库入库成功，doc_id=${response.document.doc_id} chunks=${response.inserted_chunks} task_id=${response.task_id}`,
      );
      void queryClient.invalidateQueries({ queryKey: ["knowledge-documents"] });
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : "知识库入库失败");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteKnowledgeDocument,
    onSuccess: () => {
      setMessage("知识库文档已删除");
      void queryClient.invalidateQueries({ queryKey: ["knowledge-documents"] });
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : "删除知识库文档失败");
    },
  });

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setSelectedFile(nextFile);
    if (!nextFile) {
      return;
    }

    const baseName = nextFile.name.replace(/\.[^.]+$/, "");
    const slug = slugify(baseName);
    if (slug) {
      if (!title || title === "Bitcoin Liquidity Map Q1") {
        setTitle(baseName);
      }
    }

    if (!primarySymbol && parsedSymbols[0]) {
      setPrimarySymbol(parsedSymbols[0]);
    }

    const lowerName = nextFile.name.toLowerCase();
    const isTextLike = lowerName.endsWith(".md") || lowerName.endsWith(".txt");
    if (isTextLike) {
      const nextText = await readTextFile(nextFile);
      setText(nextText);
      setMessage(`已读取 ${nextFile.name}，正文已自动填充，可继续补充元信息。`);
      return;
    }

    setMessage(`已记录文件 ${nextFile.name}。提交时将走知识库文件上传接口。`);
  };

  const handleSubmit = () => {
    if (!text.trim()) {
      setMessage("请提供可入库的知识正文内容");
      return;
    }
    if (!extraMetadata) {
      setMessage("扩展 Metadata JSON 格式错误");
      return;
    }

    mutation.mutate();
  };

  return (
    <section className="stack-layout knowledge-layout">
      <div className="panel knowledge-hero">
        <header className="knowledge-hero__head">
          <div>
            <p className="panel-kicker">知识库</p>
            <h2>知识库上传</h2>
          </div>
          <p className="knowledge-hero__lede">
            把白皮书、历史研报、治理提案与方法论文档沉淀为长期知识证据，而不是再次混入实时信号层。
          </p>
        </header>

        <div className="knowledge-hero__metrics" aria-label="知识库能力摘要">
          <article>
            <span>目标集合</span>
            <strong>knowledge_chunks</strong>
          </article>
          <article>
            <span>当前接口</span>
            <strong>/v1/knowledge/documents</strong>
          </article>
          <article>
            <span>文件上传接口</span>
            <strong>/v1/knowledge/upload</strong>
          </article>
          <article>
            <span>召回角色</span>
            <strong>知识证据</strong>
          </article>
        </div>
      </div>

      <div className="knowledge-grid">
        <div className="panel knowledge-dock">
          <header className="panel-head">
            <p className="panel-kicker">上传入口</p>
            <h2>上传舱</h2>
          </header>

          <label className="upload-dropzone" htmlFor="knowledge-file-input">
            <input
              id="knowledge-file-input"
              type="file"
              accept=".md,.txt,.pdf,.docx"
              onChange={handleFileChange}
            />
            <span className="upload-dropzone__eyebrow">拖拽、选择或替换文件</span>
            <strong>{selectedFile ? selectedFile.name : "拖入知识文档，或点击选择文件"}</strong>
            <span>
              {selectedFile
                ? `${selectedFile.type || "unknown"} · ${formatBytes(selectedFile.size)}`
                : "支持 .md / .txt / .pdf / .docx"}
            </span>
          </label>

          <div className="support-grid" aria-label="文件支持状态">
            {FILE_SUPPORT_MATRIX.map((item) => (
              <article key={item.label} className="support-card">
                <strong>{item.label}</strong>
                <p>{item.detail}</p>
              </article>
            ))}
          </div>

          <div className="field-grid">
            <label className="field">
              <span>用户 ID</span>
              <input value={userId} onChange={(event) => setUserId(event.target.value)} placeholder="u001" />
            </label>
            <label className="field">
              <span>任务 ID（可选）</span>
              <input value={taskId} onChange={(event) => setTaskId(event.target.value)} placeholder="kb-task-001" />
            </label>
            <label className="field">
              <span>标题</span>
              <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Bitcoin Liquidity Map" />
            </label>
            <label className="field">
              <span>主标的</span>
              <input value={primarySymbol} onChange={(event) => setPrimarySymbol(event.target.value.toUpperCase())} placeholder="BTC" />
            </label>
            <label className="field">
              <span>关联标的（逗号分隔）</span>
              <input value={symbolsInput} onChange={(event) => setSymbolsInput(event.target.value.toUpperCase())} placeholder="BTC,ETH,SOL" />
            </label>
            <label className="field">
              <span>来源</span>
              <input value={source} onChange={(event) => setSource(event.target.value)} placeholder="messari" />
            </label>
            <label className="field">
              <span>知识库 ID</span>
              <input value={kbId} onChange={(event) => setKbId(event.target.value)} placeholder="macro-research" />
            </label>
            <label className="field">
              <span>文档类型</span>
              <select value={docType} onChange={(event) => setDocType(event.target.value)}>
                {DOC_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>语言</span>
              <input value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="zh-CN" />
            </label>
            <label className="field">
              <span>标签（逗号分隔）</span>
              <input value={tagsInput} onChange={(event) => setTagsInput(event.target.value)} placeholder="macro,liquidity,btc" />
            </label>
            <label className="field">
              <span>发布时间（ISO）</span>
              <input value={publishedAt} onChange={(event) => setPublishedAt(event.target.value)} />
            </label>
          </div>

          <label className="field">
            <span>知识正文</span>
              <textarea
              rows={12}
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="粘贴知识正文，提交后会直接进入知识证据链路。"
            />
          </label>

          <label className="field">
            <span>扩展元信息 JSON</span>
            <textarea
              rows={4}
              value={extraMetadataJson}
              onChange={(event) => setExtraMetadataJson(event.target.value)}
              placeholder='{"author":"Research Desk"}'
            />
          </label>

          <button type="button" className="action-button" disabled={mutation.isPending} onClick={handleSubmit}>
            {mutation.isPending ? "上传中..." : "提交到知识库入口"}
          </button>

          {message ? <p className="helper-line helper-line--status">{message}</p> : null}
        </div>

        <div className="knowledge-sidecar">
          <div className="panel knowledge-panel">
            <header className="panel-head">
              <p className="panel-kicker">元信息</p>
              <h2>元信息编排</h2>
            </header>
            <div className="token-list" aria-label="当前标签与标的">
              {parsedSymbols.map((symbol) => (
                <span key={symbol} className="token-chip token-chip--accent">
                  {symbol}
                </span>
              ))}
              {parsedTags.map((tag) => (
                <span key={tag} className="token-chip">
                  #{tag}
                </span>
              ))}
              {!parsedSymbols.length && !parsedTags.length ? <p className="empty-state">添加 symbols 和 tags 以强化召回语义。</p> : null}
            </div>

            <dl className="kv-list">
              <div>
                <dt>知识角色</dt>
                <dd>作为背景证据进入 `retrieve_knowledge_evidence`</dd>
              </div>
              <div>
                <dt>接口状态</dt>
                <dd>当前已切到 `/v1/knowledge/*`，与双库职责保持一致</dd>
              </div>
              <div>
                <dt>文件指纹</dt>
                <dd>{selectedFile ? buildFingerprint(selectedFile) : "等待选择文件"}</dd>
              </div>
            </dl>
          </div>

          <div className="panel knowledge-panel">
            <header className="panel-head">
              <p className="panel-kicker">请求预览</p>
              <h2>即将写入的请求</h2>
            </header>
            <pre className="payload-preview">{JSON.stringify(payloadPreview, null, 2)}</pre>
          </div>

          <div className="panel knowledge-panel knowledge-panel--signal-split">
            <header className="panel-head">
              <p className="panel-kicker">信号 / 知识 分层</p>
              <h2>职责边界</h2>
            </header>
            <div className="split-list">
              <article>
                <span>信号向量库</span>
                <p>实时事实，本轮分析直接消费，不做自我召回。</p>
              </article>
              <article>
                <span>知识向量库</span>
                <p>长期资料，供后续研报生成时检索引用与结构性解释。</p>
              </article>
            </div>
          </div>

          <div className="panel knowledge-panel">
            <header className="panel-head">
              <p className="panel-kicker">知识索引</p>
              <h2>已入库文档</h2>
            </header>
            {documentsQuery.isLoading ? <p className="helper-line">加载中...</p> : null}
            {documentsQuery.isError ? <p className="helper-line">加载文档列表失败</p> : null}
            <div className="split-list">
              {documentsQuery.data?.items.map((item) => (
                <article key={item.doc_id}>
                  <span>{item.title}</span>
                  <p>
                    {item.doc_type} · {item.symbols.join(",") || "GENERAL"} · chunks={item.chunk_count}
                  </p>
                  <button
                    type="button"
                    className="action-button"
                    disabled={deleteMutation.isPending}
                    onClick={() => deleteMutation.mutate(item.doc_id)}
                  >
                    删除
                  </button>
                </article>
              ))}
              {!documentsQuery.data?.items.length && !documentsQuery.isLoading ? (
                <p className="empty-state">还没有知识库文档，先上传一份试试。</p>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
