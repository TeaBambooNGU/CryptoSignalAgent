import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { ingestDocuments } from "../api/client";

export function IngestPage() {
  const [userId, setUserId] = useState("u001");
  const [taskId, setTaskId] = useState("");
  const [docId, setDocId] = useState("doc-btc-001");
  const [symbol, setSymbol] = useState("BTC");
  const [source, setSource] = useState("coingecko");
  const [publishedAt, setPublishedAt] = useState(new Date().toISOString());
  const [text, setText] = useState("这里粘贴通过 MCP 获得的结构化内容文本");
  const [metadataJson, setMetadataJson] = useState("{}");
  const [message, setMessage] = useState("");

  const metadataPreview = useMemo(() => {
    try {
      return JSON.parse(metadataJson) as Record<string, unknown>;
    } catch {
      return {};
    }
  }, [metadataJson]);

  const mutation = useMutation({
    mutationFn: ingestDocuments,
    onSuccess: (response) => {
      setMessage(`入库成功，chunks=${response.inserted_chunks} task_id=${response.task_id}`);
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : "入库失败");
    },
  });

  const handleSubmit = () => {
    try {
      const parsed = JSON.parse(metadataJson) as Record<string, unknown>;
      mutation.mutate({
        user_id: userId,
        task_id: taskId || undefined,
        documents: [
          {
            doc_id: docId,
            symbol,
            source,
            published_at: publishedAt,
            text,
            metadata: parsed,
          },
        ],
      });
    } catch {
      setMessage("metadata JSON 格式错误");
    }
  };

  return (
    <section className="stack-layout">
      <div className="panel panel--single">
        <header className="panel-head">
          <p className="panel-kicker">Ingest</p>
          <h2>MCP Payload Backfill</h2>
        </header>

        <div className="field-grid">
          <label className="field">
            <span>User ID</span>
            <input value={userId} onChange={(e) => setUserId(e.target.value)} />
          </label>
          <label className="field">
            <span>Task ID (Optional)</span>
            <input value={taskId} onChange={(e) => setTaskId(e.target.value)} />
          </label>
          <label className="field">
            <span>Doc ID</span>
            <input value={docId} onChange={(e) => setDocId(e.target.value)} />
          </label>
          <label className="field">
            <span>Symbol</span>
            <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </label>
          <label className="field">
            <span>Source</span>
            <input value={source} onChange={(e) => setSource(e.target.value)} />
          </label>
          <label className="field">
            <span>Published At (ISO)</span>
            <input value={publishedAt} onChange={(e) => setPublishedAt(e.target.value)} />
          </label>
        </div>

        <label className="field">
          <span>Text</span>
          <textarea rows={7} value={text} onChange={(e) => setText(e.target.value)} />
        </label>

        <label className="field">
          <span>Metadata JSON</span>
          <textarea rows={4} value={metadataJson} onChange={(e) => setMetadataJson(e.target.value)} />
        </label>

        <button type="button" className="action-button" disabled={mutation.isPending} onClick={handleSubmit}>
          {mutation.isPending ? "Ingesting..." : "提交入库"}
        </button>

        <p className="helper-line">
          metadata 预览：<code>{JSON.stringify(metadataPreview)}</code>
        </p>
        {message ? <p className="helper-line">{message}</p> : null}
      </div>
    </section>
  );
}
