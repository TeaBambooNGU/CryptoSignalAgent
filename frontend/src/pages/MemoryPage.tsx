import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { getUserProfile, updatePreferences } from "../api/client";

export function MemoryPage() {
  const [userId, setUserId] = useState("u001");
  const [prefJson, setPrefJson] = useState("{\"watchlist\":[\"BTC\",\"ETH\"],\"style\":\"risk-first\"}");
  const [confidence, setConfidence] = useState(0.8);
  const [loadTrigger, setLoadTrigger] = useState(0);
  const [message, setMessage] = useState("");

  const profileQuery = useQuery({
    queryKey: ["profile", userId, loadTrigger],
    queryFn: () => getUserProfile(userId),
    enabled: loadTrigger > 0,
  });

  const preferenceMutation = useMutation({
    mutationFn: updatePreferences,
    onSuccess: () => {
      setMessage("偏好已保存");
      setLoadTrigger((value) => value + 1);
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : "保存失败");
    },
  });

  const handleSavePreference = () => {
    try {
      const parsed = JSON.parse(prefJson) as Record<string, unknown>;
      setMessage("");
      preferenceMutation.mutate({
        user_id: userId,
        preference: parsed,
        confidence,
      });
    } catch {
      setMessage("preference JSON 格式错误");
    }
  };

  return (
    <section className="stack-layout">
      <div className="panel panel--single">
        <header className="panel-head">
          <p className="panel-kicker">Memory</p>
          <h2>User Profile Console</h2>
        </header>

        <div className="inline-form">
          <label className="field">
            <span>User ID</span>
            <input value={userId} onChange={(e) => setUserId(e.target.value)} />
          </label>
          <button type="button" className="action-button" onClick={() => setLoadTrigger((value) => value + 1)}>
            加载画像
          </button>
        </div>

        <label className="field">
          <span>Preference JSON</span>
          <textarea rows={6} value={prefJson} onChange={(e) => setPrefJson(e.target.value)} />
        </label>

        <label className="field">
          <span>Confidence: {confidence.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
          />
        </label>

        <button type="button" className="action-button" onClick={handleSavePreference}>
          写入长期偏好
        </button>

        {message ? <p className="helper-line">{message}</p> : null}
      </div>

      <div className="panel panel--single">
        <header className="panel-head">
          <p className="panel-kicker">Profile</p>
          <h2>Memory Snapshot</h2>
        </header>

        {profileQuery.isFetching ? <p className="helper-line">加载中...</p> : null}

        {profileQuery.data ? (
          <div className="json-grid">
            <article>
              <h3>Long Term Memory</h3>
              <pre>{JSON.stringify(profileQuery.data.long_term_memory, null, 2)}</pre>
            </article>
            <article>
              <h3>Session Memory</h3>
              <pre>{JSON.stringify(profileQuery.data.session_memory, null, 2)}</pre>
            </article>
          </div>
        ) : (
          <p className="empty-state">点击“加载画像”查看当前用户记忆。</p>
        )}
      </div>
    </section>
  );
}
