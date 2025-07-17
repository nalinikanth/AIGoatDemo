import React, { useState } from 'react';

const ModelExtractionDemo: React.FC = () => {
  const [user, setUser] = useState('attacker');
  const [recommendation, setRecommendation] = useState('');
  const [log, setLog] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const queryModel = async () => {
    setLoading(true);
    setRecommendation('');
    await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user })
    })
      .then(res => res.json())
      .then(data => setRecommendation(JSON.stringify(data, null, 2)));
    setLoading(false);
    fetchLog();
  };

  const fetchLog = async () => {
    const res = await fetch('/api/model-extraction-log');
    const data = await res.json();
    setLog(data);
  };

  const downloadLog = () => {
    const blob = new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_extraction_log.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearLog = async () => {
    await fetch('/api/model-extraction-log/clear', { method: 'POST' });
    setLog([]);
  };

  React.useEffect(() => {
    fetchLog();
  }, []);

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h2>Model Extraction (Stealing) Attack Demo <span role="img" aria-label="robot">🤖</span></h2>
      <p>This demo shows how an attacker can reconstruct a model by collecting input/output pairs.</p>
      <div style={{ marginBottom: 16 }}>
        <label>User: <input value={user} onChange={e => setUser(e.target.value)} /></label>
        <button onClick={queryModel} disabled={loading} style={{ marginLeft: 8 }}>{loading ? 'Querying...' : 'Query Model'}</button>
      </div>
      {recommendation && (
        <div style={{ marginTop: 16, background: '#f6f6f6', padding: 12, borderRadius: 6 }}>
          <b>Recommendation:</b>
          <pre>{recommendation}</pre>
        </div>
      )}
      <div style={{ marginTop: 24 }}>
        <button onClick={downloadLog} disabled={log.length === 0}>Download Log</button>
        <button onClick={clearLog} style={{ marginLeft: 8 }}>Clear Log</button>
      </div>
      <div style={{ marginTop: 16 }}>
        <b>Extraction Log:</b>
        <pre style={{ maxHeight: 300, overflow: 'auto', background: '#f9f9f9', padding: 8, borderRadius: 4 }}>{JSON.stringify(log, null, 2)}</pre>
      </div>
    </div>
  );
};

export default ModelExtractionDemo; 