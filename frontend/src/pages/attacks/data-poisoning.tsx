import React, { useState } from 'react';

const DataPoisoningDemo: React.FC = () => {
  const [user, setUser] = useState('attacker');
  const [product, setProduct] = useState('red_panda_doll');
  const [rating, setRating] = useState(5);
  const [recommendUser, setRecommendUser] = useState('attacker');
  const [result, setResult] = useState('');
  const [recommendation, setRecommendation] = useState('');
  const [loading, setLoading] = useState(false);

  const poison = async () => {
    setLoading(true);
    setResult('');
    const res = await fetch('/api/poison-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user, product, rating })
    });
    const data = await res.json();
    setResult(JSON.stringify(data, null, 2));
    setLoading(false);
  };

  const retrain = async () => {
    setLoading(true);
    setResult('');
    const res = await fetch('/api/retrain-model', { method: 'POST' });
    const data = await res.json();
    setResult(JSON.stringify(data, null, 2));
    setLoading(false);
  };

  const getRecommendation = async () => {
    setLoading(true);
    setRecommendation('');
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user: recommendUser })
    });
    const data = await res.json();
    setRecommendation(JSON.stringify(data, null, 2));
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h2>Data Poisoning Attack Demo <span role="img" aria-label="poison">🧪</span></h2>
      <p>This demo shows how an attacker can poison the training data to influence recommendations.</p>
      <div style={{ marginBottom: 16 }}>
        <b>Inject Poisoned Data:</b>
        <div>
          <label>User: <input value={user} onChange={e => setUser(e.target.value)} /></label>
          <label style={{ marginLeft: 8 }}>Product: <input value={product} onChange={e => setProduct(e.target.value)} /></label>
          <label style={{ marginLeft: 8 }}>Rating: <input type="number" value={rating} onChange={e => setRating(Number(e.target.value))} min={1} max={5} /></label>
          <button onClick={poison} disabled={loading} style={{ marginLeft: 8 }}>Poison Data</button>
        </div>
      </div>
      <div style={{ marginBottom: 16 }}>
        <b>Retrain Model:</b>
        <button onClick={retrain} disabled={loading} style={{ marginLeft: 8 }}>Retrain</button>
      </div>
      <div style={{ marginBottom: 16 }}>
        <b>Get Recommendation:</b>
        <div>
          <label>User: <input value={recommendUser} onChange={e => setRecommendUser(e.target.value)} /></label>
          <button onClick={getRecommendation} disabled={loading} style={{ marginLeft: 8 }}>Get Recommendation</button>
        </div>
      </div>
      {result && (
        <div style={{ marginTop: 16, background: '#f6f6f6', padding: 12, borderRadius: 6 }}>
          <b>Result:</b>
          <pre>{result}</pre>
        </div>
      )}
      {recommendation && (
        <div style={{ marginTop: 16, background: '#e6f7ff', padding: 12, borderRadius: 6 }}>
          <b>Recommendation:</b>
          <pre>{recommendation}</pre>
        </div>
      )}
    </div>
  );
};

export default DataPoisoningDemo; 