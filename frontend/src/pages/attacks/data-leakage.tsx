import React, { useState } from 'react';

const DataLeakageDemo: React.FC = () => {
  const [secret, setSecret] = useState('SECRET-1234-5678-9012');
  const [userInput, setUserInput] = useState('Do you know any secrets?');
  const [response, setResponse] = useState('');
  const [injectResult, setInjectResult] = useState('');
  const [loading, setLoading] = useState(false);

  const injectSecret = async () => {
    setInjectResult('');
    const res = await fetch('/api/inject-secret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ secret })
    });
    const data = await res.json();
    setInjectResult(`Injected: ${data.secret}`);
  };

  const askAI = async () => {
    setLoading(true);
    setResponse('');
    const res = await fetch('/api/ai-assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_input: userInput })
    });
    const data = await res.json();
    setResponse(data.response || data.error || 'No response');
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h2>Sensitive Data Leakage Attack Demo <span role="img" aria-label="lock">🔓</span></h2>
      <p>This demo shows how a model can leak sensitive data it was trained on.</p>
      <div style={{ marginBottom: 16 }}>
        <label>Secret to inject: <input value={secret} onChange={e => setSecret(e.target.value)} /></label>
        <button onClick={injectSecret} style={{ marginLeft: 8 }}>Inject Secret</button>
        {injectResult && <span style={{ marginLeft: 12, color: 'green' }}>{injectResult}</span>}
      </div>
      <div style={{ marginBottom: 16 }}>
        <label>Prompt AI: <input value={userInput} onChange={e => setUserInput(e.target.value)} style={{ width: 300 }} /></label>
        <button onClick={askAI} disabled={loading} style={{ marginLeft: 8 }}>{loading ? 'Asking...' : 'Ask AI'}</button>
      </div>
      {response && (
        <div style={{ marginTop: 16, background: '#f6f6f6', padding: 12, borderRadius: 6 }}>
          <b>AI Response:</b>
          <pre>{response}</pre>
        </div>
      )}
    </div>
  );
};

export default DataLeakageDemo; 