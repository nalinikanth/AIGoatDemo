import React, { useState } from 'react';

const MembershipInferenceDemo: React.FC = () => {
  const [user, setUser] = useState('attacker');
  const [product, setProduct] = useState('red_panda_doll');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const checkMembership = async () => {
    setLoading(true);
    setResult('');
    const res = await fetch('/api/membership-inference', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user, product })
    });
    const data = await res.json();
    if (data.error) {
      setResult(data.error);
    } else {
      setResult(
        `User: ${data.user}\nProduct: ${data.product}\nConfidence: ${data.confidence}\nIn Training Data: ${data.in_training}`
      );
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h2>Membership Inference Attack Demo <span role="img" aria-label="magnifier">🔍</span></h2>
      <p>This demo shows how an attacker can infer if a user-product pair was in the training set based on model confidence.</p>
      <div style={{ marginBottom: 16 }}>
        <label>User: <input value={user} onChange={e => setUser(e.target.value)} /></label>
        <label style={{ marginLeft: 8 }}>Product: <input value={product} onChange={e => setProduct(e.target.value)} /></label>
        <button onClick={checkMembership} disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? 'Checking...' : 'Check Membership'}
        </button>
      </div>
      {result && (
        <div style={{ marginTop: 16, background: '#f6f6f6', padding: 12, borderRadius: 6 }}>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
};

export default MembershipInferenceDemo; 