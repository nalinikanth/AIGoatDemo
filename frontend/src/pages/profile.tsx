import React, { useState } from 'react';

const isRedTeam = process.env.NEXT_PUBLIC_RED_TEAM_MODE === 'true';

const MembershipInferencePrivacyCheck: React.FC<{ user: string }> = ({ user }) => {
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
        `Product: ${data.product}\nConfidence: ${data.confidence}\nIn Training Data: ${data.in_training}`
      );
    }
    setLoading(false);
  };

  return (
    <div style={{ marginTop: 32, padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee', maxWidth: 600 }}>
      <h3>Privacy Check: Is Your Data in the Model?</h3>
      {isRedTeam && (
        <div style={{ background: '#fffbe6', border: '1px solid #ffe58f', padding: 12, borderRadius: 6, marginBottom: 12 }}>
          <b>Red Team Challenge:</b> Can you infer if a user’s data was used to train the model?
        </div>
      )}
      <label>Product: <input value={product} onChange={e => setProduct(e.target.value)} /></label>
      <button onClick={checkMembership} disabled={loading} style={{ marginLeft: 8 }}>{loading ? 'Checking...' : 'Check Membership'}</button>
      {result && (
        <div style={{ marginTop: 16, background: '#f6f6f6', padding: 12, borderRadius: 6 }}>
          <pre>{result}</pre>
          {result.includes('true') && <div style={{ color: 'red', marginTop: 8 }}>Warning: Your data was likely used to train the model.</div>}
        </div>
      )}
    </div>
  );
};

// Example usage: replace 'attacker' with the logged-in user's ID or username
const ProfilePage: React.FC = () => {
  const user = 'attacker'; // Replace with real user context
  return (
    <div style={{ maxWidth: 800, margin: '40px auto' }}>
      <h1>User Profile</h1>
      {/* ... other profile info ... */}
      <MembershipInferencePrivacyCheck user={user} />
    </div>
  );
};

export default ProfilePage; 