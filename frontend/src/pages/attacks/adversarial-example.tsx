import React, { useState, useRef } from 'react';

const AdversarialExampleDemo: React.FC = () => {
  const [originalPrediction, setOriginalPrediction] = useState('');
  const [advPrediction, setAdvPrediction] = useState('');
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [advImage, setAdvImage] = useState<string | null>(null);
  const [epsilon, setEpsilon] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setOriginalImage(URL.createObjectURL(file));
    setAdvImage(null);
    setOriginalPrediction('');
    setAdvPrediction('');
    setLoading(true);
    // Classify original image
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch('/api/classify-image', { method: 'POST', body: formData });
    const data = await res.json();
    setOriginalPrediction(data.prediction || 'Error');
    setLoading(false);
  };

  const handleGenerateAdversarial = async () => {
    if (!fileInput.current || !fileInput.current.files || fileInput.current.files.length === 0) return;
    const file = fileInput.current.files[0];
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('epsilon', epsilon.toString());
    const res = await fetch('/api/adversarial-image', { method: 'POST', body: formData });
    const blob = await res.blob();
    const advPred = res.headers.get('X-Adversarial-Prediction') || 'Unknown';
    setAdvPrediction(advPred);
    setAdvImage(URL.createObjectURL(blob));
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h2>Adversarial Example Attack Demo <span role="img" aria-label="adversarial">🖼️</span></h2>
      <p>This demo shows how a small, crafted change to an image can fool a model.</p>
      <input type="file" accept="image/*" ref={fileInput} onChange={handleImageUpload} />
      {originalImage && (
        <div style={{ marginTop: 16 }}>
          <b>Original Image:</b><br />
          <img src={originalImage} alt="original" style={{ maxWidth: 300, border: '1px solid #ccc', borderRadius: 4 }} /><br />
          <b>Prediction:</b> {originalPrediction}
        </div>
      )}
      {originalImage && (
        <div style={{ marginTop: 16 }}>
          <label>
            <b>Epsilon (attack strength):</b>
            <input type="number" min={0.01} max={0.5} step={0.01} value={epsilon} onChange={e => setEpsilon(Number(e.target.value))} style={{ marginLeft: 8, width: 80 }} />
          </label>
          <button onClick={handleGenerateAdversarial} disabled={loading} style={{ marginLeft: 16 }}>
            {loading ? 'Generating...' : 'Generate Adversarial Image'}
          </button>
        </div>
      )}
      {advImage && (
        <div style={{ marginTop: 16 }}>
          <b>Adversarial Image:</b><br />
          <img src={advImage} alt="adversarial" style={{ maxWidth: 300, border: '1px solid #f66', borderRadius: 4 }} /><br />
          <b>Prediction:</b> {advPrediction}
        </div>
      )}
    </div>
  );
};

export default AdversarialExampleDemo; 