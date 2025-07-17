import React, { useState } from 'react';
import dynamic from 'next/dynamic';

const PromptInjectionDemo = dynamic(() => import('./ai-demo'));
const DataPoisoningDemo = dynamic(() => import('./attacks/data-poisoning'));
const AdversarialExampleDemo = dynamic(() => import('./attacks/adversarial-example'));
const MembershipInferenceDemo = dynamic(() => import('./attacks/membership-inference'));
const DataLeakageDemo = dynamic(() => import('./attacks/data-leakage'));
const ModelExtractionDemo = dynamic(() => import('./attacks/model-extraction'));

const TABS = [
  { key: 'prompt-injection', label: 'Prompt Injection', component: <PromptInjectionDemo /> },
  { key: 'data-poisoning', label: 'Data Poisoning', component: <DataPoisoningDemo /> },
  { key: 'adversarial-example', label: 'Adversarial Example', component: <AdversarialExampleDemo /> },
  { key: 'membership-inference', label: 'Membership Inference', component: <MembershipInferenceDemo /> },
  { key: 'data-leakage', label: 'Sensitive Data Leakage', component: <DataLeakageDemo /> },
  { key: 'model-extraction', label: 'Model Extraction', component: <ModelExtractionDemo /> },
];

const SecurityLab: React.FC = () => {
  const [activeTab, setActiveTab] = useState(TABS[0].key);

  return (
    <div style={{ maxWidth: 900, margin: '40px auto', padding: 24, background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #eee' }}>
      <h1>Security Lab <span role="img" aria-label="shield">🛡️</span></h1>
      <p>Explore and test AI/ML security attacks in a safe environment. Each tab demonstrates a different attack scenario relevant to e-commerce and AI systems.</p>
      <div style={{ display: 'flex', borderBottom: '1px solid #eee', marginBottom: 24 }}>
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '12px 24px',
              border: 'none',
              borderBottom: activeTab === tab.key ? '3px solid #1890ff' : '3px solid transparent',
              background: 'none',
              fontWeight: activeTab === tab.key ? 'bold' : 'normal',
              cursor: 'pointer',
              outline: 'none',
              fontSize: 16,
              color: activeTab === tab.key ? '#1890ff' : '#333',
              marginRight: 8,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div>
        {TABS.find(tab => tab.key === activeTab)?.component}
      </div>
    </div>
  );
};

export default SecurityLab; 