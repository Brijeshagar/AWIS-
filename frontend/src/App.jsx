import { useState } from 'react';
import RulesEditor from './pages/RulesEditor';
import Queue       from './pages/Queue';
import Analytics   from './pages/Analytics';
import './App.css';

const TABS = [
  { key: 'queue',     label: '📋 Submission Queue' },
  { key: 'analytics', label: '📊 Analytics'        },
  { key: 'rules',     label: '⚙️ Rules Editor'     },
];

const PAGE_META = {
  queue: {
    title: 'Submission Queue',
    desc:  'All scored permit applications, sorted by rejection risk (highest first).',
  },
  analytics: {
    title: 'Analytics',
    desc:  'Rejection trends, top risk drivers, and zone-level breakdown.',
  },
  rules: {
    title: 'Rules Editor',
    desc:  'Manage rejection rules stored in the database. Changes take effect immediately.',
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState('queue');
  const meta = PAGE_META[activeTab];

  return (
    <div className="app-shell">

      {/* ── sticky header ── */}
      <header className="app-header">
        <div className="app-header__inner">
          <div className="app-header__brand">
            <span className="brand-dot" />
            <span className="brand-name">AWIS</span>
            <span className="brand-divider">|</span>
            <span className="brand-sub" title="Adaptive Workflow Intervention System">
              Adaptive Workflow Intervention System
            </span>
          </div>
          <span className="app-header__version">v1.0</span>
        </div>

        {/* ── tab bar ── */}
        <nav className="app-tabs">
          <div className="app-tabs__inner">
            {TABS.map(t => (
              <button
                key={t.key}
                className={`app-tab ${activeTab === t.key ? 'app-tab--active' : ''}`}
                onClick={() => setActiveTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      {/* ── page ── */}
      <main className="app-main">
        <div className="page-title">
          <h1>{meta.title}</h1>
          <p className="page-desc">{meta.desc}</p>
        </div>

        {activeTab === 'queue'     && <Queue />}
        {activeTab === 'analytics' && <Analytics />}
        {activeTab === 'rules'     && <RulesEditor />}
      </main>

    </div>
  );
}
