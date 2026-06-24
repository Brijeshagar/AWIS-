import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import RuleCard from '../components/RuleCard';
import RuleForm from '../components/RuleForm';
import { API } from '../config/api';
import './RulesEditor.css';

export default function RulesEditor() {
  const [rules, setRules]           = useState([]);
  const [loading, setLoading]       = useState(true);
  const [fetchError, setFetchError] = useState(null);
  const [editingRule, setEditingRule] = useState(null);   // null = create mode
  const [activeOnly, setActiveOnly]   = useState(false);

  // ── Fetch rules ──────────────────────────────────────────────────────────
  const fetchRules = useCallback(async () => {
    setLoading(true);
    setFetchError(null);
    try {
      const params = activeOnly ? { active_only: true } : {};
      const { data } = await axios.get(`${API}/rules`, { params });
      setRules(data);
    } catch (err) {
      console.error('[RulesEditor] fetch failed:', err);
      setFetchError(err.response?.data?.detail ?? err.message);
    } finally {
      setLoading(false);
    }
  }, [activeOnly]);

  useEffect(() => { fetchRules(); }, [fetchRules]);

  // ── Create / Update ───────────────────────────────────────────────────────
  async function handleSubmit(payload, id) {
    if (id) {
      await axios.put(`${API}/rules/${id}`, payload);
    } else {
      await axios.post(`${API}/rules`, payload);
    }
    setEditingRule(null);
    await fetchRules();
  }

  // ── Delete ────────────────────────────────────────────────────────────────
  async function handleDelete(id) {
    if (!window.confirm(`Delete rule #${id}? This will deactivate it.`)) return;
    try {
      await axios.delete(`${API}/rules/${id}`);
      await fetchRules();
    } catch (err) {
      console.error('[RulesEditor] delete failed:', err);
      alert(err.response?.data?.detail ?? err.message);
    }
  }

  // ── Edit ─────────────────────────────────────────────────────────────────
  function handleEdit(rule) {
    setEditingRule(rule);
    // Scroll form into view on mobile
    document.getElementById('rules-form-panel')?.scrollIntoView({ behavior: 'smooth' });
  }

  function handleCancel() {
    setEditingRule(null);
  }

  return (
    <div className="rules-editor">
      {/* ── Left: list ───────────────────────────────────────────────── */}
      <section className="rules-editor__list-panel">
        <div className="panel-header">
          <h2>Rules <span className="count-badge">{rules.length}</span></h2>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={e => setActiveOnly(e.target.checked)}
            />
            Active only
          </label>
        </div>

        {loading && <p className="status-msg">Loading rules…</p>}
        {fetchError && <p className="status-msg status-msg--error">Error: {fetchError}</p>}

        {!loading && !fetchError && rules.length === 0 && (
          <p className="status-msg">No rules found. Create one →</p>
        )}

        <div className="rule-list">
          {rules.map(rule => (
            <RuleCard
              key={rule.id}
              rule={rule}
              onEdit={handleEdit}
              onDelete={handleDelete}
            />
          ))}
        </div>
      </section>

      {/* ── Right: form ──────────────────────────────────────────────── */}
      <section className="rules-editor__form-panel" id="rules-form-panel">
        <RuleForm
          editingRule={editingRule}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
        />
      </section>
    </div>
  );
}
