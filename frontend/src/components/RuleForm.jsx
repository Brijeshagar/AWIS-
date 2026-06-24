import { useState, useEffect } from 'react';
import './RuleForm.css';

const RULE_TYPES = [
  'zone_conflict',
  'doc_requirement',
  'fsi_limit',
  'format_check',
  'threshold',
  'flag_check',
  'document_check',
  'zone_permit_mismatch',
];

const EMPTY_FORM = {
  rule_name:   '',
  rule_type:   'zone_conflict',
  zone:        '',
  permit_type: '',
  condition:   '',
  value:       '',
  active:      true,
};

export default function RuleForm({ editingRule, onSubmit, onCancel }) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  // Populate form when editingRule changes
  useEffect(() => {
    if (editingRule) {
      setForm({
        rule_name:   editingRule.rule_name   ?? '',
        rule_type:   editingRule.rule_type   ?? 'zone_conflict',
        zone:        editingRule.zone        ?? '',
        permit_type: editingRule.permit_type ?? '',
        condition:   editingRule.condition   ?? '',
        value:       editingRule.value       ?? '',
        active:      editingRule.active      ?? true,
      });
    } else {
      setForm(EMPTY_FORM);
    }
    setError(null);
  }, [editingRule]);

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setForm(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    // Send null for empty optional strings so the backend treats them as "all"
    const payload = {
      ...form,
      zone:        form.zone.trim()        || null,
      permit_type: form.permit_type.trim() || null,
    };

    try {
      await onSubmit(payload, editingRule?.id ?? null);
      setForm(EMPTY_FORM);
    } catch (err) {
      const detail = err.response?.data?.detail ?? err.message;
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail));
    } finally {
      setSubmitting(false);
    }
  }

  function handleCancel() {
    setForm(EMPTY_FORM);
    setError(null);
    onCancel();
  }

  const isEditing = Boolean(editingRule);

  return (
    <form className="rule-form" onSubmit={handleSubmit}>
      <h3 className="rule-form__title">
        {isEditing ? `Edit Rule #${editingRule.id}` : 'New Rule'}
      </h3>

      {error && <div className="rule-form__error">{error}</div>}

      <label className="field">
        <span>Rule Name *</span>
        <input
          name="rule_name"
          value={form.rule_name}
          onChange={handleChange}
          placeholder="e.g. high_construction_cost"
          required
        />
      </label>

      <label className="field">
        <span>Rule Type *</span>
        <select name="rule_type" value={form.rule_type} onChange={handleChange} required>
          {RULE_TYPES.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Zone <small>(leave blank = all zones)</small></span>
        <input
          name="zone"
          value={form.zone}
          onChange={handleChange}
          placeholder="e.g. Industrial"
        />
      </label>

      <label className="field">
        <span>Permit Type <small>(leave blank = all types)</small></span>
        <input
          name="permit_type"
          value={form.permit_type}
          onChange={handleChange}
          placeholder="e.g. Residential"
        />
      </label>

      <label className="field">
        <span>Condition *</span>
        <input
          name="condition"
          value={form.condition}
          onChange={handleChange}
          placeholder="e.g.  >  |  ==  |  match  |  missing"
          required
        />
      </label>

      <label className="field">
        <span>Value *</span>
        <input
          name="value"
          value={form.value}
          onChange={handleChange}
          placeholder="e.g. 3  |  1  |  block"
          required
        />
      </label>

      <label className="field field--inline">
        <input
          type="checkbox"
          name="active"
          checked={form.active}
          onChange={handleChange}
        />
        <span>Active</span>
      </label>

      <div className="rule-form__actions">
        <button type="submit" className="btn btn--primary" disabled={submitting}>
          {submitting ? 'Saving…' : isEditing ? 'Update Rule' : 'Create Rule'}
        </button>
        {isEditing && (
          <button type="button" className="btn btn--ghost" onClick={handleCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
