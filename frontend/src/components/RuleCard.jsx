import './RuleCard.css';

export default function RuleCard({ rule, onEdit, onDelete }) {
  return (
    <div className={`rule-card ${rule.active ? 'active' : 'inactive'}`}>
      <div className="rule-card__header">
        <span className="rule-card__name">{rule.rule_name}</span>
        <span className={`rule-card__badge ${rule.active ? 'badge--on' : 'badge--off'}`}>
          {rule.active ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className="rule-card__meta">
        <MetaItem label="Type"        value={rule.rule_type} />
        <MetaItem label="Zone"        value={rule.zone        ?? 'All zones'} />
        <MetaItem label="Permit Type" value={rule.permit_type ?? 'All types'} />
        <MetaItem label="Condition"   value={rule.condition} />
        <MetaItem label="Value"       value={rule.value} />
      </div>

      <div className="rule-card__actions">
        <button className="btn btn--edit"   onClick={() => onEdit(rule)}>Edit</button>
        <button className="btn btn--delete" onClick={() => onDelete(rule.id)}>Delete</button>
      </div>
    </div>
  );
}

function MetaItem({ label, value }) {
  return (
    <div className="meta-item">
      <span className="meta-item__label">{label}</span>
      <span className="meta-item__value">{value}</span>
    </div>
  );
}
