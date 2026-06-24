import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
  LineChart, Line,
  LabelList,
} from 'recharts';
import { API } from '../config/api';
import './Analytics.css';

// ── colour helpers ───────────────────────────────────────────────────────────
function rateColor(rate) {
  if (rate >= 0.75) return '#f87171';   // critical — red
  if (rate >= 0.50) return '#fb923c';   // high     — orange
  if (rate >= 0.25) return '#fbbf24';   // medium   — yellow
  return '#4ade80';                      // low      — green
}

// ── custom tooltip ───────────────────────────────────────────────────────────
function ZoneTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <p className="chart-tip__title">{d.zone}</p>
      <p>Rejection rate: <strong>{(d.rejection_rate * 100).toFixed(1)}%</strong></p>
      <p>Rejected: {d.rejected} / {d.total}</p>
    </div>
  );
}

function ReasonTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tip">
      <p className="chart-tip__title">{d.label}</p>
      <p>Appearances in top-5: <strong>{d.count}</strong></p>
      <p>Avg SHAP impact: <strong>{d.avg_impact?.toFixed(3)}</strong></p>
      <p>Direction: <strong>{d.direction?.replace('_', ' ')}</strong></p>
    </div>
  );
}

function TimeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <p className="chart-tip__title">{label}</p>
      <p>Submissions: <strong>{payload[0]?.value}</strong></p>
      {payload[1] && <p>Rejections: <strong>{payload[1]?.value}</strong></p>}
    </div>
  );
}

// ── main component ───────────────────────────────────────────────────────────
export default function Analytics() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    axios.get(`${API}/analytics`)
      .then(res => { setData(res.data); setLoading(false); })
      .catch(err => {
        console.error('[Analytics] fetch error:', err);
        setError(err.response?.data?.detail ?? err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSkeleton />;
  if (error)   return <div className="analytics-error">⚠ {error}</div>;
  if (!data)   return null;

  const { overall_rejection_rate, total_submissions,
          rejection_by_zone, top_rejection_reasons,
          submissions_over_time } = data;

  return (
    <div className="analytics-page">

      {/* ── KPI strip ── */}
      <div className="kpi-strip">
        <KPI label="Total Submissions"  value={total_submissions} />
        <KPI label="Overall Rejection Rate"
             value={`${(overall_rejection_rate * 100).toFixed(1)}%`}
             highlight={overall_rejection_rate > 0.5} />
        <KPI label="Zones Tracked"      value={rejection_by_zone.length} />
        <KPI label="Top Risk Features"  value={top_rejection_reasons.length} />
      </div>

      {/* ── Chart A: Rejection rate by zone ── */}
      <section className="chart-card">
        <h3 className="chart-title">Rejection Rate by Zone</h3>
        <p className="chart-sub">Higher bar = higher proportion of rejected applications in that zone</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={rejection_by_zone}
            margin={{ top: 8, right: 24, left: 0, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2130" vertical={false} />
            <XAxis
              dataKey="zone"
              tick={{ fill: '#64748b', fontSize: 11 }}
              angle={-30}
              textAnchor="end"
              interval={0}
            />
            <YAxis
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: '#64748b', fontSize: 11 }}
              domain={[0, 1]}
            />
            <Tooltip content={<ZoneTooltip />} cursor={{ fill: '#ffffff08' }} />
            <Bar dataKey="rejection_rate" radius={[4, 4, 0, 0]}>
              {rejection_by_zone.map((entry, i) => (
                <Cell key={i} fill={rateColor(entry.rejection_rate)} />
              ))}
              <LabelList
                dataKey="rejection_rate"
                position="top"
                formatter={v => `${(v * 100).toFixed(0)}%`}
                style={{ fill: '#94a3b8', fontSize: 10 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* ── Chart B: Top rejection reasons (horizontal bars) ── */}
      <section className="chart-card">
        <h3 className="chart-title">Top Rejection Reasons</h3>
        <p className="chart-sub">Ranked by cumulative SHAP impact across all submissions</p>
        <ResponsiveContainer width="100%" height={Math.max(180, top_rejection_reasons.length * 52)}>
          <BarChart
            layout="vertical"
            data={[...top_rejection_reasons].reverse()}   /* highest at top */
            margin={{ top: 4, right: 60, left: 20, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2130" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="label"
              width={130}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <Tooltip content={<ReasonTooltip />} cursor={{ fill: '#ffffff08' }} />
            <Bar dataKey="total_impact" fill="#4f6ef7" radius={[0, 4, 4, 0]}>
              <LabelList
                dataKey="count"
                position="right"
                formatter={v => `${v}×`}
                style={{ fill: '#64748b', fontSize: 10 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </section>

      {/* ── Chart C: Submissions over time ── */}
      <section className="chart-card">
        <h3 className="chart-title">Submissions Over Time</h3>
        <p className="chart-sub">Monthly totals and rejection counts</p>
        {submissions_over_time.length === 0 ? (
          <p className="chart-empty">No time-series data yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart
              data={submissions_over_time}
              margin={{ top: 8, right: 24, left: 0, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2130" />
              <XAxis
                dataKey="period"
                tick={{ fill: '#64748b', fontSize: 11 }}
              />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<TimeTooltip />} />
              <Line
                type="monotone"
                dataKey="submissions"
                stroke="#4f6ef7"
                strokeWidth={2}
                dot={{ fill: '#4f6ef7', r: 4 }}
                activeDot={{ r: 6 }}
                name="Submissions"
              />
              <Line
                type="monotone"
                dataKey="rejection_count"
                stroke="#f87171"
                strokeWidth={2}
                dot={{ fill: '#f87171', r: 4 }}
                activeDot={{ r: 6 }}
                name="Rejections"
                strokeDasharray="5 3"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
        {/* legend */}
        <div className="chart-legend">
          <span className="legend-dot" style={{ background: '#4f6ef7' }} /> Submissions
          <span className="legend-dot" style={{ background: '#f87171', marginLeft: 16 }} /> Rejections
        </div>
      </section>

    </div>
  );
}

// ── sub-components ────────────────────────────────────────────────────────────
function KPI({ label, value, highlight }) {
  return (
    <div className={`kpi-card ${highlight ? 'kpi-card--alert' : ''}`}>
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="analytics-page">
      <div className="kpi-strip">
        {[...Array(4)].map((_, i) => <div key={i} className="kpi-skeleton" />)}
      </div>
      {[...Array(3)].map((_, i) => (
        <div key={i} className="chart-card">
          <div className="skeleton-line" style={{ width: '30%', marginBottom: 8 }} />
          <div className="skeleton-line" style={{ width: '55%', marginBottom: 20 }} />
          <div className="skeleton-chart" />
        </div>
      ))}
    </div>
  );
}
