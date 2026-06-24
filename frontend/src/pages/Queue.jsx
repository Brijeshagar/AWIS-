import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { API } from '../config/api';
import './Queue.css';

// ── helpers ─────────────────────────────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString('en-IN', {
    day:   '2-digit',
    month: 'short',
    year:  'numeric',
    hour:  '2-digit',
    minute:'2-digit',
  });
}

function firstReason(top_reasons) {
  if (!Array.isArray(top_reasons) || top_reasons.length === 0) return '—';
  const r = top_reasons[0];
  const name   = r.feature?.replace(/_/g, ' ') ?? '—';
  const dir    = r.impact === 'increases_risk' ? '↑' : '↓';
  return `${dir} ${name}`;
}

function riskLabel(score) {
  if (score >= 75) return 'critical';
  if (score >= 50) return 'high';
  if (score >= 25) return 'medium';
  return 'low';
}

// ── component ────────────────────────────────────────────────────────────────
export default function Queue() {
  const [rows,    setRows]    = useState([]);
  const [total,   setTotal]   = useState(0);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);
  const [limit,   setLimit]   = useState(50);
  const [offset,  setOffset]  = useState(0);

  const fetchSubmissions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.get(`${API}/submissions`, {
        params: { limit, offset },
      });
      setRows(data.results ?? []);
      setTotal(data.total  ?? 0);
    } catch (err) {
      console.error('[Queue] fetch error:', err);
      setError(err.response?.data?.detail ?? err.message);
    } finally {
      setLoading(false);
    }
  }, [limit, offset]);

  useEffect(() => { fetchSubmissions(); }, [fetchSubmissions]);

  const totalPages = Math.ceil(total / limit) || 1;
  const currentPage = Math.floor(offset / limit) + 1;

  function prevPage() { if (offset >= limit) setOffset(o => o - limit); }
  function nextPage() { if (offset + limit < total) setOffset(o => o + limit); }

  return (
    <div className="queue-page">

      {/* ── toolbar ── */}
      <div className="queue-toolbar">
        <div className="queue-meta">
          {!loading && !error && (
            <span className="queue-count">
              {total} submission{total !== 1 ? 's' : ''} · sorted by risk score ↓
            </span>
          )}
        </div>
        <button className="btn btn--ghost queue-refresh" onClick={fetchSubmissions} disabled={loading}>
          {loading ? 'Loading…' : '↻ Refresh'}
        </button>
      </div>

      {/* ── error ── */}
      {error && (
        <div className="queue-error">
          ⚠ Could not load submissions: {error}
        </div>
      )}

      {/* ── loading skeleton ── */}
      {loading && !error && (
        <div className="queue-loading">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="skeleton-row" />
          ))}
        </div>
      )}

      {/* ── empty state ── */}
      {!loading && !error && rows.length === 0 && (
        <div className="queue-empty">
          <span className="queue-empty__icon">📭</span>
          <p>No submissions yet.</p>
          <p className="queue-empty__sub">Use the mock portal or POST /submit to add records.</p>
        </div>
      )}

      {/* ── table ── */}
      {!loading && !error && rows.length > 0 && (
        <>
          <div className="queue-table-wrap">
            <table className="queue-table">
              <thead>
                <tr>
                  <th>#&thinsp;ID</th>
                  <th>Permit Type</th>
                  <th>Zone</th>
                  <th>Risk Score</th>
                  <th>Top Reason</th>
                  <th>Submitted At</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(row => {
                  const prediction = row.prediction ??
                    (row.risk_score >= 50 ? 'Rejected' : 'Approved');
                  const isRejected = prediction === 'Rejected';

                  return (
                    <tr key={row.id} className={isRejected ? 'row--rejected' : 'row--approved'}>
                      <td className="col-id">{row.id}</td>
                      <td>{row.permit_type ?? '—'}</td>
                      <td>{row.zone ?? '—'}</td>
                      <td>
                        <span className={`risk-pill risk-pill--${riskLabel(row.risk_score)}`}>
                          {row.risk_score?.toFixed(1)}
                        </span>
                      </td>
                      <td className="col-reason">
                        {firstReason(row.top_reasons)}
                      </td>
                      <td className="col-date">{formatDate(row.submitted_at)}</td>
                      <td>
                        <span className={`status-badge status-badge--${isRejected ? 'rejected' : 'approved'}`}>
                          {prediction}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ── pagination ── */}
          {total > limit && (
            <div className="queue-pagination">
              <button className="btn btn--ghost" onClick={prevPage} disabled={offset === 0}>
                ← Prev
              </button>
              <span className="pagination-info">
                Page {currentPage} of {totalPages} &nbsp;·&nbsp; {total} total
              </span>
              <button className="btn btn--ghost" onClick={nextPage} disabled={offset + limit >= total}>
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
