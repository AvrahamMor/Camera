/**
 * Analytics & Reports Module
 * Handles Chart.js visualizations, turnover metrics, and event audit log.
 */

let hourlyChart = null;
let turnoverPieChart = null;

async function loadAnalyticsData() {
  await Promise.all([
    fetchHourlyAnalytics(),
    fetchTableTurnoverAnalytics(),
    fetchEventLog()
  ]);
}

async function fetchHourlyAnalytics() {
  try {
    const res = await fetch('/api/analytics/hourly');
    if (!res.ok) return;
    const data = await res.json();
    renderHourlyChart(data.labels, data.turnovers, data.diners);
  } catch (err) {
    console.error('Failed to fetch hourly analytics:', err);
  }
}

function renderHourlyChart(labels, turnovers, diners) {
  const canvas = document.getElementById('chart-hourly');
  if (!canvas) return;

  if (hourlyChart) {
    hourlyChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  hourlyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'מחזורי שולחן (סבבים)',
          data: turnovers,
          backgroundColor: 'rgba(59, 130, 246, 0.75)',
          borderColor: '#3b82f6',
          borderWidth: 1,
          borderRadius: 6,
          yAxisID: 'y'
        },
        {
          label: 'נפח סועדים',
          data: diners,
          type: 'line',
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          borderWidth: 2.5,
          tension: 0.35,
          fill: true,
          pointBackgroundColor: '#10b981',
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#cbd5e1', font: { size: 11, family: 'Heebo' } }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', stepSize: 1, font: { size: 10 } },
          title: { display: true, text: 'סבבים', color: '#64748b', font: { size: 10 } }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#10b981', font: { size: 10 } },
          title: { display: true, text: 'סועדים', color: '#10b981', font: { size: 10 } }
        }
      }
    }
  });
}

async function fetchTableTurnoverAnalytics() {
  try {
    const res = await fetch('/api/analytics/tables');
    if (!res.ok) return;
    const tables = await res.json();
    renderTableTurnoverTable(tables);
    renderTurnoverPie(tables);
  } catch (err) {
    console.error('Failed to fetch table turnover metrics:', err);
  }
}

function renderTableTurnoverTable(tables) {
  const tbody = document.getElementById('analytics-tables-body');
  if (!tbody) return;

  if (tables.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-slate-500">לא נרשמו עדיין סבבי שולחן שהושלמו</td></tr>';
    return;
  }

  tbody.innerHTML = tables.map(t => `
    <tr class="hover:bg-slate-800/40 transition">
      <td class="py-2.5 px-3">
        <div class="font-semibold text-white">${t.table_name}</div>
        <div class="text-[11px] text-slate-500 font-mono">${t.table_id}</div>
      </td>
      <td class="py-2.5 px-3">
        <span class="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-bold border border-blue-500/20 text-xs">
          ${t.turnovers}
        </span>
      </td>
      <td class="py-2.5 px-3">
        <span class="text-slate-200 font-semibold">${t.avg_turnover_minutes} דק׳</span>
      </td>
      <td class="py-2.5 px-3">
        <span class="text-teal-400 font-semibold">${t.total_guests} סועדים</span>
      </td>
    </tr>
  `).join('');
}

function renderTurnoverPie(tables) {
  const canvas = document.getElementById('chart-turnover-pie');
  if (!canvas) return;

  if (turnoverPieChart) {
    turnoverPieChart.destroy();
  }

  const labels = tables.map(t => t.table_name);
  const data = tables.map(t => t.turnovers);

  const colors = [
    '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
    '#06b6d4', '#84cc16', '#f97316'
  ];

  const ctx = canvas.getContext('2d');
  turnoverPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors.slice(0, labels.length),
        borderColor: '#0f172a',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { size: 10, family: 'Inter' } }
        }
      },
      cutout: '65%'
    }
  });
}

async function fetchEventLog() {
  try {
    const res = await fetch('/api/analytics/events?limit=30');
    if (!res.ok) return;
    const events = await res.json();
    renderEventLog(events);
  } catch (err) {
    console.error('Failed to fetch event log:', err);
  }
}

function renderEventLog(events) {
  const container = document.getElementById('analytics-events-container');
  if (!container) return;

  if (events.length === 0) {
    container.innerHTML = '<div class="text-center py-4 text-slate-500">לא נרשמו אירועים במערכת עדיין</div>';
    return;
  }

  container.innerHTML = events.map(e => {
    let badgeClass = 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    let labelHebrew = e.event_type;

    if (e.event_type === 'SEATED') {
      badgeClass = 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      labelHebrew = 'התיישבות';
    } else if (e.event_type === 'VACATED') {
      badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      labelHebrew = 'פינוי שולחן';
    } else if (e.event_type === 'SERVICE_ALERT') {
      badgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      labelHebrew = 'דרוש שירות!';
    } else if (e.event_type === 'SERVICE_ACK') {
      badgeClass = 'bg-slate-700 text-slate-300 border-slate-600';
      labelHebrew = 'אושר שירות';
    }

    return `
      <div class="p-2.5 rounded-xl bg-surface-850 border border-slate-800 flex items-start justify-between text-xs">
        <div class="flex items-start space-x-2.5 space-x-reverse">
          <span class="mt-0.5 px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass}">
            ${labelHebrew}
          </span>
          <div>
            <div class="text-slate-200 font-medium">${e.details}</div>
            <div class="text-[10px] text-slate-500 font-mono">${e.date} • שולחן: ${e.table_id}</div>
          </div>
        </div>
        <span class="text-[11px] text-slate-400 font-mono">${e.timestamp}</span>
      </div>
    `;
  }).join('');
}
