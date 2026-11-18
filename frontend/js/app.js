/**
 * Main Application Controller
 * Handles WebSocket real-time synchronization, KPI updates,
 * Live floor grid, service alerts, and settings.
 */

let liveWs = null;
let currentTab = 'live';
let currentStreamMode = 'matrix'; // 'matrix' (3x3 grid) or 'single' (focused camera)
let lastAlertCount = 0;
let liveAudioEnabled = true;

document.addEventListener('DOMContentLoaded', () => {
  initLiveWebSocket();
  initSettingsListeners();
  loadSystemSettings();
  loadCameraChannels();

  // Restore last active tab from hash or localStorage
  const savedTab = window.location.hash.replace('#', '') || localStorage.getItem('visiondine_active_tab') || 'live';
  if (savedTab && ['live', 'calibration', 'analytics', 'settings'].includes(savedTab) && savedTab !== 'live') {
    switchTab(savedTab);
  }

  // Refresh KPIs periodically
  setInterval(refreshKPIs, 3000);
});

// --- 3x3 Matrix Grid vs Single Focus Mode Switcher ---
function switchStreamViewMode(mode) {
  currentStreamMode = mode;
  const btnMatrix = document.getElementById('btn-view-matrix');
  const btnSingle = document.getElementById('btn-view-single');
  const pills = document.getElementById('matrix-quick-pills');
  const selContainer = document.getElementById('camera-select-container');
  const title = document.getElementById('stream-header-title');
  const resBadge = document.getElementById('stream-resolution-badge');
  const img = document.getElementById('live-video-stream');

  if (mode === 'matrix') {
    if (btnMatrix) btnMatrix.className = 'px-3 py-1 rounded-lg text-xs font-bold transition flex items-center space-x-1.5 space-x-reverse bg-blue-600 text-white shadow-sm';
    if (btnSingle) btnSingle.className = 'px-3 py-1 rounded-lg text-xs font-bold transition flex items-center space-x-1.5 space-x-reverse text-slate-400 hover:text-white';
    if (pills) pills.classList.remove('hidden');
    if (selContainer) selContainer.classList.add('hidden');
    if (title) title.innerText = 'שידור חי • רשת 9 מצלמות (1-9)';
    if (resBadge) resBadge.innerText = 'מטריצה 3x3';
    if (img) img.src = '/api/video_feed?mode=matrix&t=' + Date.now();
  } else {
    if (btnSingle) btnSingle.className = 'px-3 py-1 rounded-lg text-xs font-bold transition flex items-center space-x-1.5 space-x-reverse bg-blue-600 text-white shadow-sm';
    if (btnMatrix) btnMatrix.className = 'px-3 py-1 rounded-lg text-xs font-bold transition flex items-center space-x-1.5 space-x-reverse text-slate-400 hover:text-white';
    if (pills) pills.classList.add('hidden');
    if (selContainer) selContainer.classList.remove('hidden');

    const sel = document.getElementById('camera-select');
    const ch = sel ? sel.value : 1;
    if (title) title.innerText = `שידור חי • מצלמה ${ch}`;
    if (resBadge) resBadge.innerText = `מיקוד מצלמה ${ch}`;
    if (img) img.src = `/api/video_feed?mode=single&t=${Date.now()}`;
  }
}

function zoomToCamera(channel) {
  quickSwitchCamera(channel);
  switchStreamViewMode('single');
}

function onVideoPlayerClick(e) {
  // If in matrix mode, clicking any quadrant zooms into that camera!
  if (currentStreamMode !== 'matrix') return;
  const img = document.getElementById('live-video-stream');
  if (!img) return;

  const rect = img.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const relX = clickX / rect.width;
  const relY = clickY / rect.height;

  // 3x3 Grid
  const col = Math.min(2, Math.floor(relX * 3));
  const row = Math.min(2, Math.floor(relY * 3));
  const channel = row * 3 + col + 1;

  if (channel >= 1 && channel <= 8) {
    showToast(`מתמקד במצלמה ${channel}...`, 'info');
    zoomToCamera(channel);
  }
}

// --- Camera Channel Quick Switcher ---
async function loadCameraChannels() {
  try {
    const res = await fetch('/api/camera/channels');
    if (res.ok) {
      const data = await res.json();
      const sel = document.getElementById('camera-select');
      if (sel && data.channels) {
        sel.innerHTML = data.channels.map(c =>
          `<option value="${c.channel}" ${c.channel === data.current_channel ? 'selected' : ''}>${c.name}${c.channel === 1 ? ' (ראשי)' : ''}</option>`
        ).join('');
      }
      const title = document.getElementById('stream-header-title');
      if (title && data.current_channel) {
        title.innerText = `שידור חי • מצלמה ${data.current_channel}`;
      }
    }
  } catch (e) {
    console.error('Failed to load camera channels:', e);
  }
}

async function quickSwitchCamera(channel) {
  const overlay = document.getElementById('video-offline-overlay');
  const overlayMsg = document.getElementById('offline-overlay-msg');
  if (overlay) {
    overlay.classList.remove('hidden');
    if (overlayMsg) overlayMsg.innerText = `מעביר למצלמה ${channel}...`;
  }

  showToast(`מעביר למצלמה ${channel}...`, 'info');

  try {
    const res = await fetch('/api/camera/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: parseInt(channel) })
    });

    if (res.ok) {
      const data = await res.json();
      showToast(`עברת בהצלחה למצלמה ${channel}`, 'success');

      const title = document.getElementById('stream-header-title');
      if (title) title.innerText = `שידור חי • מצלמה ${channel}`;

      const sel = document.getElementById('camera-select');
      if (sel) sel.value = channel;

      // Refresh live video stream
      setTimeout(() => {
        const img = document.getElementById('live-video-stream');
        if (img) {
          const modeParam = currentStreamMode === 'matrix' ? 'mode=matrix' : 'mode=single';
          img.src = `/api/video_feed?${modeParam}&t=${Date.now()}`;
        }
        if (overlay) overlay.classList.add('hidden');
      }, 600);
    } else {
      showToast('שגיאה בהחלפת ערוץ מצלמה ב-NVR', 'error');
      if (overlay) overlay.classList.add('hidden');
    }
  } catch (err) {
    console.error('Switch camera error:', err);
    showToast('שגיאת תקשורת עם השרת', 'error');
    if (overlay) overlay.classList.add('hidden');
  }
}

// --- Tab Switching Navigation ---
function switchTab(tabId) {
  currentTab = tabId;
  localStorage.setItem('visiondine_active_tab', tabId);
  try {
    if (window.location.hash.replace('#', '') !== tabId) {
      history.replaceState(null, '', '#' + tabId);
    }
  } catch (e) {}

  const tabs = ['live', 'calibration', 'analytics', 'settings'];

  tabs.forEach(t => {
    const content = document.getElementById(`tab-content-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (content) {
      if (t === tabId) {
        content.classList.remove('hidden');
      } else {
        content.classList.add('hidden');
      }
    }
    if (btn) {
      if (t === tabId) {
        btn.className = 'nav-tab active flex items-center space-x-2 px-4 py-1.5 rounded-lg text-xs font-medium transition-all text-white bg-blue-600 shadow-sm';
      } else {
        btn.className = 'nav-tab flex items-center space-x-2 px-4 py-1.5 rounded-lg text-xs font-medium transition-all text-slate-400 hover:text-white hover:bg-slate-800/60';
      }
    }
  });

  if (tabId === 'calibration') {
    initCalibrationTool();
  } else if (tabId === 'analytics') {
    loadAnalyticsData();
  } else if (tabId === 'live') {
    // Force refresh MJPEG stream if disconnected
    const img = document.getElementById('live-video-stream');
    if (img) {
      const modeParam = currentStreamMode === 'matrix' ? 'mode=matrix' : 'mode=single';
      img.src = `/api/video_feed?${modeParam}&t=${Date.now()}`;
    }
  }

  // Refresh icons
  if (window.lucide) lucide.createIcons();
}

// --- Live WebSocket Synchronization ---
function initLiveWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  try {
    liveWs = new WebSocket(wsUrl);

    liveWs.onopen = () => {
      updateConnectionIndicator(true, 'מחובר בזמן אמת');
    };

    liveWs.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        handleLiveStateUpdate(state);
      } catch (e) {
        console.error('Error parsing live WS payload:', e);
      }
    };

    liveWs.onclose = () => {
      updateConnectionIndicator(false, 'מתחבר מחדש...');
      setTimeout(initLiveWebSocket, 2500);
    };

    liveWs.onerror = () => {
      updateConnectionIndicator(false, 'מנותק מהשרת');
    };
  } catch (err) {
    console.warn('WebSocket not available, falling back to REST polling:', err);
    setInterval(pollLiveStateREST, 1000);
  }
}

async function pollLiveStateREST() {
  try {
    const res = await fetch('/api/live');
    if (res.ok) {
      const state = await res.json();
      handleLiveStateUpdate(state);
      updateConnectionIndicator(true, 'סנכרון פעיל (HTTP)');
    }
  } catch (e) {
    updateConnectionIndicator(false, 'לא מקוון');
  }
}

function updateConnectionIndicator(isConnected, text) {
  const pill = document.getElementById('connection-pill');
  const label = document.getElementById('connection-label');
  if (!pill || !label) return;

  if (isConnected) {
    pill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400';
    label.innerText = text;
  } else {
    pill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400';
    label.innerText = text;
  }
}

// --- Handle State Machine & Dashboard UI Updates ---
function handleLiveStateUpdate(state) {
  if (!state) return;

  // 1. Update Connection & Performance Badges
  if (state.connection) {
    const fpsBadge = document.getElementById('fps-badge');
    if (fpsBadge) {
      fpsBadge.innerText = `עיבוד AI: ${state.connection.inference_fps} FPS | ${state.connection.stream_fps} שידור`;
    }
    const resBadge = document.getElementById('stream-resolution-badge');
    if (resBadge && state.connection.camera_channel) {
      resBadge.innerText = `ערוץ NVR: ${state.connection.camera_channel}`;
    }
  }

  // 2. Update Top KPI Cards
  if (state.kpis) {
    const total = state.kpis.total_tables || 0;
    const occupied = state.kpis.occupied_tables || 0;
    const available = state.kpis.available_tables || 0;
    const diners = state.kpis.active_diners || 0;
    const pendingAlerts = state.kpis.pending_alerts_count || 0;
    const avgTurnover = state.kpis.avg_turnover_minutes || 0;
    const dinersToday = state.kpis.total_diners_today || 0;
    const sessionsToday = state.kpis.completed_sessions_today || 0;

    setText('kpi-total-tables', total);
    setText('kpi-occupied', occupied);
    setText('kpi-available', available);
    setText('kpi-active-diners', diners);
    setText('kpi-pending-alerts', pendingAlerts);
    setText('kpi-avg-turnover', avgTurnover > 0 ? avgTurnover : '--');
    setText('kpi-total-diners-today', dinersToday);
    setText('kpi-completed-sessions', `${sessionsToday} שולחנות הוגשו היום`);

    // Occupancy bar percentage
    const occPct = total > 0 ? (occupied / total) * 100 : 0;
    const occBar = document.getElementById('kpi-occupancy-bar');
    const availBar = document.getElementById('kpi-available-bar');
    if (occBar) occBar.style.width = `${occPct}%`;
    if (availBar) availBar.style.width = `${100 - occPct}%`;

    // Alert card styling
    const alertDot = document.getElementById('alert-pulse-dot');
    const cardAlerts = document.getElementById('card-service-alerts');
    if (pendingAlerts > 0) {
      if (alertDot) alertDot.classList.remove('hidden');
      if (cardAlerts) cardAlerts.classList.add('alert-pulse');
      if (pendingAlerts > lastAlertCount) {
        playAlertSound();
      }
    } else {
      if (alertDot) alertDot.classList.add('hidden');
      if (cardAlerts) cardAlerts.classList.remove('alert-pulse');
    }
    lastAlertCount = pendingAlerts;
  }

  // 3. Render Live Alerts Panel
  renderLiveAlerts(state.alerts || []);

  // 4. Render Live Floor Grid Cards
  renderLiveTables(state.tables || []);
}

function renderLiveAlerts(alerts) {
  const container = document.getElementById('live-alerts-list');
  const badge = document.getElementById('alerts-badge-count');
  if (!container) return;

  if (badge) badge.innerText = `${alerts.length} פעילות`;

  if (alerts.length === 0) {
    container.innerHTML = `
      <div class="p-3 rounded-xl bg-surface-850/60 border border-slate-800 text-center text-xs text-slate-500">
        אין התראות שירות פעילות כרגע. כל השולחנות מטופלים!
      </div>
    `;
    return;
  }

  container.innerHTML = alerts.map(a => `
    <div class="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs transition">
      <div class="flex items-center space-x-2.5 space-x-reverse">
        <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping"></span>
        <div>
          <div class="font-bold text-amber-300">${a.name}</div>
          <div class="text-[11px] text-slate-400">ממתין למלצר: <span class="font-mono text-amber-400 font-bold">${a.elapsed_formatted}</span></div>
        </div>
      </div>
      <button onclick="acknowledgeService('${a.id}')"
              class="px-2.5 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 text-[11px] font-semibold transition">
        אישור שירות
      </button>
    </div>
  `).join('');
}

function renderLiveTables(tables) {
  const container = document.getElementById('live-tables-container');
  const countLbl = document.getElementById('live-table-count');
  if (!container) return;

  if (countLbl) countLbl.innerText = `${tables.length} שולחנות`;

  if (tables.length === 0) {
    container.innerHTML = '<div class="text-slate-500 text-center py-4 text-xs">לא הוגדרו שולחנות עדיין. עברו ללשונית "כיול אזורי שולחן (ROI)" להוספה.</div>';
    return;
  }

  container.innerHTML = tables.map(t => {
    let statusClass = 'status-pill-free';
    let statusText = 'פנוי';
    let cardBorder = 'border-slate-800';

    if (t.status === 'OCCUPIED') {
      statusClass = 'status-pill-occupied';
      statusText = 'תפוס';
      cardBorder = 'border-rose-900/40';
    } else if (t.status === 'NEEDS_SERVICE') {
      statusClass = 'status-pill-alert';
      statusText = 'דרוש שירות!';
      cardBorder = 'border-amber-600/50 alert-pulse';
    }

    return `
      <div class="p-3 rounded-xl bg-surface-850 border ${cardBorder} flex items-center justify-between transition hover:border-slate-700">
        <div class="space-y-1">
          <div class="flex items-center space-x-2 space-x-reverse">
            <span class="font-bold text-white text-xs">${t.name}</span>
            <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${statusClass}">
              ${statusText}
            </span>
          </div>
          <div class="text-[11px] text-slate-400 flex items-center space-x-3 space-x-reverse">
            <span>סועדים: <b class="text-slate-200">${t.headcount}/${t.capacity}</b></span>
            <span>משך ישיבה: <b class="font-mono text-slate-200">${t.status === 'FREE' ? '--:--' : t.elapsed_formatted}</b></span>
          </div>
        </div>

        <div class="flex items-center space-x-2 space-x-reverse">
          ${t.status === 'NEEDS_SERVICE' ? `
            <button onclick="acknowledgeService('${t.id}')"
                    class="p-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-[10px] font-semibold transition" title="אישור שירות">
              טופל
            </button>
          ` : ''}
          <button onclick="switchTab('calibration'); selectCalibrationTable('${t.id}');"
                  class="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800" title="עריכת אזור שולחן">
            <i data-lucide="edit-3" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>
    `;
  }).join('');

  if (window.lucide) lucide.createIcons();
}

async function acknowledgeService(tableId) {
  try {
    const res = await fetch(`/api/tables/${tableId}/acknowledge_service`, { method: 'POST' });
    if (res.ok) {
      showToast(`התראת שירות אושרה עבור שולחן ${tableId}`, 'success');
    }
  } catch (err) {
    console.error('Failed to acknowledge service:', err);
  }
}

async function refreshKPIs() {
  try {
    const res = await fetch('/api/analytics/kpis');
    if (res.ok) {
      const kpis = await res.json();
      setText('kpi-avg-turnover', kpis.avg_turnover_minutes > 0 ? kpis.avg_turnover_minutes : '--');
      setText('kpi-total-diners-today', kpis.total_diners_today);
      setText('kpi-completed-sessions', `${kpis.completed_sessions_today} שולחנות הוגשו היום`);
    }
  } catch (e) { }
}

// --- Settings & Ossia VMS Configuration ---
function initSettingsListeners() {
  const radioSrc = document.querySelectorAll('input[name="video_source"]');
  radioSrc.forEach(r => {
    r.addEventListener('change', () => {
      toggleSettingsSourcePanels(r.value);
    });
  });

  // Sliders label updates
  bindSlider('cfg-target-fps', 'lbl-target-fps', val => `${parseFloat(val).toFixed(1)} FPS`);
  bindSlider('cfg-debounce', 'lbl-debounce', val => `${parseFloat(val).toFixed(1)} sec`);
  bindSlider('cfg-alert-thresh', 'lbl-alert-thresh', val => `${val} sec (${Math.round(val / 60)} min)`);
  bindSlider('cfg-conf', 'lbl-conf', val => `${parseFloat(val).toFixed(2)}`);

  // Auto-generate RTSP URL when Ossia fields change
  ['cfg-ossia-ip', 'cfg-ossia-port', 'cfg-ossia-ch', 'cfg-ossia-user', 'cfg-ossia-pass', 'cfg-ossia-stream'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateGeneratedRTSPUrl);
  });
}

function bindSlider(sliderId, labelId, formatFn) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (slider && label) {
    slider.addEventListener('input', () => {
      label.innerText = formatFn(slider.value);
    });
  }
}

function toggleSettingsSourcePanels(source) {
  const rtspPanel = document.getElementById('settings-rtsp-panel');
  const filePanel = document.getElementById('settings-file-panel');

  if (source === 'rtsp') {
    if (rtspPanel) rtspPanel.classList.remove('hidden');
    if (filePanel) filePanel.classList.add('hidden');
  } else if (source === 'file') {
    if (rtspPanel) rtspPanel.classList.add('hidden');
    if (filePanel) filePanel.classList.remove('hidden');
  } else {
    // sample
    if (rtspPanel) rtspPanel.classList.add('hidden');
    if (filePanel) filePanel.classList.add('hidden');
  }
}

function updateGeneratedRTSPUrl() {
  const ip = document.getElementById('cfg-ossia-ip').value.trim() || '192.168.1.100';
  const port = document.getElementById('cfg-ossia-port').value.trim() || '554';
  const ch = document.getElementById('cfg-ossia-ch').value.trim() || '1';
  const user = document.getElementById('cfg-ossia-user').value.trim() || 'admin';
  const pass = document.getElementById('cfg-ossia-pass').value.trim();
  const stream = document.getElementById('cfg-ossia-stream').value || '1';

  const cred = (user && pass) ? `${user}:${pass}@` : (user ? `${user}@` : '');
  const stype = (stream === '0' || stream === 0) ? 'main' : 'sub';
  const url = `rtsp://${cred}${ip}:${port}/chID=${ch}&streamType=${stype}&linkType=tcp`;
  document.getElementById('cfg-rtsp-url').value = url;
}

async function loadSystemSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) return;
    const cfg = await res.json();

    // Select video source radio
    const srcRadio = document.querySelector(`input[name="video_source"][value="${cfg.video_source}"]`);
    if (srcRadio) {
      srcRadio.checked = true;
      toggleSettingsSourcePanels(cfg.video_source);
    }

    setVal('cfg-rtsp-url', cfg.rtsp_url);
    setVal('cfg-video-file-path', cfg.video_file_path);
    setVal('cfg-target-fps', cfg.target_fps);
    setVal('cfg-debounce', cfg.debounce_seconds);
    setVal('cfg-alert-thresh', cfg.service_alert_threshold_seconds);
    setVal('cfg-conf', cfg.confidence_threshold);

    // Trigger slider labels
    document.getElementById('cfg-target-fps')?.dispatchEvent(new Event('input'));
    document.getElementById('cfg-debounce')?.dispatchEvent(new Event('input'));
    document.getElementById('cfg-alert-thresh')?.dispatchEvent(new Event('input'));
    document.getElementById('cfg-conf')?.dispatchEvent(new Event('input'));

    // Populate Ossia fields if available
    setVal('cfg-ossia-ip', cfg.ossia_ip || '192.168.1.100');
    setVal('cfg-ossia-port', cfg.ossia_port || 554);
    setVal('cfg-ossia-ch', cfg.ossia_channel || 1);
    setVal('cfg-ossia-user', cfg.ossia_user || 'admin');
    setVal('cfg-ossia-pass', cfg.ossia_pass || '');
    setVal('cfg-ossia-stream', cfg.ossia_stream !== undefined ? cfg.ossia_stream : 1);

    loadTelegramSettings();
    checkFirebaseStatus();
  } catch (err) {
    console.error('Failed to load system settings:', err);
  }
}

async function loadTelegramSettings() {
  try {
    const res = await fetch('/api/settings/telegram');
    if (res.ok) {
      const data = await res.json();
      setVal('cfg-telegram-token', data.bot_token || '');
      setVal('cfg-telegram-chat', data.chat_id || '');
      const chk = document.getElementById('cfg-telegram-enabled');
      if (chk) chk.checked = !!data.enabled;
    }
  } catch (e) { }
}

async function checkFirebaseStatus() {
  try {
    const res = await fetch('/api/settings/firebase');
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById('badge-firebase-status');
      if (badge) {
        if (data.configured) {
          badge.className = 'text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded font-mono';
          badge.innerText = 'מחובר ומסונכרן לענן!';
        } else {
          badge.className = 'text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded font-mono';
          badge.innerText = 'ממתין לקובץ הגדרות';
        }
      }
    }
  } catch (e) { }
}

async function testTelegramAlert() {
  const token = document.getElementById('cfg-telegram-token').value.trim();
  const chat = document.getElementById('cfg-telegram-chat').value.trim();
  const enabled = document.getElementById('cfg-telegram-enabled').checked;

  if (!token || !chat) {
    showToast('נא להזין Bot Token ו-Chat ID של טלגרם', 'error');
    return;
  }

  showToast('שולח הודעת בדיקה לטלגרם...', 'info');
  try {
    const res = await fetch('/api/settings/telegram', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot_token: token, chat_id: chat, enabled: enabled })
    });
    const data = await res.json();
    if (data.test_sent) {
      showToast('הודעת בדיקה נשלחה בהצלחה לטלפון!', 'success');
    } else {
      showToast('נכשל במשלוח. ודא שהתחלת שיחה עם הבוט בטלגרם.', 'error');
    }
  } catch (e) {
    showToast('שגיאה בתקשורת עם השרת', 'error');
  }
}

async function saveSystemSettings() {
  const selectedSource = document.querySelector('input[name="video_source"]:checked')?.value || 'sample';

  const payload = {
    video_source: selectedSource,
    rtsp_url: document.getElementById('cfg-rtsp-url').value.trim(),
    video_file_path: document.getElementById('cfg-video-file-path').value.trim(),
    target_fps: parseFloat(document.getElementById('cfg-target-fps').value),
    debounce_seconds: parseFloat(document.getElementById('cfg-debounce').value),
    service_alert_threshold_seconds: parseInt(document.getElementById('cfg-alert-thresh').value),
    confidence_threshold: parseFloat(document.getElementById('cfg-conf').value),
    ossia_ip: document.getElementById('cfg-ossia-ip').value.trim(),
    ossia_port: parseInt(document.getElementById('cfg-ossia-port').value),
    ossia_channel: parseInt(document.getElementById('cfg-ossia-ch').value),
    ossia_user: document.getElementById('cfg-ossia-user').value.trim(),
    ossia_pass: document.getElementById('cfg-ossia-pass').value.trim(),
    ossia_stream: parseInt(document.getElementById('cfg-ossia-stream').value)
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      // Save Telegram settings if present
      const tToken = document.getElementById('cfg-telegram-token')?.value.trim();
      const tChat = document.getElementById('cfg-telegram-chat')?.value.trim();
      const tEnabled = document.getElementById('cfg-telegram-enabled')?.checked;
      if (tToken || tChat) {
        await fetch('/api/settings/telegram', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bot_token: tToken, chat_id: tChat, enabled: tEnabled })
        });
      }

      showToast('ההגדרות נשמרו והוחלו בהצלחה! מפעיל מחדש את השידור במידת הצורך.', 'success');
      // Refresh live video
      setTimeout(() => {
        const img = document.getElementById('live-video-stream');
        if (img) img.src = '/api/video_feed?t=' + Date.now();
      }, 1000);
    } else {
      showToast('שגיאה בשמירת ההגדרות', 'error');
    }
  } catch (err) {
    console.error('Save settings error:', err);
    showToast('נכשל החיבור לשרת', 'error');
  }
}

async function testRTSPConnection() {
  const rtspUrl = document.getElementById('cfg-rtsp-url').value.trim();
  const resultDiv = document.getElementById('rtsp-test-result');
  const btn = document.getElementById('btn-test-rtsp');

  if (!rtspUrl) {
    showToast('נא להזין כתובת RTSP לבדיקה', 'error');
    return;
  }

  if (resultDiv) {
    resultDiv.classList.remove('hidden');
    resultDiv.className = 'text-xs mt-1.5 text-blue-400 font-medium';
    resultDiv.innerText = 'בודק חיבור שידור RTSP ל-NVR (המתנה עד 5 שניות)...';
  }
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/settings/test_rtsp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rtsp_url: rtspUrl })
    });
    const data = await res.json();

    if (data.success) {
      resultDiv.className = 'text-xs mt-1.5 text-emerald-400 font-medium';
      resultDiv.innerText = `✓ ${data.message}`;
      showToast('שידור ה-RTSP אומת בהצלחה!', 'success');
    } else {
      resultDiv.className = 'text-xs mt-1.5 text-rose-400 font-medium';
      resultDiv.innerText = `✗ ${data.message}`;
      showToast('בדיקת RTSP נכשלה. נא לבדוק פרטי NVR ורשת.', 'error');
    }
  } catch (err) {
    resultDiv.className = 'text-xs mt-1.5 text-rose-400 font-medium';
    resultDiv.innerText = '✗ שגיאת רשת או פסק זמן מהשרת.';
  } finally {
    if (btn) btn.disabled = false;
  }
}

function toggleVideoFullscreen() {
  const container = document.getElementById('video-container');
  if (!container) return;
  if (!document.fullscreenElement) {
    container.requestFullscreen().catch(err => alert(`Fullscreen error: ${err.message}`));
  } else {
    document.exitFullscreen();
  }
}

// --- Audio Tone Generator for Waiter Service Alert ---
function playAlertSound() {
  if (!liveAudioEnabled) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime); // A5 note
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) { }
}

// --- Utility Helpers ---
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  const bg = type === 'success' ? 'bg-emerald-600' : (type === 'error' ? 'bg-rose-600' : 'bg-blue-600');
  toast.className = `${bg} text-white text-xs px-4 py-2.5 rounded-xl shadow-lg flex items-center space-x-2 pointer-events-auto transition transform translate-y-2 opacity-0 duration-300`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('translate-y-2', 'opacity-0');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
