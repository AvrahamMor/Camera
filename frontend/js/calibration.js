/**
 * Interactive Table Zone (ROI) Calibration Tool
 * Allows restaurant managers to grab live camera snapshots,
 * draw custom polygons with mouse clicks, adjust points, and save metadata.
 */

let calCanvas, calCtx;
let calSnapshotImage = null;
let calTables = [];
let calSelectedTableId = null;
let calIsDrawingNew = false;
let calNewPolygon = []; // Array of [x_norm, y_norm]
let calDraggingPoint = null; // { tableId: str, pointIndex: int }
let calHoveredPoint = null;

function initCalibrationTool() {
  calCanvas = document.getElementById('roi-canvas');
  if (!calCanvas) return;
  calCtx = calCanvas.getContext('2d');

  // Event Listeners for Canvas
  calCanvas.addEventListener('mousedown', onCanvasMouseDown);
  calCanvas.addEventListener('mousemove', onCanvasMouseMove);
  calCanvas.addEventListener('mouseup', onCanvasMouseUp);
  calCanvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    onCanvasRightClick(e);
  });

  // Restore saved camera (fallback to ch 2 if saved channel 1/7 was removed)
  const activeChannels = [2, 3, 4, 5, 6, 8];
  let savedCam = localStorage.getItem('visiondine_active_calib_cam');
  if (!savedCam || !activeChannels.includes(parseInt(savedCam))) {
    savedCam = '2';
    localStorage.setItem('visiondine_active_calib_cam', savedCam);
  }
  const calibSel = document.getElementById('calib-camera-select');
  if (calibSel) {
    calibSel.value = savedCam;
  }

  // Auto-sync between table name and table id
  const idInput = document.getElementById('editor-table-id');
  const nameInput = document.getElementById('editor-table-name');
  if (nameInput && idInput) {
    nameInput.addEventListener('input', () => {
      const match = nameInput.value.match(/\d+/);
      if (match) {
        const num = match[0];
        if (!idInput.value || idInput.value.startsWith('table_') || /^\d+$/.test(idInput.value)) {
          idInput.value = `table_${num}`;
        }
      }
    });

    idInput.addEventListener('input', () => {
      const match = idInput.value.match(/\d+/);
      if (match) {
        const num = match[0];
        if (!nameInput.value || nameInput.value.startsWith('שולחן ') || /^\d+$/.test(nameInput.value)) {
          nameInput.value = `שולחן ${num}`;
        }
      }
    });
  }

  // Load configured tables from backend
  loadCalibrationTables();
  // Grab snapshot automatically
  calibrationGrabSnapshot(savedCam || null);
}

async function loadCalibrationTables() {
  try {
    const res = await fetch('/api/tables');
    if (res.ok) {
      calTables = await res.json();
      renderCalibrationSidebar();
      redrawCalibrationCanvas();
      
      const calibSel = document.getElementById('calib-camera-select');
      const currentCam = parseInt(calibSel ? calibSel.value : 2) || 2;
      if (calTables.length > 0 && !calSelectedTableId) {
        // Only select a table belonging to current camera so camera doesn't jump
        const tableOnCam = calTables.find(t => (t.camera_channel || 2) === currentCam);
        if (tableOnCam) {
          selectCalibrationTable(tableOnCam.id, false);
        }
      }
    }
  } catch (err) {
    console.error('Failed to load tables for calibration:', err);
  }
}

async function calibrationGrabSnapshot(channel = null) {
  const loading = document.getElementById('canvas-loading');
  if (loading) loading.classList.remove('hidden');

  const calibSel = document.getElementById('calib-camera-select');
  const ch = parseInt(channel || (calibSel ? calibSel.value : 2)) || 2;

  try {
    const res = await fetch(`/api/snapshot?channel=${ch}&clean=true&t=` + Date.now());
    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        calSnapshotImage = img;
        calCanvas.width = img.naturalWidth || 1280;
        calCanvas.height = img.naturalHeight || 720;
        if (loading) loading.classList.add('hidden');
        redrawCalibrationCanvas();
        showToast(`תמונת מצלמה ${ch} נטענה בהצלחה!`, 'success');
      };
      img.src = url;
    } else {
      throw new Error('Snapshot endpoint returned status ' + res.status);
    }
  } catch (err) {
    console.error('Failed to fetch snapshot:', err);
    if (loading) loading.classList.add('hidden');
    showToast(`נכשל בצילום פריים ממצלמה ${ch}`, 'error');
  }
}

function onCalibrationCameraChange(newChannel) {
  const ch = parseInt(newChannel) || 2;
  localStorage.setItem('visiondine_active_calib_cam', ch);

  const camInput = document.getElementById('editor-table-camera');
  if (camInput) camInput.value = ch;

  calSnapshotImage = null;
  calSelectedTableId = null;
  redrawCalibrationCanvas();
  renderCalibrationSidebar();
  calibrationGrabSnapshot(ch);
}

function redrawCalibrationCanvas() {
  if (!calCtx || !calCanvas) return;
  const w = calCanvas.width;
  const h = calCanvas.height;

  // Clear canvas
  calCtx.clearRect(0, 0, w, h);

  // 1. Draw snapshot background image or placeholder
  if (calSnapshotImage) {
    calCtx.drawImage(calSnapshotImage, 0, 0, w, h);
  } else {
    calCtx.fillStyle = '#0f172a';
    calCtx.fillRect(0, 0, w, h);
    calCtx.fillStyle = '#64748b';
    calCtx.font = '16px Heebo, sans-serif';
    calCtx.textAlign = 'center';
    calCtx.fillText('אין תמונת מצלמה זמינה. לחץ "רענן תמונה מהמצלמה"', w / 2, h / 2);
  }

  // 2. Draw configured tables belonging to this camera
  const calibSel = document.getElementById('calib-camera-select');
  const activeCam = calibSel ? parseInt(calibSel.value) : 1;
  calTables.forEach(t => {
    const tCam = t.camera_channel || 1;
    if (tCam === activeCam) {
      const isSelected = (t.id === calSelectedTableId);
      drawPolygon(t.polygon, isSelected ? '#3b82f6' : '#10b981', isSelected, t.name, t.id);
    }
  });

  // 3. Draw in-progress new polygon
  if (calIsDrawingNew && calNewPolygon.length > 0) {
    drawPolygon(calNewPolygon, '#f59e0b', true, 'שולחן חדש (בשרטוט...)', null, false);
  }
}

function drawPolygon(normPoly, strokeColor, isSelected, label, tableId, isClosed = true) {
  if (!normPoly || normPoly.length === 0) return;
  const w = calCanvas.width;
  const h = calCanvas.height;

  calCtx.beginPath();
  const firstX = normPoly[0][0] * w;
  const firstY = normPoly[0][1] * h;
  calCtx.moveTo(firstX, firstY);

  for (let i = 1; i < normPoly.length; i++) {
    calCtx.lineTo(normPoly[i][0] * w, normPoly[i][1] * h);
  }

  if (isClosed && normPoly.length >= 3) {
    calCtx.closePath();
    calCtx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.28)' : 'rgba(16, 185, 129, 0.18)';
    calCtx.fill();
  }

  calCtx.strokeStyle = strokeColor;
  calCtx.lineWidth = isSelected ? 3 : 2;
  calCtx.stroke();

  // Draw Vertex Handles
  normPoly.forEach((pt, idx) => {
    const px = pt[0] * w;
    const py = pt[1] * h;
    const isHovered = (calHoveredPoint && calHoveredPoint.tableId === tableId && calHoveredPoint.pointIndex === idx);

    calCtx.beginPath();
    calCtx.arc(px, py, isHovered ? 7 : (isSelected ? 5 : 4), 0, Math.PI * 2);
    calCtx.fillStyle = isSelected ? '#ffffff' : strokeColor;
    calCtx.fill();
    calCtx.strokeStyle = '#0f172a';
    calCtx.lineWidth = 1.5;
    calCtx.stroke();
  });

  // Draw Label Badge
  if (normPoly.length >= 3) {
    let avgX = 0, avgY = 0;
    normPoly.forEach(pt => { avgX += pt[0]; avgY += pt[1]; });
    avgX = (avgX / normPoly.length) * w;
    avgY = (avgY / normPoly.length) * h;

    calCtx.font = 'bold 12px Inter, sans-serif';
    calCtx.textAlign = 'center';
    calCtx.textBaseline = 'middle';
    const textWidth = calCtx.measureText(label).width;

    calCtx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    calCtx.fillRect(avgX - textWidth / 2 - 8, avgY - 12, textWidth + 16, 24);
    calCtx.strokeStyle = strokeColor;
    calCtx.lineWidth = 1;
    calCtx.strokeRect(avgX - textWidth / 2 - 8, avgY - 12, textWidth + 16, 24);

    calCtx.fillStyle = '#ffffff';
    calCtx.fillText(label, avgX, avgY);
  }
}

// Coordinate conversions
function getCanvasNormalizedPos(e) {
  const rect = calCanvas.getBoundingClientRect();
  const scaleX = calCanvas.width / rect.width;
  const scaleY = calCanvas.height / rect.height;
  const canvasX = (e.clientX - rect.left) * scaleX;
  const canvasY = (e.clientY - rect.top) * scaleY;
  const normX = Math.max(0, Math.min(1, canvasX / calCanvas.width));
  const normY = Math.max(0, Math.min(1, canvasY / calCanvas.height));
  return { normX, normY, canvasX, canvasY };
}

function findPointNear(canvasX, canvasY, radiusPx = 12) {
  const w = calCanvas.width;
  const h = calCanvas.height;

  // Check new in-progress polygon
  if (calIsDrawingNew) {
    for (let i = 0; i < calNewPolygon.length; i++) {
      const px = calNewPolygon[i][0] * w;
      const py = calNewPolygon[i][1] * h;
      const dist = Math.hypot(px - canvasX, py - canvasY);
      if (dist <= radiusPx) {
        return { tableId: '__new__', pointIndex: i };
      }
    }
    return null;
  }

  // Check selected table points first
  if (calSelectedTableId) {
    const table = calTables.find(t => t.id === calSelectedTableId);
    if (table && table.polygon) {
      for (let i = 0; i < table.polygon.length; i++) {
        const px = table.polygon[i][0] * w;
        const py = table.polygon[i][1] * h;
        const dist = Math.hypot(px - canvasX, py - canvasY);
        if (dist <= radiusPx) {
          return { tableId: table.id, pointIndex: i };
        }
      }
    }
  }

  // Check all other tables
  for (const t of calTables) {
    for (let i = 0; i < t.polygon.length; i++) {
      const px = t.polygon[i][0] * w;
      const py = t.polygon[i][1] * h;
      const dist = Math.hypot(px - canvasX, py - canvasY);
      if (dist <= radiusPx) {
        return { tableId: t.id, pointIndex: i };
      }
    }
  }

  return null;
}

function onCanvasMouseDown(e) {
  if (e.button !== 0) return; // Left click only
  const pos = getCanvasNormalizedPos(e);
  const nearPoint = findPointNear(pos.canvasX, pos.canvasY);

  if (nearPoint) {
    // Start dragging vertex
    calDraggingPoint = nearPoint;
    if (nearPoint.tableId !== '__new__' && nearPoint.tableId !== calSelectedTableId) {
      selectCalibrationTable(nearPoint.tableId);
    }
    return;
  }

  // If drawing new table or adding vertices to selected table
  if (calIsDrawingNew) {
    calNewPolygon.push([round4(pos.normX), round4(pos.normY)]);
    updateVerticesCount();
    redrawCalibrationCanvas();
  } else if (calSelectedTableId) {
    // Check if clicked inside another table's polygon to select it
    let clickedTable = null;
    for (const t of calTables) {
      if (pointInPoly([pos.normX, pos.normY], t.polygon)) {
        clickedTable = t;
        break;
      }
    }
    if (clickedTable) {
      selectCalibrationTable(clickedTable.id);
    } else {
      // Add vertex to currently selected table
      const table = calTables.find(t => t.id === calSelectedTableId);
      if (table) {
        table.polygon.push([round4(pos.normX), round4(pos.normY)]);
        updateVerticesCount();
        redrawCalibrationCanvas();
      }
    }
  }
}

function onCanvasMouseMove(e) {
  const pos = getCanvasNormalizedPos(e);

  if (calDraggingPoint) {
    // Update dragged vertex
    if (calDraggingPoint.tableId === '__new__') {
      calNewPolygon[calDraggingPoint.pointIndex] = [round4(pos.normX), round4(pos.normY)];
    } else {
      const table = calTables.find(t => t.id === calDraggingPoint.tableId);
      if (table) {
        table.polygon[calDraggingPoint.pointIndex] = [round4(pos.normX), round4(pos.normY)];
      }
    }
    redrawCalibrationCanvas();
    return;
  }

  // Hover detection
  const near = findPointNear(pos.canvasX, pos.canvasY);
  if (near !== calHoveredPoint) {
    calHoveredPoint = near;
    calCanvas.style.cursor = near ? 'grab' : 'crosshair';
    redrawCalibrationCanvas();
  }
}

function onCanvasMouseUp(e) {
  if (calDraggingPoint) {
    calDraggingPoint = null;
    redrawCalibrationCanvas();
  }
}

function onCanvasRightClick(e) {
  // Right click completes new polygon drawing
  if (calIsDrawingNew && calNewPolygon.length >= 3) {
    calIsDrawingNew = false;
    showToast('פוליגון נסגר. הגדר שם שולחן ומספר מושבים ולחץ שמור.', 'info');
    redrawCalibrationCanvas();
  }
}

function calibrationAddNewTable() {
  calIsDrawingNew = true;
  calNewPolygon = [];
  calSelectedTableId = '__new__';

  const nextNum = (calTables.length + 1).toString().padStart(2, '0');
  const calibSel = document.getElementById('calib-camera-select');
  const currentCam = calibSel ? parseInt(calibSel.value) : 1;

  document.getElementById('editor-table-id').value = `table_${nextNum}`;
  document.getElementById('editor-table-name').value = `שולחן ${nextNum}`;
  const camInput = document.getElementById('editor-table-camera');
  if (camInput) camInput.value = currentCam;
  document.getElementById('editor-table-capacity').value = 4;
  document.getElementById('editor-table-alert-min').value = 3;
  updateVerticesCount();

  showToast(`הוספת שולחן למצלמה ${currentCam}: סמן את 4 הפינות על גבי התמונה. מקש ימני לסגירה.`, 'info');
  redrawCalibrationCanvas();
}

function selectCalibrationTable(tableId, allowCameraSwitch = true) {
  calIsDrawingNew = false;
  calSelectedTableId = tableId;
  const table = calTables.find(t => t.id === tableId);
  if (!table) return;

  const tableCam = table.camera_channel || 1;
  const calibSel = document.getElementById('calib-camera-select');
  if (allowCameraSwitch && calibSel && parseInt(calibSel.value) !== tableCam) {
    calibSel.value = tableCam;
    localStorage.setItem('visiondine_active_calib_cam', tableCam);
    calSnapshotImage = null;
    calibrationGrabSnapshot(tableCam);
  }

  document.getElementById('editor-table-id').value = table.id;
  document.getElementById('editor-table-name').value = table.name;
  const camInput = document.getElementById('editor-table-camera');
  if (camInput) camInput.value = tableCam;
  document.getElementById('editor-table-capacity').value = table.capacity;
  document.getElementById('editor-table-alert-min').value = Math.round(table.service_threshold_sec / 60);
  updateVerticesCount();

  renderCalibrationSidebar();
  redrawCalibrationCanvas();
}

function updateVerticesCount() {
  const lbl = document.getElementById('editor-vertices-count');
  if (!lbl) return;
  if (calIsDrawingNew) {
    lbl.innerText = `${calNewPolygon.length} נקודות (בשרטוט...)`;
  } else if (calSelectedTableId) {
    const table = calTables.find(t => t.id === calSelectedTableId);
    lbl.innerText = table ? `${table.polygon.length} נקודות` : '0 נקודות';
  } else {
    lbl.innerText = '0 נקודות';
  }
}

function calibrationUndoPoint() {
  if (calIsDrawingNew && calNewPolygon.length > 0) {
    calNewPolygon.pop();
    updateVerticesCount();
    redrawCalibrationCanvas();
  } else if (calSelectedTableId) {
    const table = calTables.find(t => t.id === calSelectedTableId);
    if (table && table.polygon.length > 0) {
      table.polygon.pop();
      updateVerticesCount();
      redrawCalibrationCanvas();
    }
  }
}

function calibrationClearCurrentPolygon() {
  if (calIsDrawingNew) {
    calNewPolygon = [];
  } else if (calSelectedTableId) {
    const table = calTables.find(t => t.id === calSelectedTableId);
    if (table) table.polygon = [];
  }
  updateVerticesCount();
  redrawCalibrationCanvas();
}

async function calibrationSaveCurrentTable() {
  let tableId = document.getElementById('editor-table-id').value.trim();
  let tableName = document.getElementById('editor-table-name').value.trim();
  const capacity = parseInt(document.getElementById('editor-table-capacity').value) || 4;
  const alertMin = parseInt(document.getElementById('editor-table-alert-min').value) || 3;

  // Auto-normalize ID if user entered just a number e.g. "18" -> "table_18"
  if (/^\d+$/.test(tableId)) {
    tableId = `table_${tableId}`;
    document.getElementById('editor-table-id').value = tableId;
  }
  if (/^\d+$/.test(tableName)) {
    tableName = `שולחן ${tableName}`;
    document.getElementById('editor-table-name').value = tableName;
  }

  if (!tableId || !tableName) {
    showToast('נא להזין מזהה שולחן ושם תצוגה', 'error');
    return;
  }

  let polygonToSave = [];
  if (calIsDrawingNew) {
    polygonToSave = calNewPolygon;
  } else {
    const table = calTables.find(t => t.id === calSelectedTableId);
    polygonToSave = table ? table.polygon : [];
  }

  if (polygonToSave.length < 3) {
    showToast('פוליגון שולחן חייב לכלול לפחות 3 נקודות', 'error');
    return;
  }

  const calibSel = document.getElementById('calib-camera-select');
  const camInput = document.getElementById('editor-table-camera');
  const cameraChannel = calIsDrawingNew ? (parseInt(calibSel?.value) || 2) : (parseInt(camInput?.value) || parseInt(calibSel?.value) || 2);

  // If editing an existing table and changed its ID, pass old_id so backend deletes the previous ID and replaces it!
  const oldId = (calSelectedTableId && calSelectedTableId !== '__new__') ? calSelectedTableId : null;

  const payload = {
    id: tableId,
    name: tableName,
    camera_channel: cameraChannel,
    capacity: capacity,
    service_threshold_sec: alertMin * 60,
    polygon: polygonToSave,
    old_id: oldId
  };

  try {
    const res = await fetch('/api/tables', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      calIsDrawingNew = false;
      showToast(`שולחן "${tableName}" (${tableId}) נשמר בהצלחה!`, 'success');
      calSelectedTableId = tableId;
      await loadCalibrationTables();
      selectCalibrationTable(tableId, false);
    } else {
      const err = await res.json();
      showToast(`שגיאה: ${err.detail || 'לא ניתן לשמור שולחן'}`, 'error');
    }
  } catch (err) {
    console.error('Save table error:', err);
    showToast('נכשל החיבור לשרת', 'error');
  }
}

async function calibrationDeleteCurrentTable() {
  const targetId = calSelectedTableId || document.getElementById('editor-table-id')?.value.trim();

  if (!targetId) {
    showToast('נא לבחור שולחן מהרשימה כדי למחוק אותו', 'info');
    return;
  }

  if (targetId === '__new__') {
    calIsDrawingNew = false;
    calNewPolygon = [];
    calSelectedTableId = null;
    document.getElementById('editor-table-id').value = '';
    document.getElementById('editor-table-name').value = '';
    updateVerticesCount();
    redrawCalibrationCanvas();
    showToast('הוספת שולחן בוטלה', 'info');
    return;
  }

  const table = calTables.find(t => t.id === targetId);
  const displayName = table ? table.name : targetId;

  if (!confirm(`האם אתה בטוח שברצונך למחוק את "${displayName}"?`)) return;

  try {
    const res = await fetch(`/api/tables/${targetId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast(`שולחן "${displayName}" נמחק בהצלחה!`, 'success');
      calSelectedTableId = null;
      document.getElementById('editor-table-id').value = '';
      document.getElementById('editor-table-name').value = '';
      updateVerticesCount();
      await loadCalibrationTables();
      redrawCalibrationCanvas();
    } else {
      const err = await res.json();
      showToast(`שגיאה במחיקה: ${err.detail || 'לא ניתן למחוק'}`, 'error');
    }
  } catch (err) {
    console.error('Delete error:', err);
    showToast('שגיאה בחיבור לשרת', 'error');
  }
}

async function calibrationResetDefaults() {
  if (!confirm('האם לאפס את כל אזורי השולחנות להגדרות ברירת המחדל? שינויים מותאמים יוחלפו.')) return;
  try {
    const res = await fetch('/api/tables/reset', { method: 'POST' });
    if (res.ok) {
      showToast('אזורי ברירת המחדל שוחזרו בהצלחה!', 'success');
      await loadCalibrationTables();
    }
  } catch (err) {
    console.error('Reset error:', err);
    showToast('שגיאה באיפוס שולחנות', 'error');
  }
}

function renderCalibrationSidebar() {
  const container = document.getElementById('calibration-tables-list');
  if (!container) return;

  if (calTables.length === 0) {
    container.innerHTML = '<div class="text-slate-500 text-center py-2">לא הוגדרו שולחנות</div>';
    return;
  }

  const calibSel = document.getElementById('calib-camera-select');
  const activeCam = calibSel ? parseInt(calibSel.value) : 1;

  container.innerHTML = calTables.map(t => {
    const isSelected = (t.id === calSelectedTableId);
    const cam = t.camera_channel || 1;
    const isThisCam = (cam === activeCam);
    return `
      <div onclick="selectCalibrationTable('${t.id}')"
           class="p-2.5 rounded-lg cursor-pointer transition flex items-center justify-between ${isSelected ? 'bg-blue-600/20 border border-blue-500/50 text-white' : (isThisCam ? 'bg-surface-900 border border-slate-700 text-slate-200 hover:bg-slate-800' : 'bg-surface-950/60 border border-slate-900 text-slate-400 hover:bg-slate-900')}">
        <div>
          <div class="font-semibold text-xs flex items-center space-x-1.5 space-x-reverse">
            <span>${t.name}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded font-mono ${isThisCam ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40' : 'bg-slate-800 text-slate-400'}">מצלמה ${cam}</span>
          </div>
          <div class="text-[11px] text-slate-400 font-mono">${t.id} • ${t.capacity} מקומות</div>
        </div>
        <span class="text-[11px] text-slate-400">${t.polygon.length} נקודות</span>
      </div>
    `;
  }).join('');
}

function round4(val) {
  return Math.round(val * 10000) / 10000;
}

function pointInPoly(pt, poly) {
  const x = pt[0], y = pt[1];
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}
