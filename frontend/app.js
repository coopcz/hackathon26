const $ = id => document.getElementById(id);
const LABELS = {
  empty: {name: 'Empty', detail: 'Nobody inside', symbol: '○'},
  occupied_still: {name: 'Occupied · still', detail: 'Person seated or standing', symbol: '│'},
  occupied_moving: {name: 'Occupied · moving', detail: 'Person walking through space', symbol: '↗'}
};
const state = {
  rooms: [], roomId: localStorage.getItem('roomsense.room'), status: null,
  recordings: [], filter: 'all', lastRecordingActive: false, quality: new Map(),
  decisions: [], latestPacket: null, view: 'setup', step: 'hardware'
};

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
  if (!response.ok) {
    let body;
    try { body = await response.json(); } catch (_) {}
    throw new Error(body?.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function toast(message, bad = false) {
  const node = $('toast');
  node.textContent = message; node.className = `show${bad ? ' bad' : ''}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.className = '', 4200);
}

const selectedRoom = () => state.rooms.find(room => room.id === state.roomId) || null;
const activeRoom = () => state.status?.active_room || state.rooms.find(room => room.active) || null;
const countReady = (counts, minimum = 3) => Object.values(counts || {}).every(value => value >= minimum);
const fmt = (value, digits = 1) => Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
const shortDate = value => value ? new Date(value).toLocaleString([], {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}) : '—';

function switchView(name) {
  state.view = name;
  document.querySelectorAll('.view').forEach(node => node.classList.toggle('active', node.id === `${name}View`));
  document.querySelectorAll('.nav-item').forEach(node => node.classList.toggle('active', node.dataset.view === name));
  document.body.classList.remove('menu-open');
  if (name === 'data') renderRecordings();
}

function switchStep(name) {
  state.step = name;
  document.querySelectorAll('.step').forEach(node => node.classList.toggle('active', node.dataset.step === name));
  document.querySelectorAll('.stage-panel').forEach(node => node.classList.toggle('active', node.dataset.panel === name));
}

function renderRooms() {
  const list = $('roomList'); list.replaceChildren();
  if (!state.rooms.length) {
    const empty = document.createElement('p'); empty.className = 'no-rooms'; empty.textContent = 'No rooms mapped yet.'; list.append(empty);
  }
  state.rooms.forEach(room => {
    const button = document.createElement('button'); button.className = `room-option${room.id === state.roomId ? ' active' : ''}`;
    const dot = document.createElement('i'); const copy = document.createElement('div');
    const name = document.createElement('strong'); name.textContent = room.name;
    const detail = document.createElement('small'); detail.textContent = room.active ? 'Live model active' : room.validated ? 'Validated' : room.model_ready ? 'Model trained' : 'Setup in progress';
    copy.append(name, detail); button.append(dot, copy);
    button.onclick = () => selectRoom(room.id); list.append(button);
  });
  const room = selectedRoom(), picker = $('roomPicker');
  picker.replaceChildren();
  if (!state.rooms.length) { const option = document.createElement('option'); option.value = ''; option.textContent = 'No room selected'; picker.append(option); }
  state.rooms.forEach(item => { const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; picker.append(option); });
  picker.value = room?.id || '';
  $('setupRoomName').textContent = room?.name || 'Room setup';
  $('placementText').textContent = room?.placement || 'No placement notes saved for this room.';
  $('emptyState').hidden = state.rooms.length > 0;
  document.querySelectorAll('.view').forEach(view => { if (!state.rooms.length) view.classList.remove('active'); });
  if (state.rooms.length && !document.querySelector('.view.active')) switchView(state.view);
}

function selectRoom(roomId) {
  state.roomId = roomId; localStorage.setItem('roomsense.room', roomId);
  renderAll(); refreshRecordings(); document.body.classList.remove('menu-open');
}

function conditionMarkup(split, target) {
  const room = selectedRoom(); const counts = room?.counts?.[split] || {};
  return Object.entries(LABELS).map(([label, meta]) => {
    const count = counts[label] || 0, progress = Math.min(100, count / target * 100);
    return `<div class="condition-row"><div class="condition-title"><span class="condition-symbol">${meta.symbol}</span><div><strong>${meta.name}</strong><small>${meta.detail}</small></div></div><div class="progress-track"><i style="width:${progress}%"></i></div><div class="condition-count"><b>${count}</b> / ${target}</div><button class="button secondary record-condition" data-label="${label}" data-split="${split}">Record</button></div>`;
  }).join('');
}

function renderConditions() {
  $('trainingConditions').innerHTML = conditionMarkup('training', 10);
  $('holdoutConditions').innerHTML = conditionMarkup('holdout', 3);
  document.querySelectorAll('.record-condition').forEach(button => {
    button.disabled = !state.status?.serial?.connected || state.status?.recording?.active || !selectedRoom();
    button.onclick = () => startRecording(button.dataset.label, button.dataset.split);
  });
}

function renderWorkflow() {
  const room = selectedRoom(); if (!room) return;
  const serial = state.status?.serial || {}, quality = serial.quality || {};
  const trainCounts = room.counts?.training || {}, holdoutCounts = room.counts?.holdout || {};
  const hardwareDone = !!serial.connected && !!quality.healthy;
  const calibrationDone = countReady(trainCounts);
  const trained = !!room.model_ready, validated = !!room.validated, activated = !!room.active;
  const flags = [hardwareDone, calibrationDone, trained, validated, activated];
  $('completionValue').textContent = `${flags.filter(Boolean).length * 20}%`;
  document.querySelectorAll('.step').forEach((step, index) => {
    step.classList.toggle('complete', flags[index]);
    step.classList.toggle('attention', index === 0 && serial.connected && !quality.healthy);
  });

  $('hardwareState').textContent = !serial.connected ? 'Not connected' : quality.healthy ? 'Signal healthy' : 'Check signal';
  $('hardwareState').className = `state-label ${!serial.connected ? 'neutral' : quality.healthy ? 'good' : 'warn'}`;
  $('streamQuality').textContent = !quality.ready ? 'Waiting' : quality.healthy ? 'Healthy' : 'Attention';
  $('streamQuality').className = !quality.ready ? '' : quality.healthy ? 'good' : 'warn';
  $('streamReason').textContent = (quality.problems || []).join(' · ') || (quality.ready ? 'Live CSI is stable' : 'Connect a receiver to begin');
  $('packetRate').textContent = fmt(quality.recent_packets_per_second ?? serial.packets_per_second);
  $('rssi').textContent = fmt(quality.rssi_mean, 0);
  $('csiLength').textContent = quality.csi_length_mode ?? state.latestPacket?.csi_length ?? '—';
  $('connect').hidden = !!serial.connected; $('disconnect').hidden = !serial.connected;
  $('port').disabled = !!serial.connected; $('refresh').disabled = !!serial.connected;

  const total = Object.values(trainCounts).reduce((sum, value) => sum + value, 0);
  const recommended = 30; const coverage = Math.min(100, Math.round(total / recommended * 100));
  $('trainPercent').textContent = `${coverage}%`;
  $('trainingReadiness').textContent = calibrationDone ? (total >= recommended ? 'Recommended coverage reached' : 'Minimum coverage reached') : 'More calibration needed';
  $('trainingCopy').textContent = total >= recommended ? 'All three room conditions have ten sessions.' : `${total} of ${recommended} recommended sessions collected. Training unlocks at three per condition.`;
  $('trainingCounts').innerHTML = Object.entries(LABELS).map(([key, meta]) => `<div><span>${meta.name}</span><strong>${trainCounts[key] || 0} sessions</strong></div>`).join('');
  $('trainState').textContent = room.job?.kind === 'train' && room.job.status === 'running' ? 'Training' : trained ? 'Model ready' : 'Not trained';
  $('trainState').className = `state-label ${trained ? 'good' : room.job?.status === 'failed' ? 'bad' : room.job?.status === 'running' ? 'cyan' : 'neutral'}`;
  $('trainModel').disabled = !calibrationDone || room.job?.status === 'running';
  $('trainModel').textContent = room.job?.kind === 'train' && room.job.status === 'running' ? 'Training…' : trained ? 'Retrain room model' : 'Train room model';

  const job = room.job;
  $('trainJob').hidden = !job || (job.kind !== 'train' && !job.output);
  if (job) {
    $('trainJob').querySelector('.spinner').hidden = job.status !== 'running';
    $('trainJob').querySelector('strong').textContent = job.status === 'running' ? `${job.kind === 'train' ? 'Training' : 'Validating'} room model` : `Last job ${job.status}`;
    $('trainOutput').textContent = job.output || 'Building features and evaluating candidates…';
  }

  $('validationState').textContent = job?.kind === 'validate' && job.status === 'running' ? 'Validating' : validated ? 'Passed' : 'Not validated';
  $('validationState').className = `state-label ${validated ? 'good' : job?.kind === 'validate' && job.status === 'failed' ? 'bad' : job?.kind === 'validate' && job.status === 'running' ? 'cyan' : 'neutral'}`;
  $('validateModel').disabled = !trained || !countReady(holdoutCounts) || job?.status === 'running';
  $('validateModel').textContent = job?.kind === 'validate' && job.status === 'running' ? 'Validating…' : validated ? 'Run validation again' : 'Run validation';
  const report = $('validationReport');
  report.hidden = !room.validation_report && !(job?.kind === 'validate' && job.status === 'failed');
  if (!report.hidden) {
    const passed = !!room.validation_report;
    report.className = `validation-report${passed ? '' : ' bad'}`;
    report.innerHTML = `<strong>${passed ? 'Validation passed' : 'Validation failed'}</strong><span>${passed ? 'The room model met every deployment gate.' : 'Review the failed condition and collect replacement holdouts.'}</span><pre></pre>`;
    report.querySelector('pre').textContent = room.validation_report || job.output;
  }

  $('activationState').textContent = activated ? 'Active' : validated ? 'Ready' : 'Inactive';
  $('activationState').className = `state-label ${activated ? 'good' : validated ? 'cyan' : 'neutral'}`;
  document.querySelector('.activation-hero').classList.toggle('ready', validated);
  $('activationTitle').textContent = activated ? 'Monitoring is active' : validated ? 'Ready to deploy' : 'Validation required';
  $('activationCopy').textContent = activated ? 'This room model is serving live occupancy decisions.' : validated ? 'The holdout set passed. Activation will switch the live predictor to this model.' : 'Complete calibration, training, and holdout validation first.';
  $('modelIdentity').textContent = room.latest_model ? `${room.latest_model.split('/').pop()} · trained ${shortDate(room.trained_at)}` : 'No room-specific model available.';
  $('activateModel').disabled = !validated || activated;
  $('activateModel').textContent = activated ? 'Model active' : 'Activate model';
  renderConditions();
}

function renderSystem() {
  const serial = state.status?.serial || {}, quality = serial.quality || {}, room = selectedRoom();
  const connected = !!serial.connected;
  $('connectionBadge').innerHTML = `<i></i> ${connected ? 'Connected' : 'Offline'}`;
  $('connectionBadge').className = `badge ${connected ? quality.healthy ? 'good' : 'warn' : 'neutral'}`;
  $('modelBadge').textContent = room?.active ? 'Live model active' : room?.validated ? 'Validated model' : room?.model_ready ? 'Model not validated' : 'No room model';
  $('modelBadge').className = `badge ${room?.active ? 'good' : room?.validated ? 'cyan' : 'neutral'}`;
  $('globalStatus').textContent = connected ? quality.healthy ? 'System healthy' : 'Signal needs attention' : 'System offline';
  $('globalDetail').textContent = connected ? `${fmt(quality.recent_packets_per_second ?? serial.packets_per_second)} packets / sec` : 'No receiver connected';
  $('globalDot').className = `status-dot ${connected ? quality.healthy ? 'good' : 'warn' : 'neutral'}`;
  renderDiagnostics(serial, quality);
}

function renderDiagnostics(serial, quality) {
  $('diagPackets').textContent = serial.packet_count ?? 0; $('diagRejected').textContent = serial.rejected_count ?? 0;
  $('diagRssi').textContent = Number.isFinite(quality.rssi_mean) ? `${fmt(quality.rssi_mean, 0)} dBm` : '—';
  $('diagJitter').textContent = fmt(quality.delivery_jitter, 2); $('diagAgc').textContent = fmt(quality.agc_gain_std, 2);
  $('diagLength').textContent = quality.csi_length_mode ?? '—';
  $('diagProblems').textContent = (quality.problems || []).join(' · ') || (quality.ready ? 'No stream-quality problems detected.' : 'Connect a receiver to inspect the stream.');
}

function markUnavailable(reason) {
  $('occupancyField').className = 'occupancy-field unavailable'; $('presence').textContent = '—';
  $('confidence').textContent = '—'; $('confidenceFill').style.width = '0'; $('presenceReason').textContent = reason;
}

function drawPrediction(value) {
  if (value.source && value.source !== 'live') return;
  const active = activeRoom(); if (!active) return;
  const presence = value.presence === 'HOME' ? 'home' : 'away';
  $('occupancyField').className = `occupancy-field ${presence}`;
  $('presence').textContent = value.presence === 'HOME' ? 'OCCUPIED' : 'EMPTY';
  $('occupancyKicker').textContent = `${active.name} · room state`;
  $('confidence').textContent = `${(value.confidence * 100).toFixed(1)}%`;
  $('confidenceFill').style.width = `${value.confidence * 100}%`;
  $('presenceReason').textContent = value.presence === 'HOME' ? 'Presence detected. Comfort systems may remain active.' : 'No occupant detected. Room can enter energy-saving mode.';
  const decision = {presence, label: value.presence === 'HOME' ? 'Occupied' : 'Empty', confidence: value.confidence, at: new Date()};
  if (state.decisions.at(-1)?.label !== decision.label || Date.now() - state.decisions.at(-1).at > 10000) state.decisions.push(decision);
  state.decisions = state.decisions.slice(-5); renderTimeline();
}

function renderTimeline() {
  const target = $('decisionTimeline');
  if (!state.decisions.length) { target.innerHTML = '<p>No live decisions yet.</p>'; return; }
  target.innerHTML = state.decisions.slice().reverse().map(item => `<div class="decision ${item.presence}"><i></i><span>${item.label} · ${(item.confidence * 100).toFixed(0)}%</span><span>${item.at.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit', second: '2-digit'})}</span></div>`).join('');
}

function renderLive() {
  const room = activeRoom(), serial = state.status?.serial || {}, quality = serial.quality || {};
  $('liveRoomName').textContent = room?.name || 'No active room';
  $('liveModel').textContent = room?.latest_model?.split('/').pop() || 'None active';
  $('liveCalibration').textContent = room?.validated ? `Validated ${shortDate(room.validated_at)}` : 'Not validated';
  $('liveQuality').textContent = !quality.ready ? 'Offline' : quality.healthy ? 'Healthy' : 'Needs attention';
  $('liveRate').textContent = Number.isFinite(quality.recent_packets_per_second) ? `${fmt(quality.recent_packets_per_second)} pkt/s` : '—';
  const stale = serial.connected && serial.seconds_since_last_packet != null && serial.seconds_since_last_packet > 3;
  $('liveSignal').textContent = !serial.connected ? 'Offline' : stale ? 'Stream stalled' : quality.healthy ? 'Live' : 'Signal warning';
  $('liveSignal').className = `state-label ${!serial.connected ? 'neutral' : stale || !quality.healthy ? 'warn' : 'good'}`;
  $('liveFreshness').textContent = !room ? 'Activate a validated room model to begin.' : !serial.connected ? 'Room model is ready. Connect its receiver to begin live monitoring.' : stale ? 'The CSI stream has stopped; the prior decision was cleared.' : 'Occupancy updates approximately twice per second after the initial warm-up.';
  if (!room) markUnavailable('No validated room model is active.');
  else if (!serial.connected) markUnavailable('Receiver offline — no current room decision.');
  else if (stale) markUnavailable('CSI stream stalled — prior decision cleared.');
  else if (!state.status?.prediction?.latest) {
    const p = state.status?.prediction; markUnavailable(`Warming up signal window${p ? ` · ${p.buffer}/${p.window_packets} packets` : ''}`);
  }
}

function renderRecordingStatus() {
  const recording = state.status?.recording || {}; const bar = $('recordingBar');
  bar.hidden = !recording.active;
  if (recording.active) {
    const name = LABELS[recording.label]?.name || recording.label;
    $('recordingTitle').textContent = recording.state === 'countdown' ? `Prepare: ${name}` : `Recording: ${name}`;
    $('recordingDetail').textContent = recording.state === 'countdown' ? `Starts in ${Math.max(1, Math.ceil(recording.countdown_remaining))} seconds — move to the correct position` : `${Math.ceil(recording.recording_remaining)} seconds remaining · ${recording.packet_count} packets`;
  }
  if (state.lastRecordingActive && !recording.active) {
    refreshRooms(); refreshRecordings();
    if (recording.last_result?.stop_reason === 'automatic') toast(`Recording saved · ${recording.last_result.packet_count} packets`);
  }
  state.lastRecordingActive = !!recording.active;
}

function renderRecordings() {
  const target = $('recordingRows'), room = selectedRoom();
  let rows = state.recordings.filter(row => row.room_id === room?.id);
  if (state.filter !== 'all') rows = rows.filter(row => row.split === state.filter);
  target.replaceChildren();
  if (!rows.length) { const empty = document.createElement('div'); empty.className = 'recording-empty'; empty.textContent = room ? 'No recordings in this dataset yet.' : 'Select a room to see its recordings.'; target.append(empty); return; }
  rows.forEach(row => {
    const line = document.createElement('div'); line.className = 'recording-row';
    const condition = document.createElement('span'); condition.textContent = LABELS[row.label]?.name || row.label || 'Unknown';
    const split = document.createElement('span'); split.textContent = row.split === 'holdout' ? 'Holdout' : 'Training';
    const captured = document.createElement('span'); captured.textContent = shortDate(row.started_at);
    const quality = document.createElement('span'); const cached = state.quality.get(row.filename); quality.className = `quality-value ${cached?.usable === true ? 'good' : cached?.usable === false ? 'bad' : ''}`; quality.textContent = cached ? cached.usable ? 'Usable' : 'Rejected' : 'Not checked';
    const button = document.createElement('button'); button.className = 'button ghost'; button.textContent = cached ? 'Recheck' : 'Check'; button.onclick = () => checkQuality(row.filename, quality, button);
    line.append(condition, split, captured, quality, button); target.append(line);
  });
}

async function checkQuality(filename, label, button) {
  try { button.disabled = true; button.textContent = 'Checking…'; label.textContent = 'Analyzing';
    const result = await api(`/api/recordings/${encodeURIComponent(filename)}/quality`); state.quality.set(filename, result); renderRecordings();
    toast(result.usable ? 'Recording passed every capture-quality gate.' : `Recording rejected: ${result.problems.join(' · ')}`, !result.usable);
  } catch (error) { toast(error.message, true); renderRecordings(); }
}

function renderAll() { renderRooms(); renderSystem(); renderWorkflow(); renderLive(); renderRecordingStatus(); renderRecordings(); }

async function refreshStatus() {
  try { state.status = await api('/api/status'); renderSystem(); renderWorkflow(); renderLive(); renderRecordingStatus(); }
  catch (error) { toast(`Backend unavailable: ${error.message}`, true); }
}
async function refreshRooms() {
  try {
    state.rooms = await api('/api/rooms');
    if (!state.roomId || !state.rooms.some(room => room.id === state.roomId)) state.roomId = state.rooms.find(room => room.active)?.id || state.rooms[0]?.id || null;
    if (state.roomId) localStorage.setItem('roomsense.room', state.roomId);
    renderAll();
  } catch (error) { toast(error.message, true); }
}
async function refreshRecordings() {
  try { state.recordings = await api('/api/recordings'); renderRecordings(); } catch (error) { toast(error.message, true); }
}
async function refreshPorts() {
  try {
    const old = $('port').value, ports = await api('/api/ports'); $('port').replaceChildren();
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = ports.length ? 'Select receiver' : 'No physical receiver found'; $('port').append(placeholder);
    ports.forEach(item => { const option = document.createElement('option'); option.value = item.device; option.textContent = `${item.device} · ${item.description}`; $('port').append(option); });
    if ([...$('port').options].some(option => option.value === old)) $('port').value = old;
  } catch (error) { toast(error.message, true); }
}

async function startRecording(label, split) {
  const room = selectedRoom(); if (!room) return;
  try {
    await api('/api/recordings/start', {method: 'POST', body: JSON.stringify({label, room_id: room.id, split, notes: `${room.name} · ${split}`, delay_seconds: 10, duration_seconds: 30})});
    switchStep(split === 'training' ? 'calibrate' : 'validate'); await refreshStatus();
  } catch (error) { toast(error.message, true); }
}

async function runRoomJob(kind) {
  const room = selectedRoom(); if (!room) return;
  try { await api(`/api/rooms/${room.id}/${kind}`, {method: 'POST'}); toast(kind === 'train' ? 'Training started. The current live model will not change.' : 'Holdout validation started.'); await refreshRooms(); }
  catch (error) { toast(error.message, true); }
}

function openRoomDialog(edit = false) {
  const room = selectedRoom(); $('roomForm').dataset.edit = edit && room ? room.id : '';
  $('roomDialogTitle').textContent = edit ? 'Edit room' : 'Add a room'; $('roomName').value = edit ? room?.name || '' : ''; $('roomPlacement').value = edit ? room?.placement || '' : '';
  $('roomDialog').showModal(); setTimeout(() => $('roomName').focus(), 50);
}

async function saveRoom(event) {
  event.preventDefault();
  if (!$('roomForm').reportValidity()) return;
  const roomId = $('roomForm').dataset.edit, body = JSON.stringify({name: $('roomName').value.trim(), placement: $('roomPlacement').value.trim()});
  try {
    const room = await api(roomId ? `/api/rooms/${roomId}` : '/api/rooms', {method: roomId ? 'PATCH' : 'POST', body});
    state.roomId = room.id; localStorage.setItem('roomsense.room', room.id); $('roomDialog').close(); await refreshRooms(); toast(roomId ? 'Room details updated.' : `${room.name} created. Start by connecting its receiver.`);
  } catch (error) { toast(error.message, true); }
}

function connectEvents() {
  document.querySelectorAll('.nav-item').forEach(button => button.onclick = () => switchView(button.dataset.view));
  document.querySelectorAll('.step').forEach(button => button.onclick = () => switchStep(button.dataset.step));
  document.querySelectorAll('.next-step').forEach(button => button.onclick = () => switchStep(button.dataset.next));
  document.querySelectorAll('.filter').forEach(button => button.onclick = () => { state.filter = button.dataset.filter; document.querySelectorAll('.filter').forEach(item => item.classList.toggle('active', item === button)); renderRecordings(); });
  $('mobileMenu').onclick = () => document.body.classList.toggle('menu-open');
  $('roomPicker').onchange = event => selectRoom(event.target.value);
  $('addRoomSmall').onclick = () => openRoomDialog(false); $('createFirstRoom').onclick = () => openRoomDialog(false); $('editRoom').onclick = () => openRoomDialog(true);
  $('roomForm').onsubmit = event => {
    if (event.submitter?.value === 'cancel') { event.preventDefault(); $('roomDialog').close(); return; }
    saveRoom(event);
  };
  $('refresh').onclick = refreshPorts;
  $('connect').onclick = async () => { try { if (!$('port').value) throw new Error('Select a physical receiver first.'); await api('/api/connect', {method: 'POST', body: JSON.stringify({port: $('port').value})}); toast('Receiver connected. Waiting for a stable CSI window.'); await refreshStatus(); } catch (error) { toast(error.message, true); } };
  $('disconnect').onclick = async () => { try { await api('/api/disconnect', {method: 'POST'}); await refreshStatus(); } catch (error) { toast(error.message, true); } };
  $('stopRecording').onclick = async () => { try { await api('/api/recordings/stop', {method: 'POST'}); await refreshStatus(); } catch (error) { toast(error.message, true); } };
  $('trainModel').onclick = () => runRoomJob('train'); $('validateModel').onclick = () => runRoomJob('validate');
  $('activateModel').onclick = async () => { const room = selectedRoom(); try { await api(`/api/rooms/${room.id}/activate`, {method: 'POST'}); toast(`${room.name} is now serving live decisions.`); await refreshStatus(); await refreshRooms(); switchView('live'); } catch (error) { toast(error.message, true); } };
  $('openDiagnostics').onclick = () => $('diagnosticsDialog').showModal(); $('refreshRecordings').onclick = refreshRecordings;
}

function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws'; const socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
  socket.onmessage = event => { const message = JSON.parse(event.data); if (message.type === 'packet') state.latestPacket = message.data; else if (message.type === 'prediction') drawPrediction(message.data); };
  socket.onclose = () => setTimeout(connectWebSocket, 1200);
}

async function boot() {
  connectEvents(); await Promise.all([refreshPorts(), refreshRecordings(), refreshStatus(), refreshRooms()]);
  connectWebSocket(); setInterval(refreshStatus, 1000); setInterval(refreshRooms, 3000);
}
boot();
