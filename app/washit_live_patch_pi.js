// ==========================================================================
// WASH-IT — live backend client (PHASE-AWARE)  v2
// --------------------------------------------------------------------------
// HOW TO APPLY:
//   In your washit_v7 .html, find the existing live-backend block at the very
//   bottom of the last <script> — it starts with:
//       let liveSocket = null;
//   ...and ends with the resetDemoWithLiveSupport() function (just before
//   </script>). DELETE that whole block and PASTE everything below in its
//   place. (Keep it inside the same <script> tag.)
//
// It talks to server.py's schema: every message has `phase` + `frame`.
// Phases: PRIME, SOAP, GETSET, RUN, REMEDIATE, RINSE, DONE.
// Reuses your existing functions: STEPS, L(), SPEECH, showSubtitle,
// updateConf, setStep, syncList, doFlash, buildList, showScreen, startTimer,
// and your elements (fbb, cam-feed, meter-arc, meter-pct, sc-done, slist…).
// ==========================================================================

let liveSocket = null;
let liveConnected = false;
let __lastPhase = null;
let __doneShown = false;
let __awaitingPrime = false;
let __failed = new Set();
let __prevTheme = null;

// ── inject the full-screen phase overlay (soap / get-set / redo / rinse) ──
(function injectPhaseOverlay() {
  if (document.getElementById('phase-ov')) return;
  const css = document.createElement('style');
  css.textContent = `
    .phase-ov{position:fixed;inset:0;z-index:9999;display:none;
      flex-direction:column;align-items:center;justify-content:center;gap:14px;
      background:rgba(0,0,0,.62);backdrop-filter:blur(6px);
      font-family:var(--font-head,sans-serif);text-align:center;padding:24px}
    .phase-ov.show{display:flex}
    .phase-ov-big{font-size:clamp(60px,16vw,150px);font-weight:900;line-height:1;
      color:var(--accent,#00b4ff);filter:var(--glow,none)}
    .phase-ov-title{font-size:clamp(22px,4vw,40px);font-weight:800;color:var(--text,#fff)}
    .phase-ov-sub{font-size:clamp(14px,2.4vw,20px);color:var(--muted2,#9ab);max-width:560px}
    .phase-ov-tag{font-size:12px;letter-spacing:3px;text-transform:uppercase;
      color:var(--muted,#789)}
    .phase-ov.redo{background:#ffffff;backdrop-filter:none}
    .phase-ov.redo .phase-ov-tag{color:#e0405a;font-weight:800}
    .phase-ov.redo .phase-ov-big{color:#e0405a;filter:none}
    .phase-ov.redo .phase-ov-title{color:#16202b}
    .phase-ov.redo .phase-ov-sub{color:#46505a}
  `;
  document.head.appendChild(css);
  const ov = document.createElement('div');
  ov.id = 'phase-ov';
  ov.className = 'phase-ov';
  ov.innerHTML =
    '<div class="phase-ov-tag" id="phase-ov-tag"></div>' +
    '<div class="phase-ov-big" id="phase-ov-big"></div>' +
    '<div class="phase-ov-title" id="phase-ov-title"></div>' +
    '<div class="phase-ov-sub" id="phase-ov-sub"></div>';
  (document.getElementById('app') || document.body).appendChild(ov);
})();

// ── extra styles (red "missed" circle) + fit-to-viewport auto-scaler ──────
(function injectExtras() {
  const css = document.createElement('style');
  css.textContent = `
    html,body{overflow:hidden;height:100%}
    body{display:block}
    /* center the whole app against the VIEWPORT (not a growable body) and scale */
    #app{position:fixed;top:50%;left:50%;margin:0!important}
    #s-idle,#s-session,#s-score,#s-history,#s-leader{min-height:0}
    .si.fail .si-chk{background:#e0405a;border-color:#e0405a;color:#fff}
    .si.fail .si-name{color:#e0405a;font-weight:600}
    .si.fail{opacity:1}
  `;
  document.head.appendChild(css);

  let lastS = -1;
  function fitApp() {
    const app = document.getElementById('app');
    if (!app) return;
    const W = app.offsetWidth, H = app.offsetHeight;   // layout box (transform-independent)
    if (!W || !H) return;
    const S = Math.min((window.innerWidth - 12) / W, (window.innerHeight - 12) / H, 1);
    if (Math.abs(S - lastS) > 0.002) {
      lastS = S;
      app.style.transformOrigin = 'center center';
      app.style.transform = 'translate(-50%,-50%) scale(' + S.toFixed(3) + ')';
    }
  }
  window.__fitApp = fitApp;
  window.addEventListener('resize', function () { lastS = -1; fitApp(); });
  window.addEventListener('load', fitApp);
  setTimeout(fitApp, 120);
  setInterval(fitApp, 300);    // re-fit as the active screen changes height
})();

function showPhaseOverlay(tag, big, title, sub) {
  const ov = document.getElementById('phase-ov');
  ov.classList.add('show');
  document.getElementById('phase-ov-tag').textContent = tag || '';
  document.getElementById('phase-ov-big').textContent = (big === null || big === undefined) ? '' : big;
  document.getElementById('phase-ov-title').textContent = title || '';
  document.getElementById('phase-ov-sub').textContent = sub || '';
}
function hidePhaseOverlay() {
  const ov = document.getElementById('phase-ov');
  ov.classList.remove('show', 'redo');
}

// distinct white/red full-screen alert listing the missed steps before redo
function showRedoIntro(missedArr, countdown) {
  const ov = document.getElementById('phase-ov');
  ov.classList.add('show', 'redo');
  const names = (missedArr || []).map(function (k) { return (k + 1); }).join(', ');
  document.getElementById('phase-ov-tag').textContent = 'Redo Required';
  document.getElementById('phase-ov-big').textContent = '\u26A0';
  document.getElementById('phase-ov-title').textContent = 'Missed steps: ' + names;
  document.getElementById('phase-ov-sub').textContent = 'Get ready to redo them — ' + countdown + 's';
}

// the front-page "Start Handwashing Session" button now connects to the camera too
startDemo = function () {
  connectLiveBackend(document.getElementById('live-server-addr').value);
};

// ── visibility / activity guidance in the feedback banner ─────────────────
function showVisibilityGuidance(text) {
  const fb = document.getElementById('fbb');
  fb.className = 'fb-banner warn';
  document.getElementById('fbico').textContent = '\uD83D\uDCF7';
  document.getElementById('fbtxt').textContent = text || 'Move your hands into view';
}

// ── narrate once when a phase begins ──────────────────────────────────────
function onPhaseEnter(phase, status) {
  // theme switches to WHITE for the whole redo as a strong visual alert
  const app = document.getElementById('app');
  if (phase === 'REMEDIATE') {
    if (__prevTheme === null) __prevTheme = app.getAttribute('data-theme');
    app.setAttribute('data-theme', 'white');
  } else if (__prevTheme !== null) {
    app.setAttribute('data-theme', __prevTheme);
    __prevTheme = null;
  }

  let line = '';
  if (phase === 'PRIME') {
    line = 'Show both hands to the camera to begin.';
  } else if (phase === 'SOAP') {
    // hands detected + buffer full -> the process is starting; start the clock NOW
    startTimer();
    line = 'Hands detected. Wet your hands and apply soap.';
  } else if (phase === 'GETSET') {
    line = 'Get ready. Begin step one.';
  } else if (phase === 'REMEDIATE') {
    line = 'You missed some steps. Get ready to redo them.';
  } else if (phase === 'RINSE') {
    line = 'Now rinse your hands under running water.';
  }
  if (line) { try { SPEECH.say(line, false); } catch (e) {} showSubtitle(line); }
}

// ── per-phase renderers ───────────────────────────────────────────────────
function renderPrime(s) {
  const pct = Math.round((s.buffer_fill || 0) * 100);
  if (s.visibility_ok === false) {
    showPhaseOverlay('Waiting', null, 'Show BOTH hands to the camera',
                     'The session begins once your hands are detected');
  } else {
    showPhaseOverlay('Get ready', pct + '%', 'Hands detected — hold steady',
                     'Preparing the model… ' + pct + '%');
  }
}

function renderGate(s, title) {                       // SOAP and RINSE
  const secs = Math.max(0, Math.ceil(s.phase_remaining || 0));
  showPhaseOverlay(s.phase, secs + 's', title, s.guidance || '');
}

function renderGetset(s) {
  showPhaseOverlay('Get set', s.countdown, 'Begin Step 1 — ' + STEPS[0].name, '');
}

function renderRedo(s) {
  if (s.redo_phase === 'intro') {
    showRedoIntro(s.redo_missed, s.redo_intro_left);   // blank white alert listing missed steps
    applyCircles(s);
    return;
  }
  hidePhaseOverlay();                 // redo runs on the live session screen, not a blank overlay
  const k = s.redo_step;
  const name = STEPS[k] ? STEPS[k].name : ('Step ' + (k + 1));
  if (typeof k === 'number' && k !== cur) { cur = k; setStep(k); }
  applyCircles(s);                    // green = done, red = still-missed, highlight = current redo step

  const fb = document.getElementById('fbb');
  if (s.redo_phase === 'getready') {
    fb.className = 'fb-banner warn';
    document.getElementById('fbico').textContent = '\u23F3';
    document.getElementById('fbtxt').innerHTML =
      'Redo <b>Step ' + (k + 1) + ' — ' + name + '</b> starting in ' + s.redo_countdown + 's';
  } else {
    const left = (s.redo_remaining != null) ? Math.ceil(s.redo_remaining) + 's left' : '';
    if (s.detecting) {
      fb.className = 'fb-banner ok';
      document.getElementById('fbico').textContent = '\u2705';
      document.getElementById('fbtxt').innerHTML =
        'Detecting <b>Step ' + (k + 1) + '</b> — hold it &middot; ' + left;
    } else {
      fb.className = 'fb-banner warn';
      document.getElementById('fbico').textContent = '\u21BB';
      document.getElementById('fbtxt').innerHTML =
        'Redo <b>Step ' + (k + 1) + ' — ' + name + '</b> &middot; ' + left;
    }
    const predName = STEPS[s.predicted_step] ? STEPS[s.predicted_step].name : '—';
    if (s.confidence != null) updateConf(Math.round((s.confidence || 0) * 100), predName);
  }
}

function renderRun(s) {
  hidePhaseOverlay();

  // The backend clock is the source of truth for pacing. Advance the current
  // step and sync the checklist circles EVERY tick, BEFORE any early-return —
  // a momentary loss of hand visibility must not freeze the step timeline.
  if (typeof s.current_step === 'number' && s.current_step !== cur) {
    cur = s.current_step;
    setStep(cur);
  }
  applyCircles(s);

  if (s.buffering) {
    updateConf(Math.round((s.buffer_fill || 0) * 100), 'Buffering…');
    return;
  }
  if (s.visibility_ok === false) {
    showVisibilityGuidance(s.guidance);
    return;
  }

  const predName = STEPS[s.predicted_step] ? STEPS[s.predicted_step].name : '—';
  updateConf(Math.round((s.confidence || 0) * 100), predName);

  // feedback banner (with the per-step slot countdown so the 8s rule is visible)
  const slotTxt = (s.slot_left != null) ? ' &middot; ' + Math.ceil(s.slot_left) + 's left' : '';
  const fb = document.getElementById('fbb');
  if (s.is_compliant) {
    fb.className = 'fb-banner ok';
    document.getElementById('fbico').textContent = '\u2705';
    document.getElementById('fbtxt').innerHTML =
      'Performing <b>Step ' + (s.current_step + 1) + '</b> — ' +
      (s.cur_det_s != null ? s.cur_det_s.toFixed(1) + 's held' : 'keep going') + slotTxt;
  } else if (s.struggling) {
    fb.className = 'fb-banner warn';
    document.getElementById('fbico').textContent = '\u26A0\uFE0F';
    document.getElementById('fbtxt').innerHTML =
      (STEPS[s.current_step] ? STEPS[s.current_step].corrective : '') + slotTxt;
  } else {
    fb.className = 'fb-banner';
    document.getElementById('fbico').textContent = '\uD83D\uDC40';
    document.getElementById('fbtxt').innerHTML =
      'Watching for <b>Step ' + (s.current_step + 1) + ' — ' +
      (STEPS[s.current_step] ? STEPS[s.current_step].name : '') + '</b>' + slotTxt;
  }

  // meter = revealed steps + within-slot progress
  const meterPct = Math.round(
    (((s.completed_steps ? s.completed_steps.length : 0) + (s.progress || 0)) / 7) * 100);
  document.getElementById('meter-pct').textContent = meterPct + '%';
  document.getElementById('meter-arc').style.strokeDashoffset = 289 * (1 - meterPct / 100);
}

// paint the right-side circles: green = done, red = missed, highlight = current
function _paintCircles() {
  document.querySelectorAll('.si').forEach(function (el, i) {
    let c = 'si';
    if (done.has(i)) c = 'si done';
    else if (i === cur) c = 'si cur';
    else if (__failed.has(i)) c = 'si fail';
    el.className = c;
  });
}
function applyCircles(s) {
  const comp = s.completed_steps || [];
  __failed = new Set(s.failed_steps || []);
  document.getElementById('sc-done').textContent = comp.length;
  if (comp.length > done.size) {
    const justDone = comp[comp.length - 1];
    done = new Set(comp);
    doFlash('green');
    try { SPEECH.say(L().stepDone(L().stepNames[justDone], L().steps[justDone]), true); } catch (e) {}
    showSubtitle(L().stepDone(L().stepNames[justDone], L().steps[justDone]));
  } else {
    done = new Set(comp);
  }
  _paintCircles();
}

// ── final score screen, driven by the backend verdict ─────────────────────
function renderDone(s) {
  if (__doneShown) return;          // server keeps sending DONE; only act once
  __doneShown = true;
  hidePhaseOverlay();
  if (timerInt) clearInterval(timerInt);

  document.getElementById('st-steps').textContent = (s.steps_done != null ? s.steps_done : 0) + '/7';
  document.getElementById('st-time').textContent = (s.duration_s != null ? s.duration_s : 0) + 's';
  document.getElementById('st-order').textContent = s.order || '—';

  const grade = s.grade || '—';
  document.getElementById('scgrade').textContent = grade;
  const gmsg = (L().gradeMsg && L().gradeMsg[grade]) ? L().gradeMsg[grade] : '';
  document.getElementById('scmsg').textContent = gmsg;

  // animate the score ring + number
  const score = s.score != null ? s.score : 0;
  document.getElementById('sarc').style.strokeDashoffset = 515 * (1 - score / 100);
  let shown = 0;
  const tick = setInterval(function () {
    shown += Math.max(1, Math.round(score / 30));
    if (shown >= score) { shown = score; clearInterval(tick); }
    document.getElementById('scnum').textContent = shown;
  }, 30);

  // per-step + soap/rinse badges
  const badges = document.getElementById('scbadges');
  if (badges) {
    let html = '';
    html += _badge('Soap', s.soap_done);
    (s.results || []).forEach(function (r, i) {
      const label = (STEPS[i] ? STEPS[i].name : 'Step ' + (i + 1)) + (r.retry ? ' (redo)' : '');
      html += _badge(label, r.passed);
    });
    html += _badge('Rinse', s.rinse_done);
    badges.innerHTML = html;
  }

  try { SPEECH.say(L().scoreMsg(score, grade, gmsg), true); } catch (e) {}
  showSubtitle('Session complete — score ' + score + ', grade ' + grade);

  // save to history (same record shape the demo's showScore uses)
  try {
    var entry = { id: Date.now(), datetime: new Date().toLocaleString(),
      steps: (s.steps_done != null ? s.steps_done : 0),
      duration: (s.duration_s != null ? s.duration_s : 0),
      score: score, grade: grade, order: s.order || '—' };
    history.unshift(entry);
    if (history.length > 50) history = history.slice(0, 50);
    localStorage.setItem('washit_history', JSON.stringify(history));
  } catch (e) {}

  // kiosk auto-return countdown
  var cd = 12;
  if (cdInt) clearInterval(cdInt);
  cdInt = setInterval(function () {
    cd--;
    var el = document.getElementById('sccd');
    if (el) el.textContent = 'Returning to home in ' + cd + 's...';
    if (cd <= 0) { clearInterval(cdInt); resetDemoWithLiveSupport(); }
  }, 1000);

  showScreen('score');
}
function _badge(label, ok) {
  const cls = ok ? 'ok' : 'miss';
  const ico = ok ? '\u2713' : '\u2717';
  return '<div class="badge ' + cls + '">' + ico + ' ' + label + '</div>';
}

// ── main message handler ──────────────────────────────────────────────────
// tracks the last object URL so we can revoke it and avoid leaking memory
// as new binary frames keep arriving
let __camFeedUrl = null;
function renderCamFrame(blob) {
  const url = URL.createObjectURL(blob);
  document.getElementById('cam-feed').src = url;
  if (__camFeedUrl) URL.revokeObjectURL(__camFeedUrl);
  __camFeedUrl = url;
}

function handleLiveStatus(status) {
  const phase = status.phase || 'RUN';

  // After connect we send a reset; ignore any leftover non-PRIME messages from a
  // previous session until the fresh PRIME (server actually restarted) arrives.
  if (__awaitingPrime) {
    if (phase === 'PRIME') { __awaitingPrime = false; }
    else { return; }
  }

  if (phase !== __lastPhase) { onPhaseEnter(phase, status); __lastPhase = phase; }

  switch (phase) {
    case 'PRIME':     return renderPrime(status);
    case 'SOAP':      return renderGate(status, 'Wet hands & apply soap');
    case 'GETSET':    return renderGetset(status);
    case 'RUN':       return renderRun(status);
    case 'REMEDIATE': return renderRedo(status);
    case 'RINSE':     return renderGate(status, 'Rinse hands under water');
    case 'DONE':      return renderDone(status);
    default:          return renderRun(status);
  }
}

// ── connect / reset ───────────────────────────────────────────────────────
function connectLiveBackend(wsUrl) {
  if (liveSocket) liveSocket.close();
  cur = 0; done = new Set(); __failed = new Set(); __lastPhase = null; __doneShown = false;
  if (__prevTheme !== null) {            // restore theme if a prior redo was interrupted
    document.getElementById('app').setAttribute('data-theme', __prevTheme);
    __prevTheme = null;
  }
  if (typeof countdownAnnounced !== 'undefined') countdownAnnounced = {};
  buildList(); setStep(0); updateMeter();
  showScreen('session');
  const camCard = document.getElementById('cam-card');
  camCard.style.display = 'flex';
  camCard.closest('.top-row').classList.remove('cam-hidden');
  const manual = document.querySelector('.manual-row');
  if (manual) manual.style.display = 'none';        // hide demo buttons in live mode
  document.getElementById('timer').textContent = '00:00';
  // NOTE: deliberately NO startTimer() and NO welcome here. The UI sits in the
  // PRIME overlay (see renderPrime) and does not "start" until the server
  // detects both hands + finishes buffering and leaves PRIME (see onPhaseEnter).
  __awaitingPrime = true;
  __lastPhase = null;
  showPhaseOverlay('Connecting', null, 'Show BOTH hands to the camera',
                   'Waiting for the camera server…');

  liveSocket = new WebSocket(wsUrl);
  liveSocket.onopen = function () {
    liveConnected = true;
    console.log('[WASH-IT] connected:', wsUrl);
    liveSocket.send('reset');     // force a clean server-side session
  };
  liveSocket.onmessage = function (e) {
    // binary WS frames are JPEG image bytes (arrive as a Blob by default);
    // text WS frames are the JSON status payload -- pi_server.py sends both
    // per tick instead of embedding base64 image data inside the JSON.
    if (e.data instanceof Blob) {
      renderCamFrame(e.data);
      return;
    }
    handleLiveStatus(JSON.parse(e.data));
  };
  liveSocket.onclose = function () { liveConnected = false; console.log('[WASH-IT] closed'); };
  liveSocket.onerror = function () {
    const fb = document.getElementById('fbb');
    fb.className = 'fb-banner warn';
    document.getElementById('fbtxt').textContent =
      'Cannot reach the camera server. Is server.py running?';
  };
}

function resetLiveBackend() {
  if (liveSocket && liveSocket.readyState === WebSocket.OPEN) liveSocket.send('reset');
  cur = 0; done = new Set(); __failed = new Set(); __lastPhase = null; __doneShown = false;
}

const _origReset = (typeof resetDemo === 'function') ? resetDemo : null;
function resetDemoWithLiveSupport() {
  if (liveConnected) { resetLiveBackend(); if (liveSocket) liveSocket.close(); }
  hidePhaseOverlay();
  const camCard = document.getElementById('cam-card');
  camCard.style.display = 'none';
  camCard.closest('.top-row').classList.add('cam-hidden');
  const manual = document.querySelector('.manual-row');
  if (manual) manual.style.display = '';
  if (_origReset) _origReset();
}
