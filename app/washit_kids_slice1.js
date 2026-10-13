// ==========================================================================
// WASH-IT KIDS — SLICE 1 (v2): welcome -> hands gate -> soap gate
// --------------------------------------------------------------------------
// Strict sequence:
//   1. despite_welcome plays immediately on Start.
//   2. HANDS GATE: when welcome ends, if the server still sees no hands,
//      show a "show your hands" scene + play prompt_hands; after it ends,
//      wait 2s, replay; repeat until hands appear. Clear instantly on hands.
//   3. despite_soap plays only once hands are shown (server left PRIME).
//   4. SOAP GATE:
//        - if the child soaps WHILE Despite is still talking -> no grace/nudge.
//        - else when Despite finishes, swap the MAIN PANEL from Despite to the
//          child's LIVE CAMERA and start a 3s grace; if still not soaping after
//          3s -> prompt_soap. Live feed stays in the panel through soaping.
//   5. Server leaves SOAP -> "Soap done!" (slice 1 ends).
// ==========================================================================

let kidsSocket = null;
let __kAwaitingPrime = false;
let __kPhase = null;

let __welcomePlayed = false;
let __welcomeDone   = false;
let __soapStarted   = false;
let __soapClipDone  = false;
let __soapGraceAt   = 0;
let __camInPanel    = false;
let __cornerCam     = false;
let __soapWashFrames = 0;

// germ loop state
let __kBeaten = 0;          // germs defeated so far (from server germs_beaten)
let __detectArmed = false;  // start_detect sent for the current germ
let __teachStarted = -1;    // which step's teach clip we've begun
let __teachHold = false;    // hold next teach while a defeat cheer plays

const SOAP_GRACE_MS = 3000;
const SOAP_CONFIRM_FRAMES = 8;    // ~0.5s of washing = "soaping right now"

// Resolve prompt audio the SAME way the UI resolves its working clips, so the
// path can't miss. If PRELUDE clips are folder paths, reuse that folder;
// if they're data: URIs (embedded), we still try the folder fallback.
function mediaPath(name) {
  try {
    const ref = (typeof PRELUDE !== 'undefined' && PRELUDE[0] && PRELUDE[0].video) ? PRELUDE[0].video : '';
    if (ref && ref.indexOf('data:') !== 0 && ref.indexOf('/') !== -1) {
      const folder = ref.substring(0, ref.lastIndexOf('/') + 1);   // e.g. "kids_media/"
      return folder + name;
    }
  } catch (e) {}
  return 'kids_media/' + name;
}
const PROMPT_HANDS = () => mediaPath('prompt_hands.mp3');
const PROMPT_SOAP  = () => mediaPath('prompt_soap.mp3');

let __dbgLines = [];
function kdbg(msg) {
  let el = document.getElementById('kids-dbg');
  if (!el) {
    el = document.createElement('div');
    el.id = 'kids-dbg';
    el.style.cssText = 'position:fixed;left:10px;top:10px;z-index:9999;font:600 12px Nunito,monospace;' +
      'background:rgba(0,0,0,.78);color:#0f0;padding:8px 10px;border-radius:8px;max-width:70vw;white-space:pre-line';
    (document.getElementById('app') || document.body).appendChild(el);
  }
  __dbgLines.push(msg);
  if (__dbgLines.length > 7) __dbgLines.shift();
  el.textContent = __dbgLines.join('\n');
  console.log('[KIDS]', msg);
}

// live camera element inside the MAIN panel (Despite's spot)
function panelCamEl() {
  let el = document.getElementById('kids-panel-cam');
  if (el) return el;
  const panelVid = (typeof panelVideoEl === 'function') ? panelVideoEl() : null;
  const host = panelVid ? panelVid.parentNode
             : (document.querySelector('.watch-panel, .stage, .game-panel') || document.body);
  el = document.createElement('img');
  el.id = 'kids-panel-cam';
  el.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:none;border-radius:inherit;z-index:5';
  if (host && getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.appendChild(el);
  return el;
}
function showCamInPanel(on) {
  const cam = panelCamEl();
  if (on) {
    try { if (typeof stopPanelVideo === 'function') stopPanelVideo(); } catch (e) {}
    const pv = (typeof panelVideoEl === 'function') ? panelVideoEl() : null;
    if (pv) pv.style.display = 'none';
    const rv = (typeof realVideoEl === 'function') ? realVideoEl() : document.getElementById('real-step-video');
    if (rv) rv.style.display = 'none';
    const gif = document.getElementById('step-gif'); if (gif) gif.style.display = 'none';
    cam.style.display = 'block';
    __camInPanel = true;
  } else {
    cam.style.display = 'none';
    __camInPanel = false;
    const wc = document.getElementById('watch-caption'); if (wc) wc.textContent = '';   // clear stale caption
  }
}
function setCamCaption(txt) {
  const wc = document.getElementById('watch-caption'); if (wc) wc.textContent = txt || '';
}
function setCamFrame(b64) {
  const src = 'data:image/jpeg;base64,' + b64;
  const pc = document.getElementById('kids-panel-cam');
  if (pc && __camInPanel) pc.src = src;                 // full-panel feed (soap grace)
  const cc = document.getElementById('kids-corner-cam');
  if (cc && __cornerCam) cc.src = src;                  // corner feed (germ detect)
}

// live-camera corner inset shown ALONGSIDE the real-hands demo during detect
function ensureCornerCam() {
  let el = document.getElementById('kids-corner-cam');
  if (el) return el;
  const wrap = document.createElement('div');
  wrap.id = 'kids-corner-cam-wrap';
  wrap.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:70;width:210px;height:158px;' +
    'border-radius:14px;overflow:hidden;border:3px solid #00D4FF;box-shadow:0 6px 20px rgba(0,0,0,.4);background:#000';
  el = document.createElement('img');
  el.id = 'kids-corner-cam';
  el.style.cssText = 'width:100%;height:100%;object-fit:cover';
  wrap.appendChild(el);
  const tag = document.createElement('div');
  tag.textContent = 'YOUR HANDS';
  tag.style.cssText = 'position:absolute;left:0;right:0;bottom:0;font:700 10px Nunito,sans-serif;' +
    'text-align:center;color:#fff;background:rgba(0,0,0,.55);padding:3px;letter-spacing:1px';
  wrap.appendChild(tag);
  (document.getElementById('app') || document.body).appendChild(wrap);
  return el;
}
// real correct hands stay in the main panel; child's live feed in the corner
function showRealPlusCam(on) {
  const wrap = document.getElementById('kids-corner-cam-wrap');
  if (on) {
    ensureCornerCam();
    document.getElementById('kids-corner-cam-wrap').style.display = 'block';
    __cornerCam = true;
    showCamInPanel(false);   // main panel shows the real hands, not the full cam
  } else {
    if (wrap) wrap.style.display = 'none';
    __cornerCam = false;
  }
}

// "show your hands" scene (like the compliance overlay)
function showHandsScene(on) {
  let ov = document.getElementById('kids-hands-ov');
  if (!ov) {
    ov = document.createElement('div');
    ov.id = 'kids-hands-ov';
    ov.style.cssText = 'position:fixed;inset:0;z-index:9000;display:none;flex-direction:column;' +
      'align-items:center;justify-content:center;gap:18px;background:rgba(6,20,40,.92);' +
      'font-family:Nunito,sans-serif;text-align:center;color:#fff';
    ov.innerHTML =
      '<div style="font-size:clamp(50px,14vw,120px)">\uD83D\uDC50</div>' +
      '<div style="font-size:clamp(22px,5vw,44px);font-weight:900;color:#00D4FF">Show me your hands!</div>' +
      '<div style="font-size:clamp(14px,3vw,20px);opacity:.85">Hold both hands up in front of the camera</div>' +
      '<img id="kids-hands-cam" style="margin-top:10px;width:260px;height:195px;object-fit:cover;' +
      'border-radius:16px;border:3px solid #00D4FF;background:#000">';
    (document.getElementById('app') || document.body).appendChild(ov);
  }
  ov.style.display = on ? 'flex' : 'none';
}
function setHandsSceneFrame(b64) {
  const el = document.getElementById('kids-hands-cam');
  if (el) el.src = 'data:image/jpeg;base64,' + b64;
}

// nudge engine (A+B): full clip, cut off on compliance, replay after a 2s gap
// only if it finished while still stuck.
const Nudge = {
  audio: null, key: null, endedAt: 0, playing: false, gapMs: 2000,
  set(active, key, src) {
    if (!active) { this.stop(); return; }
    if (this.key !== key) { this.stop(); this.key = key; this.endedAt = 0; }
    if (this.playing) return;
    if (this.endedAt === 0 || (Date.now() - this.endedAt >= this.gapMs)) this.play(src);
  },
  play(src) {
    this.stop();
    try {
      this.audio = new Audio(src);
      this.playing = true;
      const self = this;
      this.audio.onended = function () { self.playing = false; self.endedAt = Date.now(); };
      this.audio.onerror = function () {
        self.playing = false; self.endedAt = Date.now();
        kdbg('\u26A0\uFE0F prompt audio NOT FOUND: ' + src + '  (put it in kids_media/)');
      };
      kdbg('\uD83D\uDD0A playing ' + src);
      this.audio.play().catch(function (e) {
        self.playing = false; self.endedAt = Date.now();
        kdbg('\u26A0\uFE0F audio blocked/failed: ' + (e && e.message ? e.message : src));
      });
    } catch (e) { this.audio = null; this.playing = false; this.endedAt = Date.now(); }
  },
  stop() {
    if (this.audio) { try { this.audio.pause(); this.audio.currentTime = 0; } catch (e) {} }
    this.audio = null; this.playing = false;
  },
  reset() { this.stop(); this.key = null; this.endedAt = 0; }
};

function playClip(item, key, onEnd) {
  if (typeof playGuideClip === 'function' && item) {
    playGuideClip(item, key, '', function () { if (onEnd) onEnd(); }, false);
  } else if (onEnd) setTimeout(onEnd, 800);
}
function playWelcome() {
  if (__welcomePlayed) return; __welcomePlayed = true;
  kdbg('WELCOME — Despite greeting');
  if (typeof setPreludeCard === 'function') setPreludeCard(0);
  if (typeof setSpotlight === 'function') setSpotlight(true);
  playClip((typeof PRELUDE !== 'undefined' ? PRELUDE[0] : null), 'prelude0', function () {
    __welcomeDone = true; kdbg('WELCOME done');
  });
}
function playSoapClip() {
  if (__soapStarted) return; __soapStarted = true;
  Nudge.stop(); showHandsScene(false);
  kdbg('SOAP — Despite explaining');
  if (typeof setPreludeCard === 'function') setPreludeCard(1);
  if (typeof setSpotlight === 'function') setSpotlight(true);
  playClip((typeof PRELUDE !== 'undefined' ? PRELUDE[1] : null), 'prelude1', function () {
    __soapClipDone = true; __soapWashFrames = 0; __soapGraceAt = 0;
    kdbg('SOAP clip done — your turn');
  });
}

// ── SINGLE VOICE CHANNEL ────────────────────────────────────────────────────
// Every Despite voice in kids mode (step narration, germ cheers, wrong clips)
// plays through here. Starting a new voice stops the current one, and the
// caller's onEnd fires on the REAL 'ended' event — so nothing ever overlaps
// and the next step waits for the previous voice to finish.
const Voice = {
  audio: null, token: 0,
  play(src, onEnd) {
    this.stop();
    const my = ++this.token;
    let done = false;
    const self = this;
    const fin = function () { if (done || my !== self.token) return; done = true; self.audio = null; if (onEnd) onEnd(); };
    try {
      this.audio = new Audio(src);
      this.audio.onended = fin;
      this.audio.onerror = function () { setTimeout(fin, 300); };
      this.audio.play().catch(function () { setTimeout(fin, 300); });
    } catch (e) { setTimeout(fin, 300); }
  },
  stop: function () { this.token++; if (this.audio) { try { this.audio.pause(); } catch (e) {} this.audio = null; } }
};

// ── STRICT PANEL: exactly ONE of {real hands, live cam, none} shows ─────────
// (Despite's video is only ever shown by playGuideClip in welcome/soap/rinse.)
function panelShow(which) {   // 'real' | 'cam' | 'none'
  const pv = (typeof panelVideoEl === 'function') ? panelVideoEl() : null;
  const rv = (typeof realVideoEl === 'function') ? realVideoEl() : null;
  const gif = document.getElementById('step-gif');
  const cam = document.getElementById('kids-panel-cam');
  if (pv) { try { pv.onended = null; pv.pause(); } catch (e) {} pv.style.display = 'none'; }  // hide Despite video
  if (gif) gif.style.display = 'none';
  if (which === 'real') {
    if (cam) cam.style.display = 'none'; __camInPanel = false;
  } else if (which === 'cam') {
    if (rv) { try { rv.pause(); } catch (e) {} rv.style.display = 'none'; }
    const rt = document.getElementById('real-tag'); if (rt) rt.classList.remove('on');
  } else {
    if (rv) rv.style.display = 'none'; if (cam) cam.style.display = 'none'; __camInPanel = false;
    const rt = document.getElementById('real-tag'); if (rt) rt.classList.remove('on');
  }
}

// ── GERM LOOP HELPERS ──────────────────────────────────────────────────────
function sendCmd(cmd) {
  try { if (kidsSocket && kidsSocket.readyState === WebSocket.OPEN)
    kidsSocket.send(JSON.stringify({ cmd: cmd })); } catch (e) {}
}
function armDetect() {
  if (__detectArmed) return;
  __detectArmed = true;
  sendCmd('start_detect');
  kdbg('detect armed — start_detect sent');
}

// TEACH: Despite explains (corner) while the verified real hands loop (main).
// When Despite's clip ends, arm detection (tells the server to start watching).
function kidsTeach(step) {
  if (__teachHold) return;                 // wait out the previous germ's cheer
  if (__teachStarted === step) return;     // already teaching this germ
  __teachStarted = step; __detectArmed = false; __retryShownFor = -1;

  // TEACHING SIDE: real-hands video only, NO camera, not scored.
  showRealPlusCam(false); showCamInPanel(false);
  if (typeof showRealStep === 'function') showRealStep(step);
  panelShow('real');
  if (typeof setDespiteInset === 'function') setDespiteInset(false);
  if (typeof hideGermBanner === 'function') hideGermBanner();
  if (typeof setStep === 'function') setStep(step);
  if (typeof bubblesExpression === 'function') bubblesExpression('neutral');
  if (typeof feedback === 'function') feedback('\uD83D\uDC40 Watch how to beat ' + STEPS[step].germName + '!', 'wait', '');
  kdbg('GERM ' + (step + 1) + ' — TEACH: 1s \u2192 instruction');

  // beat: 1s, then Despite's voice instruction (real hands keep playing, no cam)
  beat(step, 1000, function () {
    kdbg('GERM ' + (step + 1) + ' — instruction (voice + real hands, no cam)');
    Voice.play(mediaPath('voice_step' + (step + 1) + '.mp3'), function () {
      if (__teachStarted !== step) return;
      // beat: 1s, then the child's turn -> arm detection (PERFORM side)
      kdbg('GERM ' + (step + 1) + ' — instruction done: 1s \u2192 your turn');
      beat(step, 1000, function () {
        if (typeof feedback === 'function') feedback('\uD83D\uDC50 Your turn! Copy the hands!', 'ok', 'compliant');
        armDetect();
      });
    });
  });
}

// paced beat helper: fires only if we are still on the same germ (cancels
// automatically after a reset or after the germ advances).
function beat(step, ms, fn) {
  setTimeout(function () { if (__teachStarted === step) fn(); }, ms);
}

function playStepVoice(step, text, onEnd) {   // kept for compatibility
  Voice.play(mediaPath('voice_step' + (step + 1) + '.mp3'), onEnd);
}

// PERFORM SIDE: real-hands video in the panel + child's camera in the corner,
// detection on (server runs the 6s-correct / 10s-window attempt).
function kidsDetect(s) {
  if (!__detectArmed) armDetect();
  if (typeof showRealStep === 'function') showRealStep(s.kids_step);
  panelShow('real');                    // real hands in the panel (no Despite video)
  showRealPlusCam(true);                // + child's live camera in the corner
  const frac = Math.max(0, Math.min(1, s.progress || 0));
  if (typeof setLatherRing === 'function') setLatherRing(frac, s.detecting ? 'ok' : '');
  if (typeof setPips === 'function') setPips(0.5 + 0.45 * frac, !!s.detecting);
  if (typeof bubblesExpression === 'function') bubblesExpression(s.detecting ? 'ok' : 'neutral');
  const germ = STEPS[s.kids_step] ? STEPS[s.kids_step].germName : 'the germ';
  const left = (s.attempt_left != null) ? Math.ceil(s.attempt_left) + 's' : '';
  const got = (s.hold_s != null && s.needed != null) ? (s.hold_s.toFixed(1) + '/' + s.needed.toFixed(0) + 's') : '';
  if (typeof feedback === 'function')
    feedback(s.detecting ? ('\uD83D\uDD25 Keep scrubbing! ' + got)
                         : ('\uD83D\uDC50 Copy the hands to beat ' + germ + '! ' + left),
             s.detecting ? 'ok' : 'wait', s.detecting ? 'compliant' : '');
  kdbg('GERM ' + ((s.kids_step || 0) + 1) + ' — PERFORM att ' + (s.kids_attempt || 1) +
       ' \u00B7 ' + got + ' correct \u00B7 ' + left + ' left');
}

// MISS: back to the teaching side (real hands, no cam); Despite corrects by
// voice; 1s later attempt 2 arms.
function kidsRetry(s) {
  const step = s.kids_step;
  const tag = (s.kids_miss_kind === 'wrong') ? 'wrong_alert' : 'wrong_try';
  if (__retryShownFor !== step) {
    __retryShownFor = step; __detectArmed = false;
    showRealPlusCam(false); showCamInPanel(false);
    if (typeof showRealStep === 'function') showRealStep(step);
    panelShow('real');
    if (typeof bubblesExpression === 'function') bubblesExpression('wrong');
    if (typeof feedback === 'function')
      feedback(tag === 'wrong_alert' ? '\uD83D\uDE05 Whoops! Watch again!' : '\uD83E\uDD14 Not quite \u2014 copy me!', 'wrong', 'wrong');
    kdbg('GERM ' + (step + 1) + ' — MISS (' + s.kids_miss_kind + ') \u2192 ' + tag);
    Voice.play(mediaPath(tag + '.mp3'), function () {
      setTimeout(function () { armDetect(); }, 1000);   // 1s beat, then attempt 2
    });
  }
}

// OUTCOME: germ dies; cheer plays fully; 1s later the next germ's teach is released.
function kidsDefeat(step, mercy) {
  showRealPlusCam(false); showCamInPanel(false); __retryShownFor = -1;
  panelShow('none');
  if (typeof setLatherRing === 'function') setLatherRing(1, 'ok');
  if (typeof defeatGerm === 'function') defeatGerm(step);
  if (typeof showGermBanner === 'function') showGermBanner(step);
  if (typeof bubblesExpression === 'function') bubblesExpression('ok');
  if (typeof feedback === 'function')
    feedback((mercy ? '\uD83D\uDC4F Great effort! ' : '\uD83C\uDF89 ') + STEPS[step].germName + (mercy ? ' is gone!' : ' defeated!'),
             'ok', 'compliant');
  __teachStarted = -1; __teachHold = true;
  kdbg('GERM ' + (step + 1) + ' DEFEATED' + (mercy ? ' (mercy)' : '') + ' \u2192 cheer, 1s, next');
  Voice.play(mediaPath('germ_' + (step + 1) + '.mp3'), function () {
    setTimeout(function () { __teachHold = false; kdbg('cheer done + 1s \u2192 next germ'); }, 1000);
  });
}

// Play the step's narration as AUDIO only (no Despite video). Prefer the UI's
// own line-audio if present; fall back to a folder mp3, then to nothing.

// DETECT: real hands stay visible AND the child's live camera shows too, so
// they copy while being watched. 8-second attempt window.

// RETRY: first attempt missed -> play wrong_alert (different motion) or
// wrong_try (little/nothing), then re-arm detection for attempt 2.
let __retryShownFor = -1;

// DEFEAT: germ dies; celebrate, then hold before the next germ's teach.

function handleKidsStatus(s) {
  const phase = s.phase || '?';
  if (__kAwaitingPrime) { if (phase === 'PRIME') __kAwaitingPrime = false; else return; }
  __kPhase = phase;

  if (s.frame) { setHandsSceneFrame(s.frame); setCamFrame(s.frame); }

  // PRIME: hands gate — require BOTH hands (server sends both_hands)
  if (phase === 'PRIME') {
    const bothShown = (s.both_hands === true);
    if (__welcomeDone && !bothShown) {
      showHandsScene(true);
      Nudge.set(true, 'hands', PROMPT_HANDS());
      kdbg('HANDS GATE — need BOTH hands');
    } else {
      showHandsScene(false); Nudge.stop();
    }
    return;
  }
  showHandsScene(false);
  if (Nudge.key === 'hands') Nudge.stop();

  // SOAP
  if (phase === 'SOAP') {
    playSoapClip();
    // debounced "soaping RIGHT NOW" (filters twitches without latching forever)
    if (s.state === 'WASHING') __soapWashFrames = Math.min(SOAP_CONFIRM_FRAMES + 6, __soapWashFrames + 1);
    else __soapWashFrames = Math.max(0, __soapWashFrames - 1);
    const soaping = __soapWashFrames >= SOAP_CONFIRM_FRAMES;

    if (__soapClipDone) {
      if (!__camInPanel) showCamInPanel(true);          // child's live feed in Despite's spot
      if (soaping) {
        __soapGraceAt = 0;                              // they're doing it -> no nudge
        if (Nudge.key === 'soap') Nudge.stop();
        setCamCaption('\uD83E\uDEE7 Great lathering!');
      } else {
        setCamCaption('Your turn — rub with soap!');
        if (!__soapGraceAt) { __soapGraceAt = Date.now(); kdbg('Your turn — 3s grace'); }
        const graceLeft = SOAP_GRACE_MS - (Date.now() - __soapGraceAt);
        if (graceLeft <= 0) { Nudge.set(true, 'soap', PROMPT_SOAP()); kdbg('SOAP nudge'); }
        else kdbg('SOAP grace — ' + (graceLeft / 1000).toFixed(1) + 's');
      }
    }
    if (typeof setLatherRing === 'function') {
      const frac = (s.phase_remaining != null) ? Math.max(0, Math.min(1, 1 - s.phase_remaining / 5)) : 0;
      setLatherRing(frac, soaping ? 'ok' : '');
    }
    if (typeof feedback === 'function')
      feedback(soaping ? '\uD83D\uDC4F Great lathering!' : '\uD83D\uDC50 Rub your hands with soap!',
               soaping ? 'ok' : 'wait', soaping ? 'compliant' : '');
    return;
  }
  if (Nudge.key === 'soap') Nudge.stop();

  // ── GETSET: first germ incoming ─────────────────────────────────────────
  if (phase === 'GETSET') {
    showHandsScene(false); showCamInPanel(false); Nudge.reset();
    if (typeof feedback === 'function') feedback('Get ready — here comes ' + STEPS[0].germName + '!', 'wait', '');
    kdbg('GET SET — first germ coming');
    return;
  }

  // ── KIDS_RUN: teach -> detect -> defeat, per germ ───────────────────────
  if (phase === 'KIDS_RUN') {
    showHandsScene(false);
    const step = (s.kids_step != null) ? s.kids_step : 0;
    // a germ just died? (germs_beaten growth is the reliable signal)
    const nb = Array.isArray(s.germs_beaten) ? s.germs_beaten.length : __kBeaten;
    if (nb > __kBeaten) {
      const jb = Array.isArray(s.germs_beaten) ? s.germs_beaten[nb - 1] : __kBeaten;
      __kBeaten = nb;
      kidsDefeat(jb, !!s.kids_mercy);
    }
    if (s.kids_phase === 'teach') kidsTeach(step);
    else if (s.kids_phase === 'detect') kidsDetect(s);
    else if (s.kids_phase === 'retry') kidsRetry(s);
    return;
  }

  // ── RINSE: Despite shows rinsing, then the child rinses (motion-gated) ───
  if (phase === 'RINSE') {
    showHandsScene(false); showRealPlusCam(false);
    kidsRinse(s);
    return;
  }

  // ── DONE: reward + score — but not until Despite's rinse clip has finished
  //    (so his narration never collides with the congratulations) ──────────
  if (phase === 'DONE') {
    if (__rinseClipDone) kidsFinish(s);
    else __donePending = s;          // wait for the rinse clip to end
    return;
  }
}

// RINSE: Despite's video plays fully; THEN the child's camera comes on and they
// rinse with both hands for the full window before the score appears.
let __rinseStarted = false, __rinseClipDone = false, __donePending = null;
function kidsRinse(s) {
  if (!__rinseStarted) {
    __rinseStarted = true; __rinseClipDone = false;
    showCamInPanel(false);
    if (typeof hideGermBanner === 'function') hideGermBanner();
    if (typeof bubblesExpression === 'function') bubblesExpression('ok');
    kdbg('RINSE — Despite showing (video)');
    playClip(RINSE, 'rinse', function () {
      __rinseClipDone = true;
      showCamInPanel(true); setCamCaption('\uD83D\uDCA7 Rinse with BOTH hands!');
      kdbg('RINSE clip done — your turn to rinse');
      if (__donePending) { const d = __donePending; __donePending = null; kidsFinish(d); }
    });
  }
  if (__rinseClipDone) {
    showCamInPanel(true);
    const total = s.rinse_total || 10;
    const rem = (s.phase_remaining != null) ? s.phase_remaining : total;
    const frac = Math.max(0, Math.min(1, 1 - rem / total));
    const washing = (s.state === 'WASHING');
    const both = (s.both_hands === true);
    if (typeof setLatherRing === 'function') setLatherRing(frac, (washing && both) ? 'ok' : '');
    setCamCaption(both ? '\uD83D\uDCA7 Rinsing! Keep going!' : '\uD83D\uDCA7 Use BOTH hands under the water!');
    if (typeof feedback === 'function')
      feedback((washing && both) ? '\uD83D\uDCA7 Rinsing! Wash the germs away!' : '\uD83D\uDCA7 Rub BOTH hands under the water!',
               (washing && both) ? 'ok' : 'wait', (washing && both) ? 'compliant' : '');
    kdbg('RINSE — ' + rem.toFixed(1) + 's left' + (both ? '' : ' (need both hands)'));
  }
}

// FINAL SCORE: score = germs LEARNED / 7 (mercy-passes are NOT counted),
// so the reward honestly reflects what the child could do. Reuses the UI's
// score-screen elements + confetti + history.
let __kDone = false;
function kidsFinish(s) {
  if (__kDone) return; __kDone = true;
  try { clearInterval(timerInt); } catch (e) {}

  const res = Array.isArray(s.kids_results) ? s.kids_results : [];
  const learned = (s.germs_learned != null) ? s.germs_learned : res.filter(r => r.learned).length;
  const beaten = (s.germs_beaten != null) ? s.germs_beaten : res.filter(r => r.defeated).length;
  const score = (s.score != null) ? s.score : Math.round(100 * learned / 7);
  const g = (typeof gradeFor === 'function') ? gradeFor(score) : { title: '', key: '' };
  const elapsed = (s.duration_s != null) ? s.duration_s
                  : Math.floor((Date.now() - (sessionStart || Date.now())) / 1000);

  showScreen('score');
  if (typeof spawnConfetti === 'function') spawnConfetti(70);

  // stars = germs actually learned
  const starEl = document.getElementById('sc-stars');
  if (starEl) starEl.textContent = '\u2B50'.repeat(Math.min(learned, 7)) + '\u2606'.repeat(Math.max(0, 7 - learned));

  setTimeout(function () {
    const arc = document.getElementById('sarc'); if (arc) arc.style.strokeDashoffset = 515 * (1 - score / 100);
  }, 200);
  let c = 0;
  const ni = setInterval(function () {
    c = Math.min(c + 2, score);
    const el = document.getElementById('sc-num'); if (el) el.textContent = c;
    if (c >= score) clearInterval(ni);
  }, 22);

  const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
  set('sc-grade', g.title);
  const rescued = beaten - learned;
  set('sc-msg', learned === 7 ? '\uD83C\uDFC6 PERFECT! You learned all 7 steps!'
              : ('You learned ' + learned + ' of 7 steps' + (rescued > 0 ? ' (' + rescued + ' still need practice)' : '') + '!'));
  set('st-germs', learned + '/7');
  set('st-time', elapsed + 's');
  set('st-order', learned === 7 ? '\u2713 Mastered' : '~ Keep practising');

  // badges: learned = green ✓, rescued/not-learned = "practise" ~, missed = ✗
  const bg = document.getElementById('sc-badges');
  if (bg) {
    bg.innerHTML = '';
    STEPS.forEach(function (st, i) {
      const r = res[i] || {};
      let cls, mark;
      if (r.learned) { cls = 'ok'; mark = '\u2713 '; }
      else if (r.defeated) { cls = 'miss'; mark = '\u223C '; }   // rescued / not learned
      else { cls = 'miss'; mark = '\u2717 '; }
      const b = document.createElement('div');
      b.className = 'badge ' + cls;
      b.textContent = mark + st.emoji + ' ' + st.germName;
      bg.appendChild(b);
    });
  }

  try { sayDespite(learned === 7 ? 'score_perfect' : 'score_partial',
                   (typeof L === 'function' && L().scoreMsg) ? L().scoreMsg(score, g.title) : '', true); } catch (e) {}

  if (typeof history_ !== 'undefined' && typeof saveHistory === 'function') {
    history_.unshift({ id: Date.now(), datetime: new Date().toLocaleString(),
      germs: learned, duration: elapsed, score: score, grade: g.key, title: g.title,
      order: learned === 7 ? 'Mastered' : 'Partial' });
    if (history_.length > 50) history_ = history_.slice(0, 50);
    saveHistory(history_);
  }
  kdbg('DONE \u2705  learned ' + learned + '/7 · score ' + score + ' · ' + g.title);
  // auto-return to the launcher after 30s if nobody taps
  try { clearTimeout(window.__kidsAutoHome); } catch (e) {}
  window.__kidsAutoHome = setTimeout(goLauncher, 30000);
}

// ── back-to-launcher button (always visible) + navigation ─────────────────
function goLauncher() {
  try { if (typeof Voice !== 'undefined') Voice.stop(); } catch (e) {}
  try { if (kidsSocket) kidsSocket.close(); } catch (e) {}
  window.location.href = 'washit_launcher.html';
}
function ensureHomeButton() {
  if (document.getElementById('kids-home-btn')) return;
  const b = document.createElement('button');
  b.id = 'kids-home-btn';
  b.textContent = '\u2302 Change Mode';
  b.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:9500;' +
    'font:700 13px Nunito,sans-serif;color:#0b1e3a;background:#00d4ff;border:none;' +
    'border-radius:10px;padding:9px 13px;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.35)';
  b.onclick = goLauncher;
  (document.getElementById('app') || document.body).appendChild(b);
}

function connectKids(wsUrl) {
  if (kidsSocket) kidsSocket.close();
  __kAwaitingPrime = true; __kPhase = null;
  __welcomePlayed = false; __welcomeDone = false;
  __soapStarted = false; __soapClipDone = false; __soapGraceAt = 0;
  __camInPanel = false; __soapWashFrames = 0;
  __kBeaten = 0; __detectArmed = false; __teachStarted = -1; __teachHold = false;
  __cornerCam = false; __retryShownFor = -1; showRealPlusCam(false);
  __rinseStarted = false; __rinseClipDone = false; __kDone = false; __donePending = null;
  ensureHomeButton();
  if (typeof buildGermRow === 'function') buildGermRow();
  Nudge.reset(); if (typeof Voice !== 'undefined') Voice.stop();
  showHandsScene(false); showCamInPanel(false);
  if (typeof showScreen === 'function') showScreen('game');
  kdbg('connecting to ' + wsUrl + ' ...');
  kidsSocket = new WebSocket(wsUrl);
  kidsSocket.onopen = function () {
    kdbg('connected — starting welcome');
    kidsSocket.send(JSON.stringify({ cmd: 'set_mode', mode: 'kids' }));
    kidsSocket.send('reset');
    playWelcome();
  };
  kidsSocket.onmessage = function (e) {
    try { handleKidsStatus(JSON.parse(e.data)); } catch (err) { console.error(err); }
  };
  kidsSocket.onclose = function () { kdbg('disconnected'); };
  kidsSocket.onerror = function () { kdbg('\u26A0\uFE0F cannot reach server — is server.py running?'); };
}

function kidsServerAddr() {
  const box = document.getElementById('kids-server-addr') || document.getElementById('live-server-addr');
  return (box && box.value) ? box.value : (window.__kidsWs || 'ws://localhost:8001/ws');
}
startDemo = function () { connectKids(kidsServerAddr()); };

// launcher hands off the server address + optional autostart via URL params
(function kidsUrlParams() {
  function boot() {
    let p; try { p = new URLSearchParams(window.location.search); } catch (e) { return; }
    const addr = p.get('ws');
    if (addr) {
      window.__kidsWs = addr;
      const box = document.getElementById('kids-server-addr') || document.getElementById('live-server-addr');
      if (box) box.value = addr;
    }
    if (p.get('autoconnect') === '1') setTimeout(function () { connectKids(kidsServerAddr()); }, 700);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
