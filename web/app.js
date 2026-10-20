/**
 * web/app.js
 * Operation Archival: Atmospheric Nuclear Detonation Generator
 * Interactive parameter switcher, video controller, disentanglement matrix, and archive explorer.
 */

// Global State
const state = {
  burst: 'air-burst',
  yieldTier: 'tactical-yield',
  setting: 'desert-flats',
  phase: 'expansion',
  film: 'colour-restored',
  smoke: false,
  speed: 1.0,
  isPlaying: true,
  currentTab: 'generator'
};

// Video Mapping Dictionary (Guaranteed Tier 1 Static Matrix)
const videoMap = {
  'air-burst': 'videos/sample_airburst.mp4',
  'surface-tower': 'videos/sample_surface_tower.mp4',
  'surface-barge': 'videos/sample_surface_barge.mp4',
  'underwater': 'videos/test_underwater.mp4'
};

// Parameter Descriptions
const paramDescriptions = {
  burst: {
    'air-burst': 'High altitude / airdrop detonation without surface debris entrainment.',
    'surface-tower': 'Steel cab tower burst (100–700 ft) entraining ground soil into high-density radioactive stem.',
    'surface-barge': 'Anchored barge in lagoon waters producing intense localized steam column and radioactive fallout.',
    'underwater': 'Sub-surface detonation producing a massive hollow water column, cauliflower dome, and base surge.'
  },
  yieldTier: {
    'sub-kiloton': 'Fractional kiloton tactical experiment (<1 kt, e.g. Buster Able).',
    'tactical-yield': 'Tactical fission device (1–20 kt, e.g. Trinity / Hiroshima / Crossroads Able class).',
    'intermediate-yield': 'High-yield boosted fission device (20 kt–1 Mt, e.g. Greenhouse George).',
    'megaton-class': 'Multi-megaton thermonuclear stage detonation (>1 Mt, e.g. Ivy Mike / Castle Bravo).'
  }
};

// Historical Equivalents
const historicalEquivalents = {
  'air-burst': 'SHOT SUNSET (OPERATION DOMINIC 1962, PPG)',
  'surface-tower': 'SHOT APPLE-2 (OPERATION TEAPOT 1955, NTS)',
  'surface-barge': 'SHOT ZUNI (OPERATION REDWING 1956, BIKINI)',
  'underwater': 'SHOT BAKER (OPERATION CROSSROADS 1946, BIKINI)'
};

// Telemetry Estimates
const telemetryData = {
  'air-burst': { alt: '1,500 FT (AIR)', mach: 'MACH 3.1 → SUBSONIC' },
  'surface-tower': { alt: '300 FT (TOWER)', mach: 'MACH 2.8 → SUBSONIC' },
  'surface-barge': { alt: '0 FT (LAGOON)', mach: 'MACH 3.4 → SUBSONIC' },
  'underwater': { alt: '-90 FT (SUB-SURFACE)', mach: 'SUPERSONIC CAVITATION' }
};

// DOM Elements
const video = document.getElementById('main-video');
const videoSource = document.getElementById('video-source');
const playPauseBtn = document.getElementById('play-pause-btn');
const scrubberBar = document.getElementById('scrubber-bar');
const scrubberFill = document.getElementById('scrubber-fill');
const timeReadout = document.getElementById('time-readout');
const telemetryStamp = document.getElementById('telemetry-stamp');
const promptDisplay = document.getElementById('active-prompt-display');
const crtOverlay = document.getElementById('crt-overlay');
const crtToggle = document.getElementById('crt-toggle');
const loaderOverlay = document.getElementById('loader-overlay');
const synthesizeBtn = document.getElementById('synthesize-btn');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupSegmentedControls();
  setupSelectsAndSwitches();
  setupPlayerControls();
  setupActionButtons();
  buildScrubberTicks();
  updateView();
  loadHistoricalArchive();
  setupDisentanglementGrid();
});

// Tab Navigation
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      tab.classList.add('active');
      const tabId = tab.getAttribute('data-tab');
      document.getElementById(`pane-${tabId}`).classList.add('active');
      state.currentTab = tabId;
      
      if (tabId === 'disentangle') {
        renderDisentanglementGrid();
      }
    });
  });
}

// Segmented Controls (Burst, Yield, Phase)
function setupSegmentedControls() {
  setupSingleSegmented('burst-selector', 'burst', (val) => {
    document.getElementById('burst-desc').textContent = paramDescriptions.burst[val];
  });
  setupSingleSegmented('yield-selector', 'yieldTier', (val) => {
    document.getElementById('yield-desc').textContent = paramDescriptions.yieldTier[val];
  });
  setupSingleSegmented('phase-selector', 'phase');
}

function setupSingleSegmented(containerId, stateKey, callback) {
  const container = document.getElementById(containerId);
  const btns = container.querySelectorAll('.seg-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state[stateKey] = btn.getAttribute('data-value');
      if (callback) callback(state[stateKey]);
      updateView();
    });
  });
}

// Selects and Switches
function setupSelectsAndSwitches() {
  document.getElementById('setting-select').addEventListener('change', (e) => {
    state.setting = e.target.value;
    updateView();
  });
  
  document.getElementById('film-select').addEventListener('change', (e) => {
    state.film = e.target.value;
    updateView();
  });
  
  document.getElementById('smoke-switch').addEventListener('change', (e) => {
    state.smoke = e.target.checked;
    updateView();
  });
}

// Player Controls
function setupPlayerControls() {
  // Play/Pause
  playPauseBtn.addEventListener('click', togglePlayPause);
  video.addEventListener('click', togglePlayPause);

  // Time & Scrubber Update
  video.addEventListener('timeupdate', () => {
    if (!video.duration) return;
    const pct = (video.currentTime / video.duration) * 100;
    scrubberFill.style.width = `${pct}%`;
    
    const currentFrame = Math.min(16, Math.max(1, Math.floor((video.currentTime / video.duration) * 16) + 1));
    timeReadout.textContent = `${String(currentFrame).padStart(2, '0')} / 16 (${video.currentTime.toFixed(2)}s)`;
    
    // Telemetry stamp update
    telemetryStamp.textContent = `REC: 16MM 1962-DOMINIC // F:${String(currentFrame).padStart(2, '0')}/16 // ${state.phase.toUpperCase()}`;
  });

  // Scrubber click
  scrubberBar.addEventListener('click', (e) => {
    const rect = scrubberBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    if (video.duration) {
      video.currentTime = pct * video.duration;
    }
  });

  // Playback Speed
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.speed = parseFloat(btn.getAttribute('data-speed'));
      video.playbackRate = state.speed;
    });
  });

  // Frame Steppers (8 fps -> 0.125s per frame)
  document.getElementById('step-prev-btn').addEventListener('click', () => {
    video.pause();
    state.isPlaying = false;
    playPauseBtn.textContent = '▶';
    video.currentTime = Math.max(0, video.currentTime - 0.125);
  });

  document.getElementById('step-next-btn').addEventListener('click', () => {
    video.pause();
    state.isPlaying = false;
    playPauseBtn.textContent = '▶';
    video.currentTime = Math.min(video.duration || 2.0, video.currentTime + 0.125);
  });

  // CRT Telecine Filter Toggle
  crtToggle.addEventListener('change', (e) => {
    crtOverlay.style.display = e.target.checked ? 'block' : 'none';
  });
}

function togglePlayPause() {
  if (video.paused) {
    video.play();
    state.isPlaying = true;
    playPauseBtn.textContent = '⏸';
  } else {
    video.pause();
    state.isPlaying = false;
    playPauseBtn.textContent = '▶';
  }
}

function buildScrubberTicks() {
  const ticksContainer = document.getElementById('scrubber-ticks');
  ticksContainer.innerHTML = '';
  for (let i = 0; i <= 16; i++) {
    const tick = document.createElement('div');
    tick.className = 'scrubber-tick';
    tick.style.left = `${(i / 16) * 100}%`;
    ticksContainer.appendChild(tick);
  }
}

// Action Buttons
function setupActionButtons() {
  // Randomize Button
  document.getElementById('randomize-btn').addEventListener('click', () => {
    const bursts = ['air-burst', 'surface-tower', 'surface-barge', 'underwater'];
    const yields = ['sub-kiloton', 'tactical-yield', 'intermediate-yield', 'megaton-class'];
    const settings = ['desert-flats', 'tropical-ocean'];
    const phases = ['early-fireball', 'expansion', 'stabilized-cloud'];
    const films = ['colour-restored', 'black-and-white', 'faded-technicolor'];

    state.burst = bursts[Math.floor(Math.random() * bursts.length)];
    state.yieldTier = yields[Math.floor(Math.random() * yields.length)];
    state.setting = settings[Math.floor(Math.random() * settings.length)];
    state.phase = phases[Math.floor(Math.random() * phases.length)];
    state.film = films[Math.floor(Math.random() * films.length)];
    state.smoke = Math.random() > 0.6;

    // Update UI elements
    updateSegmentedUI('burst-selector', state.burst);
    updateSegmentedUI('yield-selector', state.yieldTier);
    updateSegmentedUI('phase-selector', state.phase);
    document.getElementById('setting-select').value = state.setting;
    document.getElementById('film-select').value = state.film;
    document.getElementById('smoke-switch').checked = state.smoke;

    document.getElementById('burst-desc').textContent = paramDescriptions.burst[state.burst];
    document.getElementById('yield-desc').textContent = paramDescriptions.yieldTier[state.yieldTier];

    triggerSynthesisAnimation();
  });

  // Synthesize Video Button
  synthesizeBtn.addEventListener('click', () => {
    triggerSynthesisAnimation();
  });

  // Copy Prompt
  document.getElementById('copy-prompt-btn').addEventListener('click', () => {
    const text = promptDisplay.textContent.trim();
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copy-prompt-btn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  });

  // Copy CLI Command
  document.getElementById('copy-cli-btn').addEventListener('click', () => {
    const text = document.getElementById('cli-code-display').textContent.trim();
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('copy-cli-btn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  });
}

function updateSegmentedUI(containerId, value) {
  const container = document.getElementById(containerId);
  container.querySelectorAll('.seg-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-value') === value);
  });
}

function triggerSynthesisAnimation() {
  loaderOverlay.classList.remove('hidden');
  setTimeout(() => {
    updateView();
    loaderOverlay.classList.add('hidden');
    video.play();
    state.isPlaying = true;
    playPauseBtn.textContent = '⏸';
  }, 950);
}

// Master View Update
function updateView() {
  // Construct Conditioning Prompt using canonical schema
  const smokeStr = state.smoke ? 'with smoke-trails' : 'without smoke-trails';
  let terrain = 'desert-flats';
  if (state.setting === 'tropical-ocean') {
    terrain = 'tropical-ocean';
  }
  const lighting = state.lighting || (state.setting && state.setting.includes('night') ? 'night' : (state.setting && state.setting.includes('dawn') ? 'dawn' : 'day'));
  const prompt = `${state.burst} nuclear detonation, ${state.yieldTier}, ${terrain}, ${lighting}, ${state.phase} phase, ${state.film} archival footage, ${smokeStr}`;
  promptDisplay.textContent = prompt;

  // Update CLI Code display
  const cliCommand = `python inference/generate.py \\
  --burst_type ${state.burst} \\
  --yield_tier ${state.yieldTier} \\
  --terrain ${terrain} \\
  --lighting ${lighting} \\
  --phase ${state.phase} \\
  --film_look ${state.film} \\
  ${state.smoke ? '--smoke_trails' : ''} \\
  --output samples/${state.burst}_${state.phase}.mp4`;
  document.getElementById('cli-code-display').textContent = cliCommand;

  // Update Telemetry Bar
  const tel = telemetryData[state.burst] || telemetryData['air-burst'];
  document.getElementById('tel-alt').textContent = tel.alt;
  document.getElementById('tel-mach').textContent = tel.mach;
  document.getElementById('tel-shot').textContent = historicalEquivalents[state.burst] || 'OPERATION DOMINIC';

  // Video Source Switching
  const targetVideo = videoMap[state.burst] || videoMap['air-burst'];
  if (videoSource.getAttribute('src') !== targetVideo) {
    videoSource.setAttribute('src', targetVideo);
    video.load();
    video.playbackRate = state.speed;
    video.play().catch(() => {});
  }
}

// Tab 2: Disentanglement Grid
function setupDisentanglementGrid() {
  const sweepSelect = document.getElementById('sweep-select');
  sweepSelect.addEventListener('change', renderDisentanglementGrid);
}

function renderDisentanglementGrid() {
  const grid = document.getElementById('disentangle-grid');
  grid.innerHTML = '';
  
  const sweep = document.getElementById('sweep-select').value;
  
  let variations = [];
  if (sweep === 'burst_type') {
    variations = [
      { name: 'Air Burst', tag: 'air-burst', desc: 'No soil entrainment; spherical fireball', video: videoMap['air-burst'] },
      { name: 'Surface Tower', tag: 'surface-tower', desc: 'High ground soil suction; thick radioactive stem', video: videoMap['surface-tower'] },
      { name: 'Surface Barge', tag: 'surface-barge', desc: 'Lagoon surface vaporization; steam cloud', video: videoMap['surface-barge'] },
      { name: 'Underwater', tag: 'underwater', desc: 'Hollow water column & massive base surge', video: videoMap['underwater'] }
    ];
  } else if (sweep === 'yield_tier') {
    variations = [
      { name: 'Sub-Kiloton (<1 kt)', tag: 'sub-kiloton', desc: 'Rapid cooling, compact fireball', video: videoMap['air-burst'] },
      { name: 'Tactical (1–20 kt)', tag: 'tactical-yield', desc: 'Standard Hiroshima/Trinity scale mushroom', video: videoMap['surface-tower'] },
      { name: 'Intermediate (20 kt–1 Mt)', tag: 'intermediate-yield', desc: 'High thermal pulse; massive toroidal cap', video: videoMap['underwater'] },
      { name: 'Megaton-Class (>1 Mt)', tag: 'megaton-class', desc: 'Stratospheric punchthrough; multi-tiered cloud', video: videoMap['surface-barge'] }
    ];
  } else if (sweep === 'phase') {
    variations = [
      { name: 'Early Fireball (t=0–0.3s)', tag: 'early-fireball', desc: 'High-temperature radiation front & Wilson cloud', video: videoMap['air-burst'] },
      { name: 'Fluid Expansion (t=0.3–1.2s)', tag: 'expansion', desc: 'Toroidal vortex formation and buoyant ascent', video: videoMap['surface-tower'] },
      { name: 'Stabilized Cloud (t=1.2–2.0s)', tag: 'stabilized-cloud', desc: 'Thermal equilibrium mushroom cloud cap', video: videoMap['surface-barge'] },
      { name: 'Base Surge Progression', tag: 'base-surge', desc: 'Radial outward density flow along surface', video: videoMap['underwater'] }
    ];
  } else {
    variations = [
      { name: 'High-Contrast B&W', tag: 'b&w-16mm', desc: 'Orthochromatic 16mm film; Trinity/Crossroads look', video: videoMap['air-burst'] },
      { name: 'Faded Technicolor', tag: 'faded-color', desc: 'Late 1950s dye fading and sepia degradation', video: videoMap['surface-tower'] },
      { name: 'Colour-Restored', tag: 'restored-16mm', desc: 'LLNL declassified modern digital scan', video: videoMap['underwater'] },
      { name: 'High-Grain Diagnostic', tag: 'high-grain', desc: 'High-speed Mitchell film camera telemetry', video: videoMap['surface-barge'] }
    ];
  }

  variations.forEach(item => {
    const card = document.createElement('div');
    card.className = 'grid-card';
    card.innerHTML = `
      <div class="grid-card-media">
        <video autoplay loop muted playsinline src="${item.video}"></video>
      </div>
      <div class="grid-card-info">
        <h4 class="grid-card-title">${item.name}</h4>
        <p class="grid-card-meta">${item.desc}</p>
      </div>
    `;
    grid.appendChild(card);
  });
}

// Tab 3: Historical Archive Explorer
let allShots = [];

async function loadHistoricalArchive() {
  try {
    const res = await fetch('historical_shots.json');
    if (res.ok) {
      allShots = await res.json();
      renderArchiveTable(allShots);
      setupArchiveFilters();
    }
  } catch (err) {
    console.warn('Could not load historical_shots.json dynamically:', err);
  }
}

function renderArchiveTable(shots) {
  const tbody = document.getElementById('archive-tbody');
  tbody.innerHTML = '';
  
  shots.forEach(shot => {
    const tr = document.createElement('tr');
    
    let typeClass = 'air';
    if (shot.burst_type.includes('tower')) typeClass = 'tower';
    else if (shot.burst_type.includes('barge')) typeClass = 'barge';
    else if (shot.burst_type.includes('underwater')) typeClass = 'uw';

    let splitBadge = `<span class="type-pill" style="background: rgba(56, 189, 248, 0.15); border-color: rgba(56, 189, 248, 0.4); color: #38bdf8;">${shot.split || 'train'}</span>`;
    if (shot.split === 'val') {
      splitBadge = `<span class="type-pill" style="background: rgba(234, 179, 8, 0.15); border-color: rgba(234, 179, 8, 0.4); color: #eab308;">val</span>`;
    } else if (shot.split === 'test') {
      splitBadge = `<span class="type-pill" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); color: #ef4444;">test</span>`;
    }

    tr.innerHTML = `
      <td><strong>${shot.shot_name}</strong></td>
      <td>${shot.operation}</td>
      <td>${shot.date}</td>
      <td>${shot.location}</td>
      <td><span class="type-pill ${typeClass}">${shot.burst_type}</span></td>
      <td>${shot.yield_raw}</td>
      <td>${shot.yield_tier}</td>
      <td>${splitBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

function setupArchiveFilters() {
  const searchInput = document.getElementById('archive-search');
  const typeFilter = document.getElementById('archive-type-filter');

  const filterHandler = () => {
    const query = searchInput.value.toLowerCase().trim();
    const type = typeFilter.value;

    const filtered = allShots.filter(s => {
      const matchesQuery = !query || 
        s.shot_name.toLowerCase().includes(query) || 
        s.operation.toLowerCase().includes(query) || 
        s.location.toLowerCase().includes(query);
        
      const matchesType = type === 'all' || 
        (type === 'AIR' && s.burst_type.includes('air')) ||
        (type === 'TOWER' && s.burst_type.includes('tower')) ||
        (type === 'BARGE' && s.burst_type.includes('barge')) ||
        (type === 'UW' && s.burst_type.includes('underwater'));

      return matchesQuery && matchesType;
    });

    renderArchiveTable(filtered);
  };

  searchInput.addEventListener('input', filterHandler);
  typeFilter.addEventListener('change', filterHandler);
}
