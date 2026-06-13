INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WB Vision</title>
<style>
  :root { --bg:#15171c; --panel:#1d2027; --line:#2c313c; --txt:#e6e8ee; --muted:#8b909c;
          --acc:#3da5ff; --ok:#3ecf6a; --warn:#ffcf4a; --aim:#ff5a5a; --floor:#3cc6ff; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--txt); }
  .wrap { display:flex; gap:14px; padding:14px; align-items:flex-start; }
  .video { position:relative; flex:1 1 auto; background:#000; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  #stream { display:block; width:100%; height:auto; cursor:crosshair; }
  #hint { position:absolute; left:0; right:0; bottom:0; text-align:center; padding:6px;
          background:rgba(0,0,0,.6); font-weight:600; display:none; }
  .panel { width:360px; flex:0 0 360px; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:12px; max-height:calc(100vh - 28px); overflow:auto; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
       margin:16px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
  h2:first-child { margin-top:0; }
  .btn { display:inline-block; padding:8px 10px; margin:3px 3px 3px 0; border:1px solid var(--line);
         background:#262b34; color:var(--txt); border-radius:6px; cursor:pointer; font-size:13px; }
  .btn:hover { border-color:var(--acc); }
  .btn.active { background:var(--acc); border-color:var(--acc); color:#06121f; font-weight:600; }
  .btn.ok { background:#1d3a26; border-color:#2f6b43; }
  .row { display:flex; align-items:center; gap:6px; margin:4px 0; }
  .row label { width:120px; color:var(--muted); }
  .row input { flex:1; background:#11141a; border:1px solid var(--line); color:var(--txt);
               border-radius:5px; padding:5px 6px; width:60px; }
  .grid4 { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .stat { display:flex; justify-content:space-between; padding:2px 0; }
  .stat b { color:#fff; font-weight:600; }
  .pill { font-size:12px; padding:1px 6px; border-radius:10px; background:#262b34; }
  .good { color:var(--ok); } .bad { color:var(--warn); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td { padding:2px 4px; border-top:1px solid var(--line); }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <div class="video">
    <img id="stream" src="/video" alt="stream">
    <div id="hint"></div>
  </div>

  <div class="panel">
    <h2>Status</h2>
    <div class="stat"><span>FPS in / inf</span><b id="s_fps">-</b></div>
    <div class="stat"><span>Infer ms</span><b id="s_ms">-</b></div>
    <div class="stat"><span>CPU / RAM</span><b id="s_cpu">-</b></div>
    <div class="stat"><span>People</span><b id="s_ppl">-</b></div>
    <div class="stat"><span>Calibration</span><b id="s_cal">-</b></div>

    <h2>Calibration</h2>
    <div>
      <span class="btn" id="b_aim">Set AIM</span>
      <span class="btn" id="b_quad">Set 4 corners</span>
    </div>
    <div>
      <span class="btn" id="b_quad_clear">Clear corners</span>
      <span class="btn ok" id="b_save">Save</span>
    </div>

    <h2>Edges (m) &amp; angles (deg)</h2>
    <div class="grid4">
      <div class="row"><label>AB</label><input id="e0" type="number" step="0.01"></div>
      <div class="row"><label>A1</label><input id="a0" type="number" step="1"></div>
      <div class="row"><label>BC</label><input id="e1" type="number" step="0.01"></div>
      <div class="row"><label>A2</label><input id="a1" type="number" step="1"></div>
      <div class="row"><label>CD</label><input id="e2" type="number" step="0.01"></div>
      <div class="row"><label>A3</label><input id="a2" type="number" step="1"></div>
      <div class="row"><label>DA</label><input id="e3" type="number" step="0.01"></div>
      <div class="row"><label>A4</label><input id="a3" type="number" step="1"></div>
    </div>
    <div class="muted" id="closure">closure: -</div>

    <h2>Camera &amp; room</h2>
    <div class="row"><label>Cam height, m</label><input id="v_camera_height_m" type="number" step="0.01"></div>
    <div class="row"><label>Dist to AIM, m</label><input id="v_cam_to_aim_m" type="number" step="0.01"></div>
    <div class="row"><label>Room W, m</label><input id="v_room_width_m" type="number" step="0.01"></div>
    <div class="row"><label>Room D, m</label><input id="v_room_depth_m" type="number" step="0.01"></div>

    <h2>Runtime tuning</h2>
    <div id="tuning"></div>

    <h2>Zones</h2>
    <div>
      <span class="btn" id="b_zone">Draw zone</span>
      <span class="btn" id="b_zone_finish" style="display:none">Finish</span>
    </div>
    <div id="zone_list"></div>

    <h2>Tracks</h2>
    <table><tbody id="tracks"></tbody></table>
  </div>
</div>

<script>
const src = {width:0, height:0};
let mode = null;          // null | 'aim' | 'quad' | 'zone'
let quadIndex = 0;
let zonePts = [];
const img = document.getElementById('stream');
const hint = document.getElementById('hint');

function setMode(m, text) {
  mode = m;
  document.getElementById('b_aim').classList.toggle('active', m==='aim');
  document.getElementById('b_quad').classList.toggle('active', m==='quad');
  document.getElementById('b_zone').classList.toggle('active', m==='zone');
  document.getElementById('b_zone_finish').style.display = (m==='zone' && zonePts.length>=3) ? 'inline-block' : 'none';
  hint.style.display = m ? 'block' : 'none';
  hint.textContent = text || '';
}
async function post(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body||{})});
  return r.json();
}

document.getElementById('b_aim').onclick = () => setMode(mode==='aim'?null:'aim', 'Click the AIM point on the floor');
document.getElementById('b_quad').onclick = () => { quadIndex=0; setMode(mode==='quad'?null:'quad', 'Click corner 1 (origin), then 2,3,4 clockwise'); };
document.getElementById('b_quad_clear').onclick = async () => { await post('/api/quad/clear'); };
document.getElementById('b_save').onclick = async () => { await post('/api/save'); flash('Saved'); };
document.getElementById('b_zone').onclick = () => { zonePts=[]; setMode(mode==='zone'?null:'zone', 'Click zone polygon points (3+), then Finish'); };
document.getElementById('b_zone_finish').onclick = async () => {
  if (zonePts.length>=3) {
    const name = prompt('Zone name:', 'zone') || 'zone';
    await post('/api/zone/add', {name, polygon_px: zonePts});
  }
  zonePts=[]; setMode(null);
};

img.addEventListener('click', async (e) => {
  if (!mode || !src.width) return;
  const r = img.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) / r.width * src.width);
  const y = Math.round((e.clientY - r.top) / r.height * src.height);
  if (mode==='aim') { await post('/api/aim', {x,y}); setMode(null); }
  else if (mode==='quad') {
    await post('/api/quad_point', {index:quadIndex, x, y});
    quadIndex++;
    if (quadIndex>=4) setMode(null); else setMode('quad', 'Click corner '+(quadIndex+1)+' of 4');
  }
  else if (mode==='zone') {
    zonePts.push([x,y]);
    setMode('zone', 'Zone points: '+zonePts.length+' (3+ then Finish)');
  }
});

// edge/angle/value inputs -> POST on change
for (let i=0;i<4;i++) {
  document.getElementById('e'+i).addEventListener('change', ev => post('/api/edge', {index:i, value:parseFloat(ev.target.value)||0}));
  document.getElementById('a'+i).addEventListener('change', ev => post('/api/angle', {index:i, value:parseFloat(ev.target.value)||0}));
}
['camera_height_m','cam_to_aim_m','room_width_m','room_depth_m'].forEach(k => {
  document.getElementById('v_'+k).addEventListener('change', ev => post('/api/value', {key:k, value:parseFloat(ev.target.value)||0}));
});

let editing = null;
function trackFocus(el){
  el.addEventListener('focus', () => editing = el.id);
  el.addEventListener('blur', () => { if (editing===el.id) editing=null; });
}
document.querySelectorAll('input').forEach(trackFocus);

let tuningBuilt = false;
function buildTuning(specs){
  const box = document.getElementById('tuning');
  box.innerHTML = '';
  Object.keys(specs).forEach(k => {
    const s = specs[k];
    const row = document.createElement('div'); row.className = 'row';
    const lab = document.createElement('label'); lab.textContent = s.label;
    const rng = document.createElement('input'); rng.type='range'; rng.id='t_'+k;
    rng.min=s.lo; rng.max=s.hi; rng.step=s.step; rng.style.flex='1';
    const out = document.createElement('span'); out.id='tv_'+k; out.style.width='44px';
    out.style.textAlign='right'; out.className='muted';
    rng.addEventListener('input', () => { out.textContent = rng.value; });
    rng.addEventListener('change', async () => {
      const r = await post('/api/tuning', {key:k, value:parseFloat(rng.value)});
      if (r && r.value!=null) out.textContent = r.value;
    });
    trackFocus(rng);
    row.appendChild(lab); row.appendChild(rng); row.appendChild(out);
    box.appendChild(row);
  });
  tuningBuilt = true;
}
function updateTuning(vals){
  Object.keys(vals).forEach(k => {
    if (editing==='t_'+k) return;
    const rng=document.getElementById('t_'+k), out=document.getElementById('tv_'+k);
    if (rng){ rng.value=vals[k]; if(out) out.textContent=vals[k]; }
  });
}

function flash(t){ hint.style.display='block'; hint.textContent=t; setTimeout(()=>{ if(!mode) hint.style.display='none'; }, 1200); }
function setIf(id, v){ const el=document.getElementById(id); if (el && editing!==id) el.value = v; }

async function refresh() {
  try {
    const s = await (await fetch('/api/state')).json();
    src.width = s.source.width; src.height = s.source.height;
    const st = s.status;
    document.getElementById('s_fps').textContent = st.reader_fps+' / '+st.inference_fps;
    document.getElementById('s_ms').textContent = st.infer_ms;
    document.getElementById('s_cpu').textContent = st.cpu+'% / '+st.ram+'%';
    document.getElementById('s_ppl').textContent = st.people;
    const c = s.calibration;
    const nq = (c.quad_px||[]).length;
    document.getElementById('s_cal').innerHTML = '<span class="pill '+(nq===4?'good':'bad')+'">'+nq+'/4 corners</span>';
    for (let i=0;i<4;i++){ setIf('e'+i, c.trap_edges_m[i]); setIf('a'+i, c.trap_angles_deg[i]); }
    setIf('v_camera_height_m', c.camera_height_m);
    setIf('v_cam_to_aim_m', c.cam_to_aim_m);
    setIf('v_room_width_m', c.room_width_m);
    setIf('v_room_depth_m', c.room_depth_m);
    const ce = c.closure_error_m;
    document.getElementById('closure').textContent = 'closure: ' + (ce==null ? '-' : ce.toFixed(3)+' m'+(ce<0.1?' OK':' check measurements'));
    if (!tuningBuilt && s.tuning_specs) buildTuning(s.tuning_specs);
    if (s.tuning) updateTuning(s.tuning);
    document.getElementById('zone_list').innerHTML = (c.zones||[]).map((z,i)=>
      '<div class="row"><span style="flex:1">'+z.name+' ('+ (z.polygon_px||[]).length +' pts)</span>'+
      '<span class="btn" onclick="delZone('+i+')">x</span></div>').join('');
    document.getElementById('tracks').innerHTML = (s.tracks||[]).map(t=>{
      const g = t.geo ? ('D:'+t.geo.dist_cam_m+'m ('+t.geo.x_m+','+t.geo.y_m+')'+(t.geo.zone?(' ['+t.geo.zone+']'):'')) : '<span class="muted">no geo</span>';
      return '<tr><td>ID'+t.id+'</td><td>'+t.state+'</td><td>'+g+'</td></tr>';
    }).join('');
  } catch(e) {}
}
window.delZone = async (i) => { await post('/api/zone/delete', {index:i}); };
setInterval(refresh, 1000); refresh();
</script>
</body>
</html>
"""
