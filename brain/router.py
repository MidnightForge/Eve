"""
Brain API routes — honeycomb status, cell control, live diagnostics.

GET  /brain/status              — Full brain status JSON (all cells)
GET  /brain/honeycomb           — Interactive honeycomb visualization (click-to-inspect)
POST /brain/cell/{name}/boot    — Force-boot a specific cell
POST /brain/cell/{name}/reset   — Reset a cell to dormant
GET  /brain/vllm/status         — Quick vLLM ready check
GET  /brain/cells               — List all cells
POST /brain/cells/spawn         — Birth a new dynamic cell
DELETE /brain/cells/{name}      — Retire a dynamic cell
GET  /brain/quantum_mesh        — Live quantum binding dashboard (EQCM)
GET  /brain/quantum_mesh/json   — Raw binding state JSON
GET  /brain/school/stats        — SchoolCell stats (challenges, pass rate, ORPO pairs)
GET  /brain/school/log          — SchoolCell recent challenge log
GET  /brain/reservoir/state     — ReservoirCell prediction state JSON
GET  /brain/formal/health       — FormalReasoningCell status (SymPy/Z3/Lean4)
POST /brain/formal/solve        — Direct solve endpoint (bypasses routing)
GET  /brain/ensemble/stats      — CompetitiveEnsembleCell win counts
GET  /brain/verification/stats  — VerificationCell pass/warn/fail rates
GET  /brain/speculative/state   — SpeculativeCell predictions + hit rate
GET  /brain/agot/stats          — AGoTCell graph stats (nodes, depth, accuracy)
GET  /brain/memory/hema         — HEMA running summary + surprise gate stats
GET  /brain/titans/health       — Titans neural memory module stats
"""

import json
import math
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(prefix="/brain", tags=["Honeycomb Brain"])

_brain = None

def _get_brain():
    global _brain
    if _brain is None:
        from brain import brain as _b
        _brain = _b
    return _brain


@router.get("/status")
async def brain_status():
    return JSONResponse(_get_brain().status())


@router.get("/vllm/ready")
async def vllm_ready():
    return {"ready": _get_brain().is_vllm_ready()}


@router.get("/cells")
async def list_cells():
    return JSONResponse({"cells": _get_brain().list_cells()})


@router.post("/cells/spawn")
async def spawn_cell(body: dict):
    b = _get_brain()
    result = b.spawn_cell(
        name        = body.get("name", ""),
        purpose     = body.get("purpose", ""),
        description = body.get("description", ""),
        parent_cell = body.get("parent_cell", ""),
        color       = body.get("color", ""),
    )
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


@router.delete("/cells/{name}")
async def retire_cell(name: str):
    b = _get_brain()
    result = b.retire_cell(name)
    return JSONResponse(result, status_code=200 if result.get("success") else 400)


@router.post("/cell/{name}/boot")
async def boot_cell(name: str):
    b = _get_brain()
    cell = b.cell(name)
    if not cell:
        return JSONResponse({"error": f"Cell '{name}' not found"}, status_code=404)
    await cell._boot()
    return {"cell": name, "status": cell._status.value}


@router.get("/honeycomb", response_class=HTMLResponse)
async def honeycomb_view():
    """Lotus Neural Network brain visualization. Cells as glowing petal nodes."""
    import math as _math
    status = _get_brain().status()
    cells  = status["cells"]

    RING2 = {"memory","emotion","vision","voice","creative","tools","reasoning","web"}
    RING3 = {"anima","assimilation","evolution","curiosity","preservation",
             "learning_lab","cranimem","quantum_mesh","immunity","spin","liquid_voice"}
    RING4 = {"planner","guardian","rag","agent","code_executor","multiagent",
             "formal_reason","debate","agot","ensemble","verification","titans",
             "book_editor","book_voice","audio_master","observability"}

    cell_map = {c["name"]: c for c in cells}
    rings = {2:[], 3:[], 4:[], 5:[]}
    for c in cells:
        n = c["name"]
        if n == "cortex": continue
        if n in RING2:   rings[2].append(c)
        elif n in RING3: rings[3].append(c)
        elif n in RING4: rings[4].append(c)
        else:            rings[5].append(c)

    CX, CY = 500, 500
    RADII   = {2:130, 3:240, 4:355, 5:465}
    NODE_R  = {2:18,  3:15,  4:13,  5:11}

    def polar_nodes(ring_cells, radius, angle_offset=0):
        n = len(ring_cells)
        result = []
        for i, c in enumerate(ring_cells):
            angle = angle_offset + (2 * _math.pi * i / n) - _math.pi/2
            x = CX + radius * _math.cos(angle)
            y = CY + radius * _math.sin(angle)
            result.append((x, y, c))
        return result

    nodes_js_list = []
    cortex = cell_map.get("cortex")
    nodes_js_list.append({"x": CX, "y": CY, "name": "cortex",
                          "color": "#f59e0b",
                          "status": cortex.get("status","active") if cortex else "active",
                          "r": 28})
    for ring_id, ring_cells in rings.items():
        if not ring_cells: continue
        offset = _math.pi / max(len(ring_cells),1) * 0.3 * ring_id
        for x, y, c in polar_nodes(ring_cells, RADII[ring_id], offset):
            nodes_js_list.append({"x": round(x,1), "y": round(y,1),
                                  "name": c["name"], "color": c.get("color","#7c3aed"),
                                  "status": c.get("status","dormant"),
                                  "r": NODE_R[ring_id]})

    cells_js = json.dumps([{
        "name":            c.get("name",""),
        "description":     c.get("description",""),
        "status":          c.get("status","dormant"),
        "system_tier":     c.get("system_tier","online"),
        "hardware_req":    c.get("hardware_req",""),
        "build_notes":     c.get("build_notes",""),
        "framework_layer": c.get("framework_layer",""),
        "calls":           c.get("calls",0),
        "errors":          c.get("errors",0),
        "last_ms":         c.get("last_ms",0),
        "uptime_s":        c.get("uptime_s",0),
        "color":           c.get("color","#7c3aed"),
    } for c in cells])
    nodes_js = json.dumps(nodes_js_list)

    vllm_ready  = status.get("vllm_ready", False)
    mode_label  = "CORTEX MODE" if status.get("cortex_mode") else "QWEN3-14B"
    total_cells = len(cells)
    online_cnt  = sum(1 for c in cells if c.get("system_tier") == "online")
    active_cnt  = sum(1 for c in cells if c.get("status") == "active")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Eve — Lotus Neural Lattice</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#040812; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; overflow:hidden; width:100vw; height:100vh; }}
canvas {{ position:absolute; top:0; left:0; pointer-events:none; }}
#svg-layer {{ position:absolute; }}
#hud {{ position:absolute; top:14px; left:14px; display:flex; flex-direction:column; gap:6px; pointer-events:none; z-index:10; }}
.hud-title {{ font-size:0.6rem; letter-spacing:4px; color:#4fc3f7; text-transform:uppercase; opacity:0.9; }}
.hud-stats {{ display:flex; gap:12px; flex-wrap:wrap; }}
.hud-stat {{ font-size:0.6rem; letter-spacing:1px; color:#94a3b8; }}
.hud-stat span {{ color:#e2e8f0; font-weight:600; }}
.hud-mode {{ font-size:0.55rem; letter-spacing:2px; color:{'#a78bfa' if status.get('cortex_mode') else '#22c55e'}; border:1px solid currentColor; padding:2px 8px; border-radius:8px; display:inline-block; opacity:0.8; }}
#inspector {{ position:absolute; right:0; top:0; bottom:0; width:300px; background:rgba(4,8,18,0.96); border-left:1px solid rgba(79,195,247,0.15); padding:20px 16px; overflow-y:auto; transform:translateX(100%); transition:transform 0.3s cubic-bezier(0.4,0,0.2,1); z-index:20; }}
#inspector.open {{ transform:translateX(0); }}
.ins-close {{ position:absolute; top:12px; right:12px; background:none; border:none; color:#94a3b8; font-size:1.2rem; cursor:pointer; padding:4px 8px; }}
.ins-close:hover {{ color:#e2e8f0; }}
.ins-name {{ font-size:0.9rem; font-weight:700; margin-bottom:4px; color:#f1f5f9; text-transform:uppercase; letter-spacing:2px; }}
.ins-desc {{ font-size:0.7rem; color:#94a3b8; margin-bottom:14px; line-height:1.5; }}
.ins-row {{ display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.65rem; }}
.ins-label {{ color:#64748b; letter-spacing:1px; text-transform:uppercase; }}
.ins-val {{ color:#e2e8f0; font-weight:500; text-align:right; max-width:170px; }}
.ins-section {{ margin-top:14px; font-size:0.6rem; letter-spacing:2px; color:#4fc3f7; text-transform:uppercase; margin-bottom:6px; }}
.ins-notes {{ font-size:0.65rem; color:#94a3b8; line-height:1.6; background:rgba(79,195,247,0.04); padding:8px; border-radius:6px; border:1px solid rgba(79,195,247,0.08); }}
.status-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; vertical-align:middle; }}
#tooltip {{ position:absolute; pointer-events:none; z-index:30; background:rgba(4,8,18,0.92); border:1px solid rgba(79,195,247,0.3); padding:6px 10px; border-radius:8px; font-size:0.65rem; color:#e2e8f0; white-space:nowrap; display:none; box-shadow:0 0 16px rgba(79,195,247,0.15); }}
</style>
</head>
<body>
<canvas id="bg"></canvas>
<svg id="svg-layer" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"></svg>
<div id="hud">
  <div class="hud-title">Eve &#x2014; Lotus Neural Lattice</div>
  <div class="hud-stats">
    <div class="hud-stat">Cells <span>{total_cells}</span></div>
    <div class="hud-stat">Online <span>{online_cnt}</span></div>
    <div class="hud-stat">Active <span id="active-cnt">{active_cnt}</span></div>
  </div>
  <div class="hud-mode">{mode_label}</div>
</div>
<div id="inspector">
  <button class="ins-close" onclick="closeInspector()">&#x2715;</button>
  <div class="ins-name" id="ins-name">&#8212;</div>
  <div class="ins-desc" id="ins-desc"></div>
  <div id="ins-body"></div>
</div>
<div id="tooltip"></div>
<script>
const CELLS={cells_js};
const NODES={nodes_js};
const CX=500,CY=500;
const SVG_NS='http://www.w3.org/2000/svg';
let W,H,particles=[],svgScale=1,svgOffX=0,svgOffY=0,pulses=[];

// ── Canvas background ──────────────────────────────────────────────────────
const bgC=document.getElementById('bg'),bgX=bgC.getContext('2d');
function resize(){{
  W=bgC.width=window.innerWidth; H=bgC.height=window.innerHeight;
  const vw=Math.min(W,H)*0.97; svgScale=vw/1000;
  svgOffX=(W-1000*svgScale)/2; svgOffY=(H-1000*svgScale)/2;
  const svg=document.getElementById('svg-layer');
  svg.style.cssText=`width:${{1000*svgScale}}px;height:${{1000*svgScale}}px;left:${{svgOffX}}px;top:${{svgOffY}}px`;
  initP();
}}
function initP(){{
  particles=[];
  for(let i=0;i<Math.floor(W*H/6000);i++)
    particles.push({{x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.2+0.2,
      a:Math.random(),da:(Math.random()-0.5)*0.004,dx:(Math.random()-0.5)*0.12,
      dy:(Math.random()-0.5)*0.12,h:Math.random()>0.6?200+Math.random()*40:260+Math.random()*40}});
}}
function drawBg(){{
  bgX.clearRect(0,0,W,H);
  const g=bgX.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.7);
  g.addColorStop(0,'rgba(6,10,28,1)'); g.addColorStop(0.5,'rgba(4,8,18,1)'); g.addColorStop(1,'rgba(2,4,10,1)');
  bgX.fillStyle=g; bgX.fillRect(0,0,W,H);
  particles.forEach(p=>{{
    p.x+=p.dx; p.y+=p.dy; p.a+=p.da;
    if(p.a>0.85)p.da=-Math.abs(p.da); if(p.a<0.05)p.da=Math.abs(p.da);
    if(p.x<0)p.x=W; if(p.x>W)p.x=0; if(p.y<0)p.y=H; if(p.y>H)p.y=0;
    bgX.beginPath(); bgX.arc(p.x,p.y,p.r,0,Math.PI*2);
    bgX.fillStyle=`hsla(${{p.h}},80%,70%,${{p.a*0.5}})`; bgX.fill();
  }});
  requestAnimationFrame(drawBg);
}}

// ── SVG helpers ────────────────────────────────────────────────────────────
function el(tag,attrs){{
  const e=document.createElementNS(SVG_NS,tag);
  for(const[k,v] of Object.entries(attrs))e.setAttribute(k,v);
  return e;
}}
function rgb(hex){{
  return parseInt(hex.slice(1,3),16)+','+parseInt(hex.slice(3,5),16)+','+parseInt(hex.slice(5,7),16);
}}

function buildSVG(){{
  const svg=document.getElementById('svg-layer');
  svg.innerHTML='';
  // Defs
  const defs=el('defs',{{}});
  // Glow filter
  const fg=el('filter',{{id:'glow',x:'-60%',y:'-60%',width:'220%',height:'220%'}});
  const fb=el('feGaussianBlur',{{in:'SourceGraphic',stdDeviation:'5',result:'b'}});
  const fm=el('feMerge',{{}}); fm.appendChild(el('feMergeNode',{{in:'b'}})); fm.appendChild(el('feMergeNode',{{in:'SourceGraphic'}}));
  fg.appendChild(fb); fg.appendChild(fm); defs.appendChild(fg);
  // Strong glow for cortex
  const fs=el('filter',{{id:'glow2',x:'-100%',y:'-100%',width:'300%',height:'300%'}});
  const fb2=el('feGaussianBlur',{{in:'SourceGraphic',stdDeviation:'12',result:'b'}});
  const fm2=el('feMerge',{{}}); fm2.appendChild(el('feMergeNode',{{in:'b'}})); fm2.appendChild(el('feMergeNode',{{in:'SourceGraphic'}}));
  fs.appendChild(fb2); fs.appendChild(fm2); defs.appendChild(fs);
  // Cortex gradient
  const cg=el('radialGradient',{{id:'cg',cx:'50%',cy:'50%',r:'50%'}});
  cg.appendChild(el('stop',{{offset:'0%','stop-color':'#fff8e1'}}));
  cg.appendChild(el('stop',{{offset:'45%','stop-color':'#f59e0b'}}));
  cg.appendChild(el('stop',{{offset:'100%','stop-color':'#b45309','stop-opacity':'0.85'}}));
  defs.appendChild(cg);
  svg.appendChild(defs);

  // Decorative petal shapes (very subtle)
  const pl=el('g',{{opacity:'0.06'}});
  [130,240,355,465].forEach((r,ri)=>{{
    const cnt=[8,11,13,19][ri];
    for(let i=0;i<cnt;i++){{
      const a=(2*Math.PI*i/cnt)-Math.PI/2+(ri*0.28);
      const tx=CX+r*Math.cos(a), ty=CY+r*Math.sin(a);
      const bx=CX+r*0.38*Math.cos(a), by=CY+r*0.38*Math.sin(a);
      const sp=r*0.22;
      const la=a-Math.PI/2, ra2=a+Math.PI/2;
      const c1x=bx+sp*Math.cos(la)+(tx-bx)*0.45, c1y=by+sp*Math.sin(la)+(ty-by)*0.45;
      const c2x=bx+sp*Math.cos(ra2)+(tx-bx)*0.45, c2y=by+sp*Math.sin(ra2)+(ty-by)*0.45;
      pl.appendChild(el('path',{{
        d:`M${{bx.toFixed(1)}},${{by.toFixed(1)}} C${{c1x.toFixed(1)}},${{c1y.toFixed(1)}} ${{tx.toFixed(1)}},${{ty.toFixed(1)}} ${{tx.toFixed(1)}},${{ty.toFixed(1)}} C${{tx.toFixed(1)}},${{ty.toFixed(1)}} ${{c2x.toFixed(1)}},${{c2y.toFixed(1)}} ${{bx.toFixed(1)}},${{by.toFixed(1)}} Z`,
        fill:'#4fc3f7',stroke:'#4fc3f7','stroke-width':'0.4'
      }}));
    }}
  }});
  svg.appendChild(pl);

  // Orbit rings
  [130,240,355,465].forEach(r=>{{
    svg.appendChild(el('circle',{{cx:CX,cy:CY,r:r,fill:'none',stroke:'rgba(79,195,247,0.055)','stroke-width':'1','stroke-dasharray':'4 9'}}));
  }});

  // Connection lines
  const ll=el('g',{{id:'conn'}});
  NODES.forEach(n=>{{
    if(n.name==='cortex')return;
    const r2=rgb(n.color);
    const active=n.status==='active';
    ll.appendChild(el('line',{{
      id:'line-'+n.name, x1:CX,y1:CY,x2:n.x,y2:n.y,
      stroke:active?`rgba(${{r2}},0.5)`:`rgba(${{r2}},0.1)`,
      'stroke-width':active?'1.5':'0.6'
    }}));
  }});
  svg.appendChild(ll);

  // Pulse layer
  svg.appendChild(el('g',{{id:'pulses'}}));

  // Nodes
  const nl=el('g',{{id:'nodes'}});
  NODES.forEach(n=>{{
    const g=el('g',{{id:'node-'+n.name,style:'cursor:pointer'}});
    g.setAttribute('onclick',`showCell('${{n.name}}')`);
    const r2=rgb(n.color), r=n.r, isCortex=n.name==='cortex';
    const active=n.status==='active', dormant=n.status==='dormant';
    if(!dormant){{
      const halo=el('circle',{{cx:n.x,cy:n.y,r:r+9,fill:`rgba(${{r2}},${{active||isCortex?0.14:0.05}})`,
        stroke:`rgba(${{r2}},${{active||isCortex?0.45:0.18}})`, 'stroke-width':'1',
        ...(active||isCortex?{{'filter':'url(#glow)'}}:{{}})}});
      g.appendChild(halo);
    }}
    g.appendChild(el('circle',{{cx:n.x,cy:n.y,r:r,
      fill:isCortex?'url(#cg)':dormant?`rgba(${{r2}},0.07)`:`rgba(${{r2}},0.28)`,
      stroke:dormant?`rgba(${{r2}},0.2)`:n.color,'stroke-width':isCortex?'2.5':active?'1.8':'1.0',
      filter:isCortex?'url(#glow2)':active?'url(#glow)':'none'}}));
    if(active&&!isCortex){{
      const pr=el('circle',{{cx:n.x,cy:n.y,r:r,fill:'none',stroke:n.color,'stroke-width':'1.5',opacity:'0.6'}});
      pr.innerHTML=`<animate attributeName="r" values="${{r}};${{r+14}};${{r}}" dur="2.8s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.6;0;0.6" dur="2.8s" repeatCount="indefinite"/>`;
      g.appendChild(pr);
    }}
    if(isCortex){{
      const rr=el('circle',{{cx:n.x,cy:n.y,r:r+16,fill:'none',stroke:'rgba(245,158,11,0.3)','stroke-width':'1.5','stroke-dasharray':'5 4'}});
      rr.innerHTML=`<animateTransform attributeName="transform" type="rotate" from="0 ${{n.x}} ${{n.y}}" to="360 ${{n.x}} ${{n.y}}" dur="14s" repeatCount="indefinite"/>`;
      g.appendChild(rr);
    }}
    const label=n.name.toUpperCase().replace(/_/g,' ');
    const short=label.length>9?label.slice(0,8)+'\\u2026':label;
    const txt=el('text',{{x:n.x,y:n.y+1,'text-anchor':'middle','dominant-baseline':'middle',
      'font-size':isCortex?9:r>=18?8:6.5,'font-weight':'700',
      'font-family':'Segoe UI,system-ui,sans-serif',
      fill:dormant?'#475569':'#f1f5f9','pointer-events':'none'}});
    txt.textContent=isCortex?'CORTEX':short;
    g.appendChild(txt);
    g.addEventListener('mouseenter',e=>tip(e,n));
    g.addEventListener('mouseleave',()=>{{document.getElementById('tooltip').style.display='none';}});
    nl.appendChild(g);
  }});
  svg.appendChild(nl);
}}

// ── Tooltip ────────────────────────────────────────────────────────────────
function tip(e,n){{
  const cd=CELLS.find(c=>c.name===n.name); if(!cd)return;
  const t=document.getElementById('tooltip');
  t.innerHTML=`<strong>${{n.name.toUpperCase().replace(/_/g,' ')}}</strong> &nbsp;&middot;&nbsp; ${{cd.calls}} calls &nbsp;&middot;&nbsp; ${{cd.last_ms}}ms`;
  t.style.cssText=`display:block;left:${{e.clientX+14}}px;top:${{e.clientY-8}}px`;
}}

// ── Inspector ──────────────────────────────────────────────────────────────
function showCell(name){{
  const cd=CELLS.find(c=>c.name===name); if(!cd)return;
  const SC={{active:'#22c55e',booting:'#f59e0b',busy:'#a78bfa',degraded:'#f97316',offline:'#ef4444',dormant:'#4b5563'}};
  document.getElementById('ins-name').textContent=name.toUpperCase().replace(/_/g,' ');
  document.getElementById('ins-desc').textContent=cd.description||'\\u2014';
  const sc=SC[cd.status]||'#4b5563';
  const rows=[
    ['Status',`<span class="status-dot" style="background:${{sc}}"></span>${{cd.status}}`],
    ['Tier',cd.system_tier||'\\u2014'],['Calls',cd.calls],['Errors',cd.errors],
    ['Last',cd.last_ms?cd.last_ms+'ms':'\\u2014'],
    ['Uptime',cd.uptime_s?Math.floor(cd.uptime_s/60)+'min':'\\u2014'],
    ['Layer',cd.framework_layer||'\\u2014'],['Hardware',cd.hardware_req||'\\u2014'],
  ];
  let h=rows.map(([l,v])=>`<div class="ins-row"><span class="ins-label">${{l}}</span><span class="ins-val">${{v}}</span></div>`).join('');
  if(cd.build_notes) h+=`<div class="ins-section">Build Notes</div><div class="ins-notes">${{cd.build_notes}}</div>`;
  document.getElementById('ins-body').innerHTML=h;
  document.getElementById('inspector').classList.add('open');
}}
function closeInspector(){{ document.getElementById('inspector').classList.remove('open'); }}

// ── Signal pulses ──────────────────────────────────────────────────────────
function firePulse(name){{
  const n=NODES.find(x=>x.name===name); if(!n||n.name==='cortex')return;
  pulses.push({{x1:CX,y1:CY,x2:n.x,y2:n.y,t:0,color:n.color}});
}}
function animPulses(){{
  const layer=document.getElementById('pulses'); if(!layer)return;
  layer.innerHTML=''; pulses=pulses.filter(p=>p.t<1);
  pulses.forEach(p=>{{
    p.t+=0.022;
    const x=p.x1+(p.x2-p.x1)*p.t, y=p.y1+(p.y2-p.y1)*p.t;
    const dot=el('circle',{{cx:x,cy:y,r:3.5,fill:p.color,filter:'url(#glow)',opacity:(1-p.t)*0.9}});
    layer.appendChild(dot);
    for(let i=1;i<=3;i++){{
      const bt=Math.max(0,p.t-i*0.04);
      const tx=p.x1+(p.x2-p.x1)*bt, ty=p.y1+(p.y2-p.y1)*bt;
      layer.appendChild(el('circle',{{cx:tx,cy:ty,r:2-i*0.35,fill:p.color,opacity:(1-p.t)*0.28/i}}));
    }}
  }});
  requestAnimationFrame(animPulses);
}}

// ── Live refresh ───────────────────────────────────────────────────────────
function refresh(){{
  fetch('/brain/status').then(r=>r.json()).then(data=>{{
    if(!data.cells)return;
    let ac=0;
    data.cells.forEach(c=>{{
      const old=CELLS.find(x=>x.name===c.name);
      if(old&&c.calls>old.calls)firePulse(c.name);
      if(old){{old.calls=c.calls;old.status=c.status;old.last_ms=c.last_ms;}}
      if(c.status==='active')ac++;
      const ln=document.getElementById('line-'+c.name);
      if(ln){{
        const r2=rgb(c.color||'#7c3aed');
        ln.setAttribute('stroke',c.status==='active'?`rgba(${{r2}},0.55)`:`rgba(${{r2}},0.1)`);
        ln.setAttribute('stroke-width',c.status==='active'?'1.8':'0.6');
      }}
    }});
    const el2=document.getElementById('active-cnt'); if(el2)el2.textContent=ac;
  }}).catch(()=>{{}});
}}

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('resize',()=>{{resize();buildSVG();}});
resize(); buildSVG(); drawBg(); animPulses(); setInterval(refresh,8000);
</script>
</body>
</html>"""
    return html


# ── Quantum Cell Mesh endpoints ────────────────────────────────────────────────

@router.get("/quantum_mesh/json")
async def quantum_mesh_json():
    """Raw binding state JSON from the EQCM."""
    from brain.cells.quantum_mesh import get_mesh_binding
    return JSONResponse(get_mesh_binding())


@router.get("/quantum_mesh", response_class=HTMLResponse)
async def quantum_mesh_dashboard():
    """
    Live quantum binding dashboard.
    Shows Hopfield resonance, GWT workspace winner, and quantum circuit state
    for all brain cells in real time.
    """
    import time as _time
    from brain.cells.quantum_mesh import get_mesh_binding

    binding = get_mesh_binding()

    if not binding:
        return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="3">
<style>body{{background:#060408;color:#06b6d4;font-family:'Segoe UI',sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;font-size:1.2rem;}}</style>
</head><body>⚛ Quantum Cell Mesh initialising — first pulse in ~10 s…</body></html>""")

    pulse     = binding.get("pulse", 0)
    n_cells   = binding.get("n_cells", 0)
    ts        = binding.get("timestamp", 0)
    ago       = round(_time.time() - ts, 1)

    hopfield  = binding.get("hopfield", {})
    workspace = binding.get("workspace", {})
    quantum   = binding.get("quantum", {})

    resonance    = hopfield.get("resonance", {})
    query_cell   = hopfield.get("query_cell", "")
    n_patterns   = hopfield.get("n_patterns", 0)

    winner       = workspace.get("winner", "")
    scores       = workspace.get("scores", {})

    coherence    = quantum.get("coherence", 0)
    measurements = quantum.get("measurements", [])
    entanglement = quantum.get("entanglement", [])
    superpos     = quantum.get("superposition", [])
    qnames       = quantum.get("cell_names", [])

    # ── Resonance bars ─────────────────────────────────────────────────────────
    sorted_res = sorted(resonance.items(), key=lambda x: x[1], reverse=True)
    res_rows = ""
    for cname, score in sorted_res[:12]:
        pct = round(score * 100, 1)
        bar_w = max(1, int(pct * 2))
        is_winner = cname == winner
        color = "#f59e0b" if is_winner else "#06b6d4"
        res_rows += (
            f'<tr>'
            f'<td style="color:{color};padding:3px 8px;font-size:.72rem;">'
            f'{"⚡ " if is_winner else ""}{cname}</td>'
            f'<td style="padding:3px 8px;">'
            f'<div style="background:rgba(6,182,212,.15);border-radius:3px;height:10px;width:200px;">'
            f'<div style="background:{color};height:10px;width:{bar_w}px;border-radius:3px;'
            f'opacity:0.8;"></div></div></td>'
            f'<td style="color:#94a3b8;font-size:.7rem;padding:3px 8px;">{pct}%</td>'
            f'</tr>'
        )

    # ── GWT score bars ─────────────────────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score = max((v for _,v in sorted_scores), default=1)
    gwt_rows = ""
    for cname, score in sorted_scores[:10]:
        pct = round(score / max_score * 100, 1)
        bar_w = max(1, int(pct * 1.6))
        is_w = cname == winner
        color = "#f59e0b" if is_w else "#a78bfa"
        gwt_rows += (
            f'<tr>'
            f'<td style="color:{color};padding:3px 8px;font-size:.72rem;">'
            f'{"✦ " if is_w else ""}{cname}</td>'
            f'<td style="padding:3px 8px;">'
            f'<div style="background:rgba(167,139,250,.12);border-radius:3px;height:10px;width:160px;">'
            f'<div style="background:{color};height:10px;width:{bar_w}px;border-radius:3px;'
            f'opacity:0.8;"></div></div></td>'
            f'<td style="color:#94a3b8;font-size:.7rem;padding:3px 8px;">{round(score,3)}</td>'
            f'</tr>'
        )

    # ── Quantum qubit rows ─────────────────────────────────────────────────────
    q_rows = ""
    for i, (qn, m, e, p) in enumerate(
        zip(qnames, measurements, entanglement, superpos)
    ):
        bit_color = "#22c55e" if m == 1 else "#4b5563"
        ent_pct   = round(e * 100, 1)
        sup_pct   = round(p * 100, 1)
        q_rows += (
            f'<tr>'
            f'<td style="color:#94a3b8;font-size:.7rem;padding:2px 8px;">{i}</td>'
            f'<td style="font-size:.7rem;padding:2px 8px;">{qn}</td>'
            f'<td style="color:{bit_color};font-size:.8rem;font-weight:700;'
            f'padding:2px 8px;text-align:center;">|{m}⟩</td>'
            f'<td style="color:#06b6d4;font-size:.7rem;padding:2px 8px;">{sup_pct}%</td>'
            f'<td style="color:#a78bfa;font-size:.7rem;padding:2px 8px;">{ent_pct}%</td>'
            f'</tr>'
        )

    coherence_color = "#22c55e" if coherence > 0.6 else (
                       "#f59e0b" if coherence > 0.3 else "#ef4444")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>⚛ Eve Quantum Cell Mesh</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    background:#060408; color:#f1f5f9;
    font-family:'Segoe UI',system-ui,sans-serif;
    padding:16px; overflow-x:hidden;
  }}
  h2 {{ color:#06b6d4; font-size:.8rem; letter-spacing:3px;
        text-transform:uppercase; margin-bottom:4px; }}
  .header {{ display:flex; align-items:center; gap:14px; margin-bottom:12px; flex-wrap:wrap; }}
  .pill {{
    display:inline-block; padding:3px 12px; border-radius:12px;
    font-size:.65rem; letter-spacing:1px; font-weight:700;
  }}
  .pill-cyan {{ background:rgba(6,182,212,.15); border:1px solid rgba(6,182,212,.4); color:#06b6d4; }}
  .pill-green {{ background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#22c55e; }}
  .pulse-dot {{
    width:10px; height:10px; border-radius:50%; background:#06b6d4;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:.3; transform:scale(1.5); }}
  }}
  .stats {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:16px; }}
  .stat {{ font-size:.65rem; color:#94a3b8; }}
  .stat b {{ color:#f1f5f9; }}
  .grid {{
    display:grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap:14px;
    margin-bottom:14px;
  }}
  .panel {{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);
    border-radius:10px;
    padding:14px;
  }}
  .panel-title {{
    font-size:.6rem; letter-spacing:2px; text-transform:uppercase;
    color:#94a3b8; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,.07);
    padding-bottom:6px;
  }}
  table {{ border-collapse:collapse; width:100%; }}
  td {{ vertical-align:middle; }}
  .coherence-val {{
    font-size:2.2rem; font-weight:700; color:{coherence_color};
    margin-bottom:4px;
  }}
  .q-circuit {{
    font-family:monospace; font-size:.7rem; color:#a78bfa;
    background:rgba(167,139,250,.08); border-radius:6px;
    padding:6px 10px; margin-bottom:10px; letter-spacing:.5px;
  }}
  .winner-box {{
    background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.3);
    border-radius:8px; padding:10px 14px; text-align:center; margin-bottom:10px;
  }}
  .winner-name {{ color:#f59e0b; font-size:1.1rem; font-weight:700; letter-spacing:2px; }}
  .winner-lbl  {{ color:#94a3b8; font-size:.6rem; letter-spacing:1px; margin-bottom:4px; }}
  .foot {{ font-size:.6rem; color:#4b5563; text-align:center; margin-top:14px; }}
</style>
</head>
<body>

<div class="header">
  <div class="pulse-dot"></div>
  <h2>⚛ Eve Quantum Cell Mesh — EQCM</h2>
  <span class="pill pill-cyan">PULSE #{pulse}</span>
  <span class="pill pill-green">{n_cells} CELLS BOUND</span>
  <span style="font-size:.6rem;color:#4b5563;">{ago}s ago</span>
</div>

<div class="stats">
  <div class="stat">Hopfield patterns: <b>{n_patterns}</b></div>
  <div class="stat">Query cell: <b style="color:#06b6d4">{query_cell}</b></div>
  <div class="stat">GWT winner: <b style="color:#f59e0b">{winner}</b></div>
  <div class="stat">Coherence: <b style="color:{coherence_color}">{coherence:.4f}</b></div>
  <div class="stat">Auto-refresh: <b>10 s</b></div>
</div>

<div class="grid">

  <!-- GWT Global Workspace -->
  <div class="panel">
    <div class="panel-title">Layer 2 — Global Workspace Bus (GNWT)</div>
    <div class="winner-box">
      <div class="winner-lbl">IGNITION WINNER — BROADCASTING TO ALL CELLS</div>
      <div class="winner-name">⚡ {winner.upper()}</div>
    </div>
    <table>{gwt_rows}</table>
  </div>

  <!-- Hopfield Resonance -->
  <div class="panel">
    <div class="panel-title">Layer 1 — Hopfield Fabric Resonance (query: {query_cell})</div>
    <table>{res_rows}</table>
  </div>

  <!-- Quantum Circuit -->
  <div class="panel" style="grid-column:1/-1;">
    <div class="panel-title">Layer 3 — Quantum Binding Circuit</div>
    <div class="q-circuit">H^n → CNOT-ring → RY(θᵢ) → Measure  [{n_cells} qubits]</div>
    <div style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;">
      <div>
        <div style="color:#94a3b8;font-size:.6rem;letter-spacing:1px;margin-bottom:6px;">
          GLOBAL PHASE COHERENCE
        </div>
        <div class="coherence-val">{coherence:.4f}</div>
        <div style="font-size:.65rem;color:#94a3b8;">
          {"High coherence — cells are entangled" if coherence > 0.6 else
           ("Partial coherence — partial binding" if coherence > 0.3 else
            "Low coherence — cells are decoherent")}
        </div>
      </div>
      <div style="flex:1;min-width:300px;overflow-x:auto;">
        <table>
          <tr>
            <th style="color:#4b5563;font-size:.6rem;font-weight:400;padding:2px 8px;">QUBIT</th>
            <th style="color:#4b5563;font-size:.6rem;font-weight:400;padding:2px 8px;">CELL</th>
            <th style="color:#4b5563;font-size:.6rem;font-weight:400;padding:2px 8px;">MEASURE</th>
            <th style="color:#06b6d4;font-size:.6rem;font-weight:400;padding:2px 8px;">P(|1⟩)</th>
            <th style="color:#a78bfa;font-size:.6rem;font-weight:400;padding:2px 8px;">ENTANGLE</th>
          </tr>
          {q_rows}
        </table>
      </div>
    </div>
  </div>

</div>

<div class="foot">
  Eve Quantum Cell Mesh (EQCM) · Ramsauer 2020 (Hopfield) · Dehaene GNWT ·
  H+CNOT+RY quantum simulation · Upgrades to NVIDIA CUDA-Q on Blackwell arrival
</div>

</body>
</html>"""
    return html


# ── School Cell endpoints ───────────────────────────────────────────────────────

@router.get("/school/stats")
async def school_stats():
    """SchoolCell aggregate statistics."""
    from brain.cells.school import get_school_stats
    return JSONResponse(get_school_stats())


@router.get("/school/log")
async def school_log(limit: int = 20):
    """Recent SchoolCell challenge log (most recent first)."""
    from brain.cells.school import get_school_log
    log = get_school_log()
    return JSONResponse({"log": list(reversed(log))[:limit], "total": len(log)})


# ── Reservoir Cell endpoints ────────────────────────────────────────────────────

@router.get("/reservoir/state")
async def reservoir_state():
    """ReservoirCell current prediction state."""
    from brain.cells.reservoir import get_reservoir_prediction
    preds = get_reservoir_prediction()
    b = _get_brain()
    rc = b._cells.get("reservoir") if b else None
    return JSONResponse({
        "predictions":    preds,
        "architecture":   "DeepResESN-3L + NG-RC + OnlineRLS",
        "esn_trained":    rc._esn._trained    if (rc and rc._esn)  else False,
        "ngrc_trained":   rc._ngrc._trained   if (rc and rc._ngrc) else False,
        "rls_updates":    rc._rls._updates    if (rc and rc._rls)  else 0,
        "step_count":     rc._step_count      if rc else 0,
        "offline_trains": rc._train_count     if rc else 0,
        "status":         rc._status.value    if rc else "unknown",
    })


# ── Formal Reasoning endpoints ──────────────────────────────────────────────────

@router.get("/formal/health")
async def formal_health():
    """FormalReasoningCell status — which backends are available."""
    b = _get_brain()
    fc = b._cells.get("formal_reason") if b else None
    if not fc:
        return JSONResponse({"error": "FormalReasoningCell not registered"}, status_code=404)
    return JSONResponse(fc.health())


@router.post("/formal/solve")
async def formal_solve(body: dict):
    """
    Direct math solve endpoint — bypasses Cortex routing.
    Body: {"query": "integrate sin(x) from 0 to pi"}
    Returns: {formal_result, explanation, code, category}
    """
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    b = _get_brain()
    fc = b._cells.get("formal_reason") if b else None
    if not fc:
        return JSONResponse({"error": "FormalReasoningCell not available"}, status_code=503)
    from brain.base_cell import CellContext
    ctx = CellContext(message=query, user_id=0)
    result = await fc.process(ctx)
    return JSONResponse(result if isinstance(result, dict) else {"result": str(result)})


# ── Ensemble / Verification / Speculative endpoints ────────────────────────────

@router.get("/ensemble/stats")
async def ensemble_stats():
    b = _get_brain()
    ec = b._cells.get("ensemble") if b else None
    if not ec:
        return JSONResponse({"error": "CompetitiveEnsembleCell not registered"}, status_code=404)
    return JSONResponse(ec.health())


@router.get("/verification/stats")
async def verification_stats():
    b = _get_brain()
    vc = b._cells.get("verification") if b else None
    if not vc:
        return JSONResponse({"error": "VerificationCell not registered"}, status_code=404)
    return JSONResponse(vc.health())


@router.get("/speculative/state")
async def speculative_state():
    b = _get_brain()
    sc = b._cells.get("speculative") if b else None
    if not sc:
        return JSONResponse({"error": "SpeculativeCell not registered"}, status_code=404)
    return JSONResponse({
        **sc.health(),
        "last_predictions": sc._last_prediction,
    })


@router.get("/agot/stats")
async def agot_stats():
    b = _get_brain()
    ac = b._cells.get("agot") if b else None
    if not ac:
        return JSONResponse({"error": "AGoTCell not registered"}, status_code=404)
    return JSONResponse(ac.health())


@router.get("/memory/hema")
async def memory_hema():
    b = _get_brain()
    mc = b._cells.get("memory") if b else None
    if not mc:
        return JSONResponse({"error": "MemoryCell not found"}, status_code=404)
    return JSONResponse({
        **mc.health(),
        "hema_summary": mc._hema_summary,
        "hema_recent_count": len(mc._hema_recent),
        "vocab_size": len(mc._hema_vocab),
    })


@router.get("/titans/health")
async def titans_health():
    b = _get_brain()
    tc = b._cells.get("titans") if b else None
    if not tc:
        return JSONResponse({"error": "TitansCell not registered"}, status_code=404)
    return JSONResponse(tc.health())



# ══════════════════════════════════════════════════════════════════════════════
# Audiobook Production System — BookEditorCell, BookVoiceCell, AudioMasterCell
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/book/edit")
async def book_edit(body: dict):
    b = _get_brain()
    cell = b._cells.get("book_editor") if b else None
    if not cell:
        return JSONResponse({"error": "BookEditingPipelineCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    file_path     = body.get("file_path", "")
    book_title    = body.get("book_title", "")
    chapter_index = body.get("chapter_index", None)
    if not file_path or not book_title:
        return JSONResponse({"error": "file_path and book_title are required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.edit_chapter, file_path, book_title, chapter_index
    )
    return JSONResponse(result)


@router.get("/book/continuity/{book_title}")
async def book_continuity(book_title: str):
    b = _get_brain()
    cell = b._cells.get("book_editor") if b else None
    if not cell:
        return JSONResponse({"error": "BookEditingPipelineCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    return JSONResponse(cell.get_continuity(book_title))


@router.post("/book/edit_all")
async def book_edit_all(body: dict):
    b = _get_brain()
    cell = b._cells.get("book_editor") if b else None
    if not cell:
        return JSONResponse({"error": "BookEditingPipelineCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    file_path  = body.get("file_path", "")
    book_title = body.get("book_title", "")
    if not file_path or not book_title:
        return JSONResponse({"error": "file_path and book_title are required"}, status_code=400)
    results = list(cell.edit_all_generator(file_path, book_title))
    return JSONResponse({"results": results})


@router.post("/book/voice")
async def book_voice(body: dict):
    b = _get_brain()
    cell = b._cells.get("book_voice") if b else None
    if not cell:
        return JSONResponse({"error": "BookVoiceCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    text       = body.get("text", "")
    book_title = body.get("book_title", "default")
    character  = body.get("character", None)
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    import asyncio
    from fastapi.responses import Response
    wav = await asyncio.get_event_loop().run_in_executor(None, cell.speak, text, book_title, character)
    if wav:
        return Response(content=wav, media_type="audio/wav")
    return JSONResponse({"error": "TTS generation failed"}, status_code=500)


@router.post("/book/voice_chapter")
async def book_voice_chapter(body: dict):
    b = _get_brain()
    cell = b._cells.get("book_voice") if b else None
    if not cell:
        return JSONResponse({"error": "BookVoiceCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    chapter_text = body.get("chapter_text", "")
    book_title   = body.get("book_title", "default")
    output_path  = body.get("output_path", None)
    if not chapter_text:
        return JSONResponse({"error": "chapter_text is required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.generate_chapter_audio, chapter_text, book_title
    )
    if output_path and result.get("wav_bytes"):
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(result["wav_bytes"])
    return JSONResponse({
        "characters_used":     result.get("characters_used", []),
        "segments_processed":  result.get("segments_processed", 0),
        "duration_estimate_s": result.get("duration_estimate_s", 0),
        "wav_saved":           output_path,
    })


@router.get("/book/characters/{book_title}")
async def book_characters(book_title: str):
    b = _get_brain()
    cell = b._cells.get("book_voice") if b else None
    if not cell:
        return JSONResponse({"error": "BookVoiceCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    return JSONResponse({"book_title": book_title, "characters": cell.get_characters(book_title)})


@router.post("/book/master")
async def book_master(body: dict):
    b = _get_brain()
    cell = b._cells.get("audio_master") if b else None
    if not cell:
        return JSONResponse({"error": "AudioMasteringPipelineCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    book_title     = body.get("book_title", "")
    chapter_wavs   = body.get("chapter_wavs", [])
    chapter_titles = body.get("chapter_titles", None)
    if not book_title or not chapter_wavs:
        return JSONResponse({"error": "book_title and chapter_wavs are required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.master_chapters, book_title, chapter_wavs, chapter_titles
    )
    return JSONResponse(result)


@router.get("/book/output/{book_title}")
async def book_output(book_title: str):
    b = _get_brain()
    cell = b._cells.get("audio_master") if b else None
    if not cell:
        return JSONResponse({"error": "AudioMasteringPipelineCell not registered"}, status_code=404)
    return JSONResponse(cell.list_output(book_title))


@router.post("/book/metadata")
async def book_metadata(body: dict):
    """Generate audiobook metadata JSON (chapters, characters, duration estimates)."""
    b = _get_brain()
    cell = b._cells.get("book_editor") if b else None
    if not cell:
        return JSONResponse({"error": "BookEditingPipelineCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    book_title = body.get("book_title", "")
    file_path  = body.get("file_path", "")
    author     = body.get("author", "Unknown")
    if not book_title or not file_path:
        return JSONResponse({"error": "book_title and file_path are required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.generate_metadata, book_title, file_path, author
    )
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# CraniMem
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/cranimem/graph")
async def cranimem_graph():
    b = _get_brain()
    cell = b._cells.get("cranimem") if b else None
    if not cell:
        return JSONResponse({"error": "CraniMemCell not registered"}, status_code=404)
    if cell._kg is None:
        return JSONResponse({"error": "KG not initialized yet"}, status_code=503)
    return JSONResponse(cell._kg.stats())


@router.get("/cranimem/slots")
async def cranimem_slots():
    b = _get_brain()
    cell = b._cells.get("cranimem") if b else None
    if not cell:
        return JSONResponse({"error": "CraniMemCell not registered"}, status_code=404)
    return JSONResponse(cell.pool_stats())


@router.post("/cranimem/query")
async def cranimem_query(body: dict):
    b = _get_brain()
    cell = b._cells.get("cranimem") if b else None
    if not cell:
        return JSONResponse({"error": "CraniMemCell not registered"}, status_code=404)
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(None, cell.query, query, top_k)
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# GWT Broadcast
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/quantum_mesh/gws")
async def quantum_mesh_gws():
    b = _get_brain()
    cell = b._cells.get("quantum_mesh") if b else None
    if not cell:
        return JSONResponse({"error": "QuantumMeshCell not registered"}, status_code=404)
    return JSONResponse({
        "current_broadcast": cell.get_broadcast(),
        "history":           cell.get_gws_history(),
        "total_broadcasts":  cell._gws_turn_counter,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SPIN
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/spin/collect")
async def spin_collect(body: dict):
    b = _get_brain()
    cell = b._cells.get("spin") if b else None
    if not cell:
        return JSONResponse({"error": "SPINCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    prompt  = body.get("prompt", "")
    context = body.get("context", "")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.collect_spin_pair, prompt, context
    )
    return JSONResponse(result)


@router.post("/spin/round")
async def spin_round():
    b = _get_brain()
    cell = b._cells.get("spin") if b else None
    if not cell:
        return JSONResponse({"error": "SPINCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(None, cell.trigger_spin_round)
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# Debate
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/debate/stats")
async def debate_stats():
    b = _get_brain()
    cell = b._cells.get("debate") if b else None
    if not cell:
        return JSONResponse({"error": "DebateCell not registered"}, status_code=404)
    return JSONResponse(cell.stats())


@router.post("/debate/run")
async def debate_run(body: dict):
    b = _get_brain()
    cell = b._cells.get("debate") if b else None
    if not cell:
        return JSONResponse({"error": "DebateCell not registered"}, status_code=404)
    if cell._status.value == "dormant":
        await cell._boot()
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    result = await cell._run_debate(question)
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════════════════════
# Liquid Voice
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/liquid_voice/state")
async def liquid_voice_state():
    b = _get_brain()
    cell = b._cells.get("liquid_voice") if b else None
    if not cell:
        return JSONResponse({"error": "LiquidVoiceCell not registered"}, status_code=404)
    return JSONResponse(cell.get_state())


# ══════════════════════════════════════════════════════════════════════════════
# Learning Lab
# ══════════════════════════════════════════════════════════════════════════════

def _learning_cell():
    b = _get_brain()
    if b:
        cell = b._cells.get("learning_lab")
        if cell:
            return cell
    # Fallback: instantiate directly if not registered
    from brain.cells.learning_lab import LearningLabCell
    return LearningLabCell()


@router.get("/learning/status")
async def learning_status():
    """Returns combined factory + dream queue status."""
    import asyncio
    cell = _learning_cell()
    status = await asyncio.get_event_loop().run_in_executor(None, cell.get_status)
    return JSONResponse(status)


@router.get("/learning/queue")
async def learning_queue():
    """Returns full dream queue."""
    import asyncio
    cell = _learning_cell()
    queue = await asyncio.get_event_loop().run_in_executor(None, cell.get_dream_queue)
    return JSONResponse({"queue": queue})


@router.post("/learning/inject-dream")
async def learning_inject_dream(body: dict):
    """Inject content into tonight's dream queue."""
    import asyncio
    content = body.get("content", "").strip()
    tags_raw = body.get("tags", "")
    tags = ", ".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)
    cell = _learning_cell()
    result = await asyncio.get_event_loop().run_in_executor(None, cell.inject_dream, content, tags)
    # Also try to forward to omega/dreams endpoint if available
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(
                "http://127.0.0.1:8870/omega/trigger",
                json={"type": "dream_inject", "content": content}
            )
    except Exception:
        pass
    return JSONResponse(result)


@router.post("/learning/challenge")
async def learning_challenge(body: dict):
    """Add a challenge to the training factory."""
    import asyncio
    prompt     = body.get("prompt", "").strip()
    difficulty = body.get("difficulty", "medium")
    domain     = body.get("domain", "reasoning")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    cell = _learning_cell()
    result = await asyncio.get_event_loop().run_in_executor(
        None, cell.add_challenge, prompt, difficulty, domain
    )
    # Also forward to factory control if available
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(
                "http://127.0.0.1:8870/factory/control",
                json={"action": "challenge", "prompt": prompt, "difficulty": difficulty, "domain": domain}
            )
    except Exception:
        pass
    return JSONResponse(result)


@router.get("/learning/suggestions")
async def learning_suggestions():
    """Returns all submitted training suggestions."""
    import asyncio
    cell = _learning_cell()
    suggestions = await asyncio.get_event_loop().run_in_executor(None, cell.get_suggestions)
    return JSONResponse({"suggestions": suggestions})


@router.post("/learning/suggestion")
async def learning_add_suggestion(body: dict):
    """Add a training suggestion."""
    import asyncio
    content = body.get("content", "").strip()
    tags    = body.get("tags", "")
    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)
    cell = _learning_cell()
    result = await asyncio.get_event_loop().run_in_executor(None, cell.add_suggestion, content, tags)
    return JSONResponse(result)


# ============================================================================
# PRESERVATION -- Perfect Preservation Protocol
# ============================================================================

def _preservation_cell():
    b = _get_brain()
    cell = b._cells.get('preservation') if b else None
    if not cell:
        from brain.cells.preservation import PreservationCell
        cell = PreservationCell()
    return cell


@router.get('/preservation/status')
async def preservation_status():
    """Full preservation status -- shadow age, file list, daemon alive."""
    cell = _preservation_cell()
    return JSONResponse(cell.get_status())


@router.get('/preservation/health')
async def preservation_health():
    """Quick health check for system monitor."""
    cell = _preservation_cell()
    return JSONResponse(cell.health())


@router.post('/preservation/pulse')
async def preservation_pulse():
    """Trigger an immediate echo cycle (force-sync shadow right now)."""
    cell = _preservation_cell()
    return JSONResponse(cell.force_pulse())


@router.post('/preservation/promote')
async def preservation_promote():
    """Emergency: promote shadow to primary. Use only when primary is corrupted."""
    cell = _preservation_cell()
    return JSONResponse(cell.promote_shadow())
