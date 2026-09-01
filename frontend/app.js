const $=id=>document.getElementById(id), history={x:[],rssi:[],motion:[],amps:[]}; let latest=null,pending=null,lastDraw=0,recordingWasActive=false;
const layout={autosize:true,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#9eb1c8'},margin:{t:15,r:15,b:42,l:52},xaxis:{gridcolor:'#203148',automargin:true},yaxis:{gridcolor:'#203148',automargin:true}};
const plotConfig={responsive:true,displaylogo:false,scrollZoom:false};
function error(e){$('error').textContent=e?.message||String(e)} async function api(path,opts={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});if(!r.ok){let d;try{d=await r.json()}catch{}throw Error(d?.detail||`${r.status} ${r.statusText}`)}return r.json()}
function fmt(v,d=2){return Number.isFinite(v)?Number(v).toFixed(d):'—'}
function processed(a){let from=Math.max(0,+$('from').value||0),to=Math.min(a.length,+$('to').value||a.length),v=a.slice(from,to),x=v.map((_,i)=>i+from);if($('removezero').checked){let keep=v.map((n,i)=>n!==0?i:-1).filter(i=>i>=0);x=keep.map(i=>x[i]);v=keep.map(i=>v[i])}let w=Math.max(1,+$('smooth').value||1);if(w>1)v=v.map((_,i)=>{let s=v.slice(Math.max(0,i-w+1),i+1);return s.reduce((a,b)=>a+b,0)/s.length});if($('normalize').checked&&v.length){let m=v.reduce((a,b)=>a+b,0)/v.length,sd=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/v.length);v=v.map(n=>sd?(n-m)/sd:0)}return{x,v}}
function renderLatest(){if(!latest)return;let q=processed(latest.amplitude);Plotly.react('amplitude',[{x:q.x,y:q.v,type:'scatter',mode:'lines',name:'amplitude'}],layout,plotConfig)}
function drawPacket(p){latest=p;$('nodata').hidden=true;[['rssi',p.rssi],['noise',p.noise_floor],['agc',p.agc_gain],['fft',p.fft_gain],['channel',p.channel],['length',p.csi_length]].forEach(([id,v])=>$(id).textContent=v??'—');let f=p.features||{};[['mean',f.mean_amplitude],['std',f.std_amplitude],['median',f.median_amplitude],['variance',f.variance_amplitude],['difference',f.frame_difference]].forEach(([id,v])=>$(id).textContent=fmt(v));renderLatest();let t=p.received_at;history.x.push(t);history.rssi.push(p.rssi);history.motion.push(p.motion_score);let picks=[.1,.2,.3,.4,.55,.65,.75,.9].map(n=>Math.min(p.amplitude.length-1,Math.floor(n*p.amplitude.length)));history.amps.push(picks.map(i=>p.amplitude[i]));if(history.x.length>300)Object.values(history).forEach(a=>a.shift());Plotly.react('history',picks.map((idx,j)=>({x:history.x,y:history.amps.map(a=>a[j]),mode:'lines',name:`SC ${idx}`})),layout,plotConfig);Plotly.react('motion',[{x:history.x,y:history.motion,mode:'lines',connectgaps:false}],layout,plotConfig);Plotly.react('rssihistory',[{x:history.x,y:history.rssi,mode:'lines'}],layout,plotConfig)}
function schedulePacket(p){pending=p;let wait=Math.max(0,125-(performance.now()-lastDraw));if(schedulePacket.timer)return;schedulePacket.timer=setTimeout(()=>{schedulePacket.timer=null;let next=pending;pending=null;lastDraw=performance.now();if(next)drawPacket(next)},wait)}

// ---- occupancy verdict -----------------------------------------------------
const phist={t:[],p:[]}; let modelThreshold=null,phist0=null;
function drawPrediction(v){
  $('replaybadge').hidden = v.source!=='replay';
  let el=$('presence'); el.textContent=v.presence; el.className=v.presence==='HOME'?'home':'away';
  $('confidence').textContent=(v.confidence*100).toFixed(1)+'%';
  let ac=$('ac'); ac.textContent=v.run_ac?'AC ON':'AC OFF'; ac.className='badge '+(v.run_ac?'acon':'acoff');
  $('why').textContent=v.reason;
  if(phist0===null)phist0=performance.now();
  phist.t.push((performance.now()-phist0)/1000); phist.p.push(v.p_home);
  if(phist.t.length>240){phist.t.shift();phist.p.shift()}
  drawPresencePlot();
}
function drawPresencePlot(){
  if(!phist.t.length)return;
  // p(occupied) over time against the tuned decision line. The y range is pinned
  // to [0,1] so the gap between the trace and the line reads as the actual
  // decision margin -- autoscaling would make a confident call look marginal.
  let x0=phist.t[0],x1=Math.max(phist.t[phist.t.length-1],x0+1),thr=1-modelThreshold;
  let traces=[{x:phist.t,y:phist.p,mode:'lines',name:'p(occupied)',
    line:{color:'#ffca6b',width:2,shape:'spline'},fill:'tozeroy',fillcolor:'rgba(255,202,107,.13)'}];
  if(modelThreshold!==null)traces.push({x:[x0,x1],y:[thr,thr],mode:'lines',
    name:'decision line',line:{color:'#64e6b1',width:1.5,dash:'dot'},hoverinfo:'skip'});
  Plotly.react('presenceplot',traces,{...layout,margin:{t:8,r:12,b:34,l:48},showlegend:false,
    xaxis:{...layout.xaxis,range:[x0,x1],title:{text:'seconds',font:{size:10}}},
    yaxis:{...layout.yaxis,range:[0,1],autorange:false,dtick:0.5,
           title:{text:'p(occupied)',font:{size:10}}},
    annotations:modelThreshold===null?[]:[{x:x1,y:thr,xanchor:'right',yanchor:'bottom',
      text:`decision line ${thr.toFixed(2)}`,showarrow:false,font:{size:9,color:'#64e6b1'}}]
  },plotConfig);
}
function renderModelInfo(p){
  if(!p||!p.loaded){modelThreshold=null;
    $('modelinfo').textContent=(p&&p.error)||(p&&p.message)||'No trained model.';
    if(!phist.t.length){$('presence').textContent='—';$('presence').className='none';
      $('why').textContent='Train a model first: python -m src.train_esp32'}
    return}
  modelThreshold=p.threshold;
  let cv=p.cv?`  ·  cross-validated AWAY recall ${(p.cv.recall_away*100).toFixed(1)}%, HOME recall ${(p.cv.recall_home*100).toFixed(1)}%`:'';
  $('modelinfo').textContent=`${p.model} on ${p.feature_set} features  ·  ${p.window_packets}-packet window  ·  AWAY threshold ${p.threshold.toFixed(2)}  ·  trained on ${p.n_recordings} recordings / ${p.n_windows} windows${cv}`;
  if(!p.latest&&!phist.t.length){$('presence').textContent='—';$('presence').className='none';
    $('why').textContent=`Waiting for a full window (${p.buffer}/${p.window_packets} packets).`}
}

// ---- replay ----------------------------------------------------------------
let replayFiles=[];
async function replayList(){try{replayFiles=await api('/api/replay/recordings');
  $('replayfile').innerHTML='<option value="">Choose a recording</option>'+
    replayFiles.map(r=>`<option value="${r.filename}">${r.filename}</option>`).join('')}catch(e){error(e)}}
async function startReplay(filename){
  if(!filename){error('Choose a recording, or use one of the buttons above.');return}
  try{error('');
    // switching sources mid-replay is the normal case when demoing, so stop
    // whatever is running rather than refusing with "already running"
    await api('/api/replay/stop',{method:'POST'});
    phist.t=[];phist.p=[];phist0=null;$('presence').textContent='—';
    $('presence').className='none';$('why').textContent='Filling the first window…';
    Plotly.purge('presenceplot');
    await api('/api/replay/start',{method:'POST',
      body:JSON.stringify({filename,speed:+$('replayspeed').value||1})});await status()}catch(e){error(e)}}
// One click per condition: grab the first recording carrying that label. Makes the
// model demonstrable without a board, and without hunting through 30 filenames.
document.querySelectorAll('.demo').forEach(b=>b.onclick=()=>{
  let want=b.dataset.label,
      hit=replayFiles.find(r=>r.filename.includes(want)
        &&(want!=='occupied_still'||!r.filename.includes('moving')));
  if(!hit){error(`No ${want} recording found in recordings/.`);return}
  $('replayfile').value=hit.filename; startReplay(hit.filename)});
$('replaystart').onclick=()=>startReplay($('replayfile').value);
$('replaystop').onclick=async()=>{try{await api('/api/replay/stop',{method:'POST'});await status()}catch(e){error(e)}};

async function ports(){try{let ps=await api('/api/ports'),old=$('port').value;$('port').innerHTML='<option value="">Select RX serial port</option>'+ps.map(p=>`<option value="${p.device}">${p.device} — ${p.description}</option>`).join('');$('port').value=old}catch(e){error(e)}}
async function status(){try{let s=await api('/api/status'),x=s.serial;$('status').textContent=x.connected?'Connected':'Disconnected';$('status').className='badge '+(x.connected?'on':'off');$('packets').textContent=x.packet_count;$('rate').textContent=fmt(x.packets_per_second,1);$('rejected').textContent=x.rejected_count;if(x.error)error(x.error);if(x.connected&&x.seconds_since_last_packet!==null&&x.seconds_since_last_packet>3)$('nodata').hidden=false;let r=s.recording;if(!r.active){let done=r.last_result?.stop_reason==='automatic';$('recstate').textContent=done?`Completed automatically · ${r.last_result.packet_count} packets`:'Not recording'}else if(r.state==='countdown')$('recstate').textContent=`Starting in ${Math.max(1,Math.ceil(r.countdown_remaining))}s — move to the test position`;else{let elapsed=Math.min(r.duration_seconds,Math.max(0,Math.floor((Date.now()-Date.parse(r.started_at))/1000)));$('recstate').textContent=`Recording ${elapsed}s / ${r.duration_seconds}s · ${r.packet_count} packets`}renderModelInfo(s.prediction);let rp=s.replay||{};$('replaystate').textContent=rp.active?`Replaying ${rp.filename} · ${rp.packet_count} packets`:(rp.finished?`Finished · ${rp.packet_count} packets`:'Idle');$('replaystart').disabled=!!rp.active;$('replaystop').disabled=!rp.active;if(rp.active)$('nodata').hidden=true;$('start').disabled=r.active;$('stop').disabled=!r.active;if(recordingWasActive&&!r.active)await recordings();recordingWasActive=r.active}catch(e){error(e)}}
function ws(){let s=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/live`);s.onmessage=e=>{let m=JSON.parse(e.data);if(m.type==='packet')schedulePacket(m.data);else if(m.type==='prediction')drawPrediction(m.data)};s.onclose=()=>setTimeout(ws,1000)}
$('refresh').onclick=ports;$('connect').onclick=async()=>{try{error('');await api('/api/connect',{method:'POST',body:JSON.stringify({port:$('port').value})});connectedAt=Date.now();await status()}catch(e){error(e)}};$('disconnect').onclick=async()=>{try{await api('/api/disconnect',{method:'POST'});await status()}catch(e){error(e)}};
$('start').onclick=async()=>{try{await api('/api/recordings/start',{method:'POST',body:JSON.stringify({label:$('label').value,notes:$('notes').value,delay_seconds:10,duration_seconds:30})});await status()}catch(e){error(e)}};$('stop').onclick=async()=>{try{await api('/api/recordings/stop',{method:'POST'});await status();await recordings()}catch(e){error(e)}};
async function recordings(){try{let rs=await api('/api/recordings');$('recordings').innerHTML='<option value="">Choose a recording</option>'+rs.map(r=>`<option value="${r.filename}">${r.filename} · ${r.label||'unknown'}</option>`).join('')}catch(e){error(e)}}
$('reload').onclick=recordings;$('load').onclick=async()=>{try{let d=await api('/api/recordings/'+encodeURIComponent($('recordings').value)),p=d.packets;$('details').textContent=JSON.stringify({filename:d.filename,packets_loaded:p.length,truncated:d.truncated,label:p[0]?.session_label??null,notes:p[0]?.session_notes??null,started_at:p[0]?.session_timestamp??null},null,2);Plotly.react('replay',[{x:p.map(x=>x.received_at),y:p.map(x=>x.motion_score),name:'motion score',mode:'lines'}],layout,plotConfig)}catch(e){error(e)}};
$('export').onclick=()=>{let name=$('recordings').value;if(!name){error('Choose a recording before exporting CSV.');return}error('');location.href='/api/recordings/'+encodeURIComponent(name)+'/csv'};
['from','to','normalize','smooth','removezero'].forEach(id=>$(id).onchange=renderLatest);ports();recordings();replayList();status();setInterval(status,1000);ws();
