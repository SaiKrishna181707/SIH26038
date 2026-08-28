import React, { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, AlertTriangle, Bell, CheckCircle2, ChevronDown, ClipboardList, CloudOff, Eye, FileText, Globe2, LayoutDashboard, Menu, ScanEye, Settings, ShieldCheck, UploadCloud, Users, X } from 'lucide-react';
import './styles.css';

const probabilities = [['No DR', 2], ['Mild DR', 5], ['Moderate DR', 91], ['Severe DR', 2], ['Proliferative DR', 0]];
const findings = ['Microaneurysm regions', 'Hemorrhage regions', 'Exudate regions'];

function App() {
  const [page, setPage] = useState('screening');
  const [mobile, setMobile] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(false);
  const input = useRef(null);

  const selectImage = (f) => {
    if (!f || !f.type.startsWith('image/')) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(false);
  };

  const analyze = () => {
    if (!file) return;
    setAnalyzing(true);
    setTimeout(() => { setAnalyzing(false); setResult(true); }, 900);
  };

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(''); setResult(false);
  };

  const nav = (p) => { setPage(p); setMobile(false); };

  return <div className="app">
    <aside className={`sidebar ${mobile ? 'open' : ''}`}>
      <div className="brand">
        <div className="brand-mark"><Eye size={18}/></div>
        <div><b>RetinaCare</b><span>Clinical Screening Console</span></div>
        <button className="mobile-close" onClick={() => setMobile(false)}><X size={18}/></button>
      </div>
      <div className="workspace"><span className="live-dot"/> Rural screening workspace</div>
      <div className="section-label">WORKSPACE</div>
      <Nav icon={StethoscopeIcon} text="Screening" active={page === 'screening'} onClick={() => nav('screening')}/>
      <Nav icon={LayoutDashboard} text="Overview" active={page === 'overview'} onClick={() => nav('overview')}/>
      <Nav icon={Users} text="Patient history" active={page === 'history'} onClick={() => nav('history')}/>
      <Nav icon={FileText} text="Reports" active={page === 'reports'} onClick={() => nav('reports')}/>
      <div className="section-label system-label">SYSTEM</div>
      <Nav icon={Settings} text="Settings"/><Nav icon={ShieldCheck} text="Help & guidance"/>
      <div className="sidebar-bottom">
        <div className="system-status"><b><span className="live-dot"/> AI system ready</b><span>Model · EfficientNet-B0</span><span>Last sync · 09:42 IST</span></div>
        <div className="operator"><div>SK</div><span><b>Screening operator</b><small>Rural care unit</small></span></div>
      </div>
    </aside>
    {mobile && <div className="overlay" onClick={() => setMobile(false)}/>} 
    <main>
      <header className="topbar">
        <button className="menu" onClick={() => setMobile(true)}><Menu size={19}/></button>
        <div><b>{page === 'screening' ? 'Screening' : page[0].toUpperCase() + page.slice(1)}</b><span>SIH26038 · Explainable AI for diabetic retinopathy screening</span></div>
        <div className="top-actions"><span className="offline"><CloudOff size={13}/> Offline-ready</span><span>Last synced <b>09:42</b></span><Bell size={16}/><span className="lang"><Globe2 size={13}/> EN <ChevronDown size={11}/></span></div>
      </header>
      <section className="page">
        {page === 'screening' ? <Screening {...{file,preview,analyzing,result,input,selectImage,analyze,clear}}/> : <SimplePage page={page} nav={nav}/>} 
      </section>
    </main>
  </div>;
}

function Screening({file,preview,analyzing,result,input,selectImage,analyze,clear}) {
  return <>
    <div className="page-head"><div><small>SCREENING WORKSPACE</small><h1>Retinal screening</h1><p>Capture or upload a fundus image, then review the AI-assisted assessment.</p></div><div className="head-tools"><span><i/> Session ready</span><button className="outline">Export</button></div></div>
    <div className="screen-grid">
      <div>
        <Panel title="Patient details" step="STEP 01"><div className="fields"><Field label="Patient ID" value="RC-10428"/><Field label="Patient name" value="Sunita Devi"/><Field label="Age" value="52"/></div></Panel>
        <Panel title="Fundus image" step="STEP 02" action={file && <button className="plain" onClick={clear}>Clear</button>}>
          {!file ? <div className="drop" onClick={() => input.current.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault();selectImage(e.dataTransfer.files[0])}}>
            <div className="upload-mark"><UploadCloud size={20}/></div><b>Drop fundus image here</b><span>JPG, JPEG or PNG · up to 10 MB</span><button className="primary" onClick={(e) => {e.stopPropagation();input.current.click()}}>Choose image</button><input ref={input} hidden type="file" accept="image/jpeg,image/png,image/jpg" onChange={e => selectImage(e.target.files[0])}/>
          </div> : <div className="image-work"><img src={preview} alt="Fundus preview"/><div><em><CheckCircle2 size={13}/> Image loaded</em><b>{file.name}</b><span>{(file.size/1024/1024).toFixed(2)} MB · RGB image</span><div className="actions"><button className="primary" onClick={analyze} disabled={analyzing}>{analyzing ? 'Analyzing…' : 'Analyze image'}</button><button className="secondary" onClick={() => input.current.click()}>Change</button></div></div></div>}
        </Panel>
        {result && <Panel title="Visual explanation" step="STEP 04"><div className="explain"><div><small>ORIGINAL</small><img src={preview} alt="Original"/></div><div><small>AI ATTENTION MAP</small><div className="heat"><img src={preview} alt="Attention map"/><i/><i/></div></div></div><p className="muted">Highlighted regions indicate areas that influenced the current prototype score.</p><span className="prototype-note">Prototype visualization · final evidence model will be connected through the backend.</span></Panel>}
      </div>
      <div>
        {!result ? <div className="empty-result"><div><Eye size={22}/></div><small>AI ASSESSMENT</small><h2>No screening result</h2><p>Upload a retinal image on the left to run the AI-assisted screening workflow.</p></div> : <>
          <Panel title="AI assessment" step="STEP 03"><div className="assessment-head"><div><small>SCREENING RESULT</small><h2>Moderate Diabetic Retinopathy</h2><span className="severity">Moderate</span><strong className="confidence">91.4% <em>confidence</em></strong></div><div className="score"><b>91</b><span>CONF.</span></div></div><div className="prob-title"><b>Class probability</b><span>Model output</span></div>{probabilities.map(([name,value]) => <div className={`prob ${name === 'Moderate DR' ? 'selected' : ''}`} key={name}><div><span>{name}</span><b>{value}%</b></div><i><u style={{width:`${value}%`}}/></i></div>)}</Panel>
          <Panel title="Clinical decision support" step="REVIEW"><div className="decision"><AlertTriangle size={17}/><div><b>Specialist evaluation recommended</b><p>The current AI screening indicates findings that should be reviewed by an eye-care professional.</p></div></div><button className="primary wide">Create referral</button><div className="safety"><ShieldCheck size={13}/> Screening support only · not a diagnosis</div></Panel>
          <Panel title="Detected findings" step="EVIDENCE">{findings.map(x => <div className="finding" key={x}><CheckCircle2 size={14}/>{x}<small>prototype</small></div>)}</Panel>
        </>}
      </div>
    </div>
  </>;
}

function SimplePage({page,nav}) { return <><div className="page-head"><div><small>{page.toUpperCase()}</small><h1>{page === 'history' ? 'Patient history' : page === 'reports' ? 'Screening reports' : 'Overview'}</h1><p>Operational view of the RetinaCare screening workspace.</p></div>{page !== 'overview' && <button className="primary" onClick={() => nav('screening')}>New screening</button>}</div><div className="metric-strip">{[['Screenings today','14','6 pending sync'],['Referrals raised','3','This week'],['Awaiting review','2','Low confidence'],['Mean confidence','89.6%','Last 30 screenings']].map(x=><div className="metric-card"><small>{x[0]}</small><strong>{x[1]}</strong><span>{x[2]}</span></div>)}</div><Panel title="Recent screening activity"><table><thead><tr><th>Date</th><th>Patient</th><th>Result</th><th>Confidence</th><th>Status</th></tr></thead><tbody>{[['28 Aug 2026','Sunita Devi','Moderate DR','91.4%','Pending'],['12 Feb 2026','Sunita Devi','Mild DR','87.3%','Reviewed'],['04 Feb 2026','Ramesh Kumar','No DR','95.1%','Reviewed'],['19 Jan 2026','Anita Sharma','Severe DR','88.2%','Reviewed']].map(r=><tr><td>{r[0]}</td><td><b>{r[1]}</b></td><td><span className="table-tag">{r[2]}</span></td><td className="green-text">{r[3]}</td><td>{r[4]}</td></tr>)}</tbody></table></Panel></> }
function Nav({icon:Icon,text,active,onClick}) { return <button className={`nav ${active?'active':''}`} onClick={onClick}><Icon size={16}/>{text}</button> }
function Field({label,value}) { return <label className="field"><span>{label}</span><input defaultValue={value}/></label> }
function Panel({title,step,action,children}) { return <section className="panel"><div className="panel-head"><div><b>{title}</b>{step && <small>{step}</small>}</div>{action}</div><div className="panel-body">{children}</div></section> }
function StethoscopeIcon(){return <Activity size={16}/>}

createRoot(document.getElementById('root')).render(<App/>);
