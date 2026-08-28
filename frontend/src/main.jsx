import React from 'react';
import { createRoot } from 'react-dom/client';
import { Eye, LayoutDashboard, Stethoscope, Users, FileText, Settings, HelpCircle, Bell, CloudOff, UploadCloud, Activity, CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';
import './styles.css';

const classes = [['No DR',2],['Mild DR',5],['Moderate DR',91],['Severe DR',2],['Proliferative DR',0]];

function App(){
  const [image,setImage]=React.useState(null);
  const [preview,setPreview]=React.useState('');
  const [analyzing,setAnalyzing]=React.useState(false);
  const [result,setResult]=React.useState(false);
  const input=React.useRef();
  const select=e=>{const f=e.target.files?.[0];if(!f)return;setImage(f);setPreview(URL.createObjectURL(f));setResult(false)};
  const analyze=()=>{if(!image)return;setAnalyzing(true);setTimeout(()=>{setAnalyzing(false);setResult(true)},900)};
  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brandIcon"><Eye size={18}/></div><div><b>RetinaCare</b><span>Clinical Screening Console</span></div></div>
      <div className="workspace"><i/> Rural screening workspace</div>
      <div className="sectionLabel">WORKSPACE</div>
      <Nav icon={Stethoscope} text="Screening" active/><Nav icon={LayoutDashboard} text="Overview"/><Nav icon={Users} text="Patient history"/><Nav icon={FileText} text="Reports"/>
      <div className="sectionLabel system">SYSTEM</div><Nav icon={Settings} text="Settings"/><Nav icon={HelpCircle} text="Help & guidance"/>
      <div className="sidebarBottom"><div className="status"><b><i/> AI system ready</b><span>Model · EfficientNet-B0</span><span>Last sync · 09:42 IST</span></div><div className="operator"><div>SK</div><span><b>Screening operator</b><small>Rural care unit</small></span></div></div>
    </aside>
    <main>
      <header><button className="mobileMenu">☰</button><div><b>Screening</b><span>SIH26038 · Explainable AI for diabetic retinopathy screening</span></div><div className="headerRight"><span className="offline"><CloudOff size={13}/> Offline-ready</span><span>Last synced <b>09:42</b></span><Bell size={17}/><span className="lang">EN ▾</span></div></header>
      <div className="page">
        <div className="pageHead"><div><small>SCREENING WORKSPACE</small><h1>Retinal screening</h1><p>Capture or upload a fundus image, then review the AI-assisted assessment.</p></div><div className="toolbar"><span><i/> Session ready</span><button>Export</button></div></div>
        <div className="grid">
          <div>
            <Panel title="Patient details" step="STEP 01"><div className="fields"><Field label="Patient ID" value="RC-10428"/><Field label="Patient name" value="Sunita Devi"/><Field label="Age" value="52"/></div></Panel>
            <Panel title="Fundus image" step="STEP 02" action={image&&<button className="plain" onClick={()=>{setImage(null);setPreview('');setResult(false)}}>Clear</button>}>
              {!image?<div className="drop" onClick={()=>input.current.click()}><div className="uploadIcon"><UploadCloud size={21}/></div><b>Drop fundus image here</b><span>JPG, JPEG or PNG · up to 10 MB</span><button className="primary">Choose image</button><input ref={input} hidden type="file" accept="image/*" onChange={select}/></div>:<div className="imageWork"><img src={preview}/><div><em><CheckCircle2 size={14}/> Image loaded</em><b>{image.name}</b><span>{(image.size/1024/1024).toFixed(2)} MB · RGB image</span><div className="actions"><button className="primary" onClick={analyze} disabled={analyzing}>{analyzing?'Analyzing…':'Analyze image'}</button><button className="secondary" onClick={()=>input.current.click()}>Change</button></div></div></div>}
            </Panel>
            {result&&<Panel title="Visual explanation" step="STEP 04"><div className="explain"><div><small>ORIGINAL</small><img src={preview}/></div><div><small>AI ATTENTION MAP</small><div className="heat"><img src={preview}/><i/><i/></div></div></div><p className="muted">Highlighted regions indicate areas that influenced the current prototype score.</p></Panel>}
          </div>
          <div>
            {!result?<div className="empty"><div><Eye size={25}/></div><small>AI ASSESSMENT</small><h2>No screening result</h2><p>Upload a retinal image on the left to run the AI-assisted screening workflow.</p></div>:<>
              <Panel title="AI assessment" step="STEP 03"><div className="resultHead"><div><small>SCREENING RESULT</small><h2>Moderate Diabetic Retinopathy</h2><span className="badge">Moderate</span><strong className="confidence">91.4% <em>confidence</em></strong></div><div className="score"><b>91</b><span>CONF.</span></div></div><div className="probTitle"><b>Class probability</b><span>Model output</span></div>{classes.map(([n,v])=><div className={'prob '+(n==='Moderate DR'?'selected':'')}><div><span>{n}</span><b>{v}%</b></div><i><u style={{width:v+'%'}}/></i></div>)}</Panel>
              <Panel title="Clinical decision support" step="REVIEW"><div className="decision"><AlertTriangle/><div><b>Specialist evaluation recommended</b><p>The current AI screening indicates findings that should be reviewed by an eye-care professional.</p></div></div><button className="primary wide">Create referral</button><div className="safety"><ShieldCheck size={14}/> Screening support only · not a diagnosis</div></Panel>
              <Panel title="Detected findings" step="EVIDENCE">{['Microaneurysm regions','Hemorrhage regions','Exudate regions'].map(x=><div className="finding"><CheckCircle2 size={14}/>{x}<small>prototype</small></div>)}</Panel></>}
          </div>
        </div>
      </div>
    </main>
  </div>
}
function Nav({icon:Icon,text,active}){return <button className={'nav '+(active?'active':'')}><Icon size={16}/>{text}</button>}
function Field({label,value}){return <label className="field"><span>{label}</span><input defaultValue={value}/></label>}
function Panel({title,step,action,children}){return <section className="panel"><div className="panelHead"><div><b>{title}</b><small>{step}</small></div>{action}</div><div className="panelBody">{children}</div></section>}
createRoot(document.getElementById('root')).render(<App/>);
