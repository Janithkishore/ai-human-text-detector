import { useEffect, useState } from 'react';
import { Brain, Clock3, Loader2, RotateCcw, ShieldCheck, Sparkles } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API}/history`);
      if (res.ok) setHistory(await res.json());
    } catch { /* API may not be running yet */ }
  };

  useEffect(() => { loadHistory(); }, []);

  const detect = async () => {
    if (!text.trim()) return setError('Paste some text before detecting.');
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await fetch(`${API}/detect`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Detection failed');
      setResult(data); loadHistory();
    } catch (err) { setError(err.message || 'Unable to connect to the backend.'); }
    finally { setLoading(false); }
  };

  const clear = () => { setText(''); setResult(null); setError(''); };
  const human = result?.human_probability || 0;
  const ai = result?.ai_probability || 0;

  return (
    <main className="page">
      <header className="hero">
        <div className="brand"><div className="logo"><Brain size={25}/></div><span>TextLens AI</span></div>
        <div className="status"><span className="dot"/> RoBERTa powered</div>
        <h1>AI or <span>Human?</span></h1>
        <p>Analyze writing with a transformer-based classifier and get a probability breakdown in seconds.</p>
      </header>

      <section className="workspace">
        <div className="card input-card">
          <div className="card-head"><div><h2>Analyze text</h2><p>Paste a paragraph, essay, article, or message.</p></div><span className="badge"><ShieldCheck size={15}/> Private</span></div>
          <textarea value={text} onChange={e => setText(e.target.value)} maxLength={12000} placeholder="Paste your text here..." />
          <div className="toolbar"><span>{text.length.toLocaleString()} / 12,000 characters</span><div><button className="secondary" onClick={clear}><RotateCcw size={15}/> Clear</button><button className="primary" onClick={detect} disabled={loading}>{loading ? <><Loader2 className="spin" size={16}/> Analyzing...</> : <><Sparkles size={16}/> Detect text</>}</button></div></div>
          {error && <div className="error">{error}</div>}
        </div>

        <div className="card result-card">
          {!result && !loading && <div className="empty"><Brain size={42}/><h3>Your result appears here</h3><p>Run a detection to see whether the text is more likely AI-generated or human-written.</p></div>}
          {loading && <div className="empty"><Loader2 className="spin" size={42}/><h3>Analyzing writing...</h3><p>The RoBERTa model is evaluating the text.</p></div>}
          {result && <>
            <div className="result-label"><span>Classification</span><strong className={result.label === 'AI' ? 'ai' : 'human'}>{result.label === 'AI' ? 'Likely AI-generated' : 'Likely Human-written'}</strong></div>
            <div className="score"><div className="score-number">{Math.max(ai, human).toFixed(1)}<small>%</small></div><div className="score-caption">confidence</div></div>
            <div className="bars"><div><div className="bar-title"><span>AI-generated</span><b>{ai.toFixed(2)}%</b></div><div className="bar"><i style={{width: `${ai}%`}}/></div></div><div><div className="bar-title"><span>Human-written</span><b>{human.toFixed(2)}%</b></div><div className="bar"><i style={{width: `${human}%`}}/></div></div></div>
            <p className="disclaimer">Detection is probabilistic and should not be treated as definitive proof of authorship.</p>
          </>}
        </div>
      </section>

      <section className="card history"><div className="card-head"><div><h2>Recent detections</h2><p>Your latest analyses stored in the database.</p></div><Clock3 size={20}/></div>{history.length === 0 ? <p className="muted">No detection history yet.</p> : <div className="history-list">{history.map(item => <div className="history-row" key={item.id}><div className="snippet">{item.text}</div><span className={item.label === 'AI' ? 'pill ai' : 'pill human'}>{item.label}</span><b>{Math.max(item.ai_probability, item.human_probability).toFixed(1)}%</b></div>)}</div>}</section>
      <footer>TextLens AI • RoBERTa classification • Flask API • MySQL</footer>
    </main>
  );
}
