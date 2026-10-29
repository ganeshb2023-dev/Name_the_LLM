'use client';

import { useMemo, useState } from 'react';
import { Activity, ArrowDownRight, ArrowRight, CheckCircle2, CircleHelp, FileText, Fingerprint, LoaderCircle, RotateCcw, ScanText, ShieldCheck, Sparkles, Target } from 'lucide-react';
import type { ClassificationResult } from '@/lib/types';

const MAX = 20_000;
const sample = 'A useful way to understand machine learning is to imagine teaching a computer with examples instead of writing every rule by hand. If you show it many labelled pictures of cats and dogs, it can learn patterns that help it classify a new picture.';

export default function Classifier() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const count = useMemo(() => text.length, [text]);

  function reset() { setText(''); setResult(null); setError(''); }
  async function classify() {
    setError(''); setResult(null);
    if (text.trim().length < 20) { setError('Enter at least 20 characters so the classifier has enough text to analyze.'); return; }
    if (text.length > MAX) { setError(`The maximum passage length is ${MAX.toLocaleString()} characters.`); return; }
    setLoading(true);
    try {
      const response = await fetch('/api/classify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : 'Something went wrong. Please try again.');
      setResult(data as ClassificationResult);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not complete the request. Please try again.'); }
    finally { setLoading(false); }
  }

  return <main className="page-shell">
    <nav className="topbar"><a className="brand" href="#top" aria-label="Name That LLM home"><span className="brand-mark"><Fingerprint size={21}/></span><span>origin<span className="brand-dot">.</span>ai</span></a><div className="top-links"><span className="status-dot"/> <span>Text origin lab</span><a href="#how-it-works">How it works <ArrowDownRight size={14}/></a></div></nav>
    <section className="hero" id="top">
      <div className="eyebrow"><Sparkles size={14}/> THE LLM IDENTIFICATION CHALLENGE</div>
      <h1>Every model has<br/><span>a fingerprint.</span></h1>
      <p className="hero-copy">Explore the patterns that can help distinguish AI-generated text. Paste a passage to send it through the classifier workflow.</p>
      <div className="hero-meta"><span><ShieldCheck size={15}/> Transparent workflow</span><span><Activity size={15}/> 18 candidate classes*</span><span><Target size={15}/> Balanced accuracy</span></div>
    </section>

    <section className="workspace" aria-label="Text classifier">
      <div className="panel input-panel">
        <div className="panel-heading"><div><div className="step-label">STEP 01 <span>· INPUT</span></div><h2>Drop in a passage</h2></div><span className="icon-box"><FileText size={19}/></span></div>
        <label className="sr-only" htmlFor="passage">Text passage to analyze</label>
        <textarea id="passage" value={text} onChange={(e) => { setText(e.target.value); setError(''); setResult(null); }} maxLength={MAX} placeholder="Paste an AI-generated response here…\n\nLonger passages can provide more writing-style signals." spellCheck={false}/>
        <div className="input-footer"><span className={count > MAX * .9 ? 'count count-warn' : 'count'}>{count.toLocaleString()} / {MAX.toLocaleString()} characters</span><button className="text-button" type="button" onClick={() => { setText(sample); setError(''); setResult(null); }}>Try sample <ArrowRight size={14}/></button></div>
        {error && <p className="error-message" role="alert">{error}</p>}
        <button className="primary-button" onClick={classify} disabled={loading || text.trim().length < 20 || count > MAX}>{loading ? <><LoaderCircle className="spin" size={17}/> Processing passage…</> : <>Analyze passage <ArrowRight size={17}/></>}</button>
        <p className="privacy-note"><ShieldCheck size={14}/> Your text is sent only to the configured classifier service. Avoid pasting sensitive information.</p>
      </div>

      <div className="panel output-panel" aria-live="polite">
        <div className="panel-heading"><div><div className="step-label">STEP 02 <span>· ANALYSIS</span></div><h2>Classification result</h2></div><span className="icon-box purple"><ScanText size={19}/></span></div>
        {!result && !loading && <div className="empty-state"><div className="orbit"><div className="orbit-core"><Fingerprint size={35}/></div><span className="orbit-dot dot-one"/><span className="orbit-dot dot-two"/><span className="orbit-dot dot-three"/></div><h3>Ready when you are</h3><p>Submit a passage to check the classifier connection and see the analysis result.</p><div className="empty-divider"/><div className="mini-row"><span className="mini-icon"><CircleHelp size={16}/></span><span><b>No guesswork</b><small>Unconfigured models never receive a fabricated label.</small></span></div></div>}
        {loading && <div className="loading-state"><LoaderCircle className="spin" size={31}/><h3>Processing text</h3><p>Sending the passage to the classifier service.</p></div>}
        {result && !result.configured && <div className="not-configured"><div className="notice-icon"><CircleHelp size={23}/></div><span className="result-kicker">MODEL CONNECTION NEEDED</span><h3>UI ready. Model not connected.</h3><p>{result.message}</p><div className="setup-steps"><div><span>1</span> Train a classifier on labelled competition data.</div><div><span>2</span> Deploy an API that serves real predictions.</div><div><span>3</span> Set <code>CLASSIFIER_API_URL</code> in Vercel.</div></div></div>}
        {result?.configured && <div className="live-result"><div className="live-pill"><CheckCircle2 size={14}/> LIVE MODEL RESPONSE</div><span className="result-kicker">TOP PREDICTED LABEL</span><h3 className="prediction-label">{result.label}</h3><p className="result-caption">Returned by your configured classifier. Scores are shown as provided by the model API.</p><div className="scores-heading"><span>TOP SCORES</span><span>MODEL OUTPUT</span></div>{result.scores?.map((item) => <div className="score-row" key={item.label}><span className="score-label">{item.label}</span><div className="score-track"><div className="score-fill" style={{ width: `${Math.max(0, Math.min(100, item.score * 100))}%` }}/></div><span className="score-value">{(item.score * 100).toFixed(1)}%</span></div>)}<button className="secondary-button" onClick={reset}><RotateCcw size={15}/> Analyze another passage</button></div>}
        <div className="output-foot"><span className="status-dot muted"/> <span>{result?.configured ? 'Classifier connected' : 'Awaiting trained model API'}</span><span className="output-foot-right">Inference panel <ArrowDownRight size={13}/></span></div>
      </div>
    </section>

    <section className="how-section" id="how-it-works"><div className="section-intro"><div className="eyebrow"><Activity size={14}/> UNDER THE HOOD</div><h2>From raw text to<br/>a model prediction.</h2><p>A good detector learns patterns from labelled examples; it does not reliably identify a model from intuition alone.</p></div><div className="steps-grid"><article className="step-card"><span className="card-number">01</span><div className="card-icon"><ScanText size={20}/></div><h3>Extract signals</h3><p>Character and word n-grams can capture recurring phrasing, punctuation and formatting patterns.</p></article><article className="step-card"><span className="card-number">02</span><div className="card-icon"><Fingerprint size={20}/></div><h3>Learn fingerprints</h3><p>A classifier learns associations between these signals and the actual labels in the training data.</p></article><article className="step-card"><span className="card-number">03</span><div className="card-icon"><Target size={20}/></div><h3>Validate honestly</h3><p>Prompt-grouped validation helps measure generalization to unseen prompts and reduces leakage.</p></article></div></section>
    <section className="disclaimer"><div className="disclaimer-icon"><CircleHelp size={19}/></div><p><b>Research note</b><br/>Text-only attribution is probabilistic and can be unreliable for short, edited or similar-model responses. A score is not proof of authorship. <span>*Connect the official competition's actual label set before treating 18 as definitive.</span></p></section>
    <footer className="footer"><a className="brand footer-brand" href="#top"><span className="brand-mark"><Fingerprint size={17}/></span><span>origin<span className="brand-dot">.</span>ai</span></a><span>Built for transparent LLM attribution research.</span><a href="https://www.kaggle.com/competitions/name-that-llm" target="_blank" rel="noreferrer">Kaggle challenge <ArrowRight size={13}/></a></footer>
  </main>;
}
