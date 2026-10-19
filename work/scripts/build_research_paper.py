"""Build a public-safe paper from executed notebook outputs, without gated data.

Run from the repository root: python work/scripts/build_research_paper.py
This reconstructs the report, not the underlying warehouse analysis.
"""
from pathlib import Path
from html.parser import HTMLParser
from html import escape
import json, hashlib, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
FIG = ROOT / 'work/figures'
OUT = ROOT / 'work/outputs'
for folder in [DOCS, FIG, OUT]:
    folder.mkdir(parents=True, exist_ok=True)
REPO = 'https://github.com/omsawarkar/omsawarkar'

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=[]; self.cell=None
    def handle_starttag(self, tag, attrs):
        if tag=='tr': self.row=[]
        if tag in ('th','td'): self.cell=''
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in ('th','td') and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell=None
        if tag=='tr' and self.row: self.rows.append(self.row)

def tables(name):
    p = ROOT/'work/notebooks'/f'{name}.ipynb'
    notebook = json.loads(p.read_text())
    found=[]
    for cell in notebook['cells']:
        for output in cell.get('outputs',[]):
            assert output.get('output_type')!='error', f'Error output in {name}'
            raw=output.get('data',{}).get('text/html')
            if raw:
                parser=TableParser(); parser.feed(''.join(raw)); found.append(parser.rows)
    return found

def pick(items, *headers):
    return next(t for t in items if all(h in t[0] for h in headers))

validation_tables=tables('w06_validation_audit')
comparison=pick(validation_tables,'stage','method','average_precision','roc_auc')
folds=pick(validation_tables,'stage','test_clients','average_precision')
queue_tables=tables('w07_action_playbook')
top10=pick(queue_tables,'rank','score','impressions','clicks','action')
playbook=json.loads((OUT/'action_playbook_receipt.json').read_text())
assert playbook['queue_items']==6607 and playbook['positive_score_items']==3280

def dictionaries(table):
    return [dict(zip(table[0], row)) for row in table[1:]]

results=dictionaries(comparison)
fold_results=dictionaries(folds)
top_rows=dictionaries(top10)
assert len(results)==4 and len(top_rows)==10
assert results[2]['items']=='6531'
assert results[2]['average_precision']=='0.8566'
assert results[3]['average_precision']=='0.8025'

metrics={'provenance':'Extracted from saved executed notebook outputs; not a new model run',
    'comparison':[{k:v for k,v in row.items() if k} for row in results],
    'folds':[{k:v for k,v in row.items() if k} for row in fold_results],
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'work/notebooks/w05_model.ipynb',ROOT/'work/notebooks/w06_validation_audit.ipynb',ROOT/'work/notebooks/w07_action_playbook.ipynb']}}
(OUT/'paper_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'axes.edgecolor':'#cbd5e1','text.color':'#152936','axes.labelcolor':'#152936'})
colors=['#087e8b','#d56b36']
fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for ax,key,title in zip(axs,['average_precision','roc_auc'],['Average precision','ROC-AUC']):
    x=[0,1]; width=.32
    for j,(name,c) in enumerate(zip(['Baseline','Logistic regression'],colors)):
        vals=[float(results[j][key]),float(results[j+2][key])]
        bars=ax.bar([i+(j-.5)*width for i in x],vals,width,color=c,label=name)
        ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
    ax.set_xticks(x,['Original split\n246 items','Pooled grouped folds\n6,531 items'])
    ax.set_ylim(0,1); ax.set_title(title,loc='left',fontweight='bold'); ax.set_ylabel('Score (0–1)')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].legend(loc='lower left',frameon=False,fontsize=9)
fig.savefig(FIG/'validation_comparison.svg',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2.5),layout='constrained')
sizes=[int(r['items']) for r in fold_results[::2]]
bars=ax.barh([f'Fold {i}' for i in range(1,6)],sizes,color='#087e8b')
ax.bar_label(bars,padding=5);ax.set_xlim(0,max(sizes)*1.2);ax.invert_yaxis()
ax.set_xlabel('Validation items');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.savefig(FIG/'fold_sizes.svg',bbox_inches='tight');plt.close(fig)

def inline_svg(name):
    raw=(FIG/name).read_text(); raw=raw[raw.index('<svg'):]
    prefix=name.replace('.svg','')+'_'
    for identifier in set(re.findall(r'id="([^"]+)"',raw)):
        raw=raw.replace('id="'+identifier+'"','id="'+prefix+identifier+'"')
        raw=raw.replace('#'+identifier+'"','#'+prefix+identifier+'"').replace('#'+identifier+')','#'+prefix+identifier+')')
    return raw
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link_notebook(name,label):
    return f'<a href="{REPO}/blob/main/work/notebooks/{name}.ipynb">{label}</a>'

abstract=('Can a simple peer-gap score help an editor prioritize pages for investigation, and does logistic regression improve its ranking? '
 'This study uses the March 2026 warehouse partition containing 9,841,378 daily rows, with 6,531 eligible items from 21 clients in the later-outcome comparison. '
 'Early-March measurements define a transparent baseline and five model inputs, while late-March CTR defines a diagnostic outcome evaluated with client-grouped splits. '
 'Both methods achieved Precision@10 of 1.00 in pooled grouped evaluation, but the baseline exceeded the model on average precision (0.8566 versus 0.8025). '
 'The resulting historical playbook supports human investigation of below-peer CTR, without establishing that content updates improve traffic.')

sections=[]
def section(id,title,body): sections.append((id,title,body))
section('problem','01 / The decision: which pages deserve a closer look?', '''
<p>An editor has time to inspect ten pages, not thousands. A poor recommendation wastes editorial effort; a missed issue delays a useful investigation. The output should therefore be an understandable review order, with reasons and a human decision before changes.</p>
<p>The initial lane concerned content-refresh opportunities. The implemented analysis narrows this to search click-through rate (CTR), because the daily table does not contain update history. It cannot diagnose staleness. The unit of recommendation is one pseudonymized client–content pair.</p>''')
section('data','02 / Data: one month, a restricted eligible population', '''
<p>The source is the FlyRank Internship Warehouse release v20260703. Its full daily table contains 78,835,655 rows; this analysis reads only <code>fact_content_daily_performance/month=2026-03/data_0.parquet</code>. It does not analyze the entire release.</p>'''+table(['Check','Saved result'],[
['March source rows','9,841,378; 1–31 March 2026; 55 clients'],['Source grain','Date × client × content; zero duplicate key groups and missing keys'],
['Availability in full March','GSC: 3,611,061 rows; GA4: 413,966; both: 364,347'],
['Early window','1–20 March; both availability flags TRUE; ≥10 observed days; positive impressions'],
['Historical queue','6,607 items; 6,565 meet further baseline scoring requirements'],
['Comparison population','6,531 items from 21 clients; 4,289 positive and 2,242 negative labels']])+'''
<p>GSC measures search performance; GA4 measures site visits and engagement. Availability flags prevent zero-filled unavailable measurements from being read as real zeros. Initial filters exclude items entirely; the queue’s insufficient-data category covers only remaining items that fail subsequent scoring checks.</p>
<p>Dimensions and the query-level table were not joined. Private client names, page addresses, raw queries, and pseudonymous identifiers are omitted from this page. June remains unused; no sealed-test evaluation is claimed.</p>''')
section('method','03 / Methodology: a baseline before a learned model', '''
<h3>Baseline and signal checks</h3><p>The early-window bucket comparisons showed lower pooled CTR in worse position groups: 0.778%, 0.702%, 0.547%, and 0.205% across positions 1–3, &gt;3–10, &gt;10–20, and &gt;20. Higher-volume groups also had lower median CTR within each position group, opposite to the tested hypothesis. These descriptive checks motivated position-and-volume peers, not causal claims.</p>
<p>Position is impression-weighted across observed days, excluding nonpositive position values. Volume groups are 100–499, 500–1,999, and 2,000+ impressions. A peer group must contain at least 20 items. Each item contributes to its own group median.</p>
<div class="formula">Review score = impressions × max(peer median CTR − item CTR, 0)</div>
<p>All rates in this formula are fractions. The score is a peer-gap index, not recoverable clicks. Noneligible queue items receive zero; ties are resolved by impressions and stable identifier ordering.</p>
<h3>Five inputs and one later-outcome proxy</h3><p>The model inputs are early-window impressions, clicks, CTR, sessions, and engagement rate. CTR is total clicks / impressions; engagement rate is total engaged sessions / sessions. Undefined denominators remain missing. The pipeline applies training-only median imputation, log(1+x), standardization, and logistic regression (seed 42; maximum 2,000 iterations).</p>
<p>The binary outcome is 1 if CTR during <strong>21–31 March</strong> is below the item’s frozen early-window peer median; otherwise it is 0. Evaluation requires ≥5 GSC-available outcome days and ≥100 outcome impressions. This is a later observed CTR proxy, distinct from Week 3’s click-decline diagnostic and from the original human-review objective.</p>
<h3>Validation and leakage boundaries</h3><p>The original GroupShuffleSplit held out six clients (246 items); five-fold GroupKFold subsequently produced out-of-fold predictions for all 6,531 items. Preprocessing is fitted inside each training split. IDs, peer benchmarks, baseline scores, product flags, and late measurements are excluded from the five model inputs.</p>
<p>The frozen benchmark includes held-out clients’ <em>early</em> observations. Thus this is a known-reference-population experiment, not fully inductive prediction for unseen clients. Outcome-coverage selection is retrospective. The original 246-item test set was already inspected, so the expanded evaluation is exploratory rather than a fresh untouched holdout.</p>''')
metric_rows=[]
for r in results:
    metric_rows.append([r['stage'].replace('After: pooled out-of-fold','Pooled grouped folds').replace('Before: original split','Original split'),r['method'],r['items'],r['base_rate'],r['precision_at_10'],r['precision_at_20'],r['average_precision'],r['roc_auc']])
section('results','04 / The original model advantage did not persist', '''
<p>On the original split, logistic regression improved average precision and ROC-AUC, but tied Precision@10 and lost at Precision@20. In the expanded pooled evaluation, it tied the top-of-list metrics while scoring below the baseline on both broader summaries.</p>'''+table(['Evaluation','Method','Items','Base rate','P@10','P@20','Avg. precision','ROC-AUC'],metric_rows)+
 '<figure>'+inline_svg('validation_comparison.svg')+'<figcaption>Figure 1. Compare methods within each evaluation population. The baseline is stronger on both pooled summaries; changes between populations are not controlled estimates of a validation effect.</figcaption></figure>'+'''
<p>Precision@10 is the positive-label share of the ten highest scores, with deterministic tie handling. Average precision summarizes precision over recall; ROC-AUC measures discrimination. Neither validates whether an editor should change content. A pooled base rate of 0.6567 is high, and ten selected items give a coarse, uncertain estimate.</p>
<details><summary>Inspect all five grouped folds</summary>'''+table(['Fold','Method','Items','Clients','Base rate','P@10','P@20','Avg. precision','ROC-AUC'],[[r['stage'],r['method'],r['items'],r['test_clients'],r['base_rate'],r['precision_at_10'],r['precision_at_20'],r['average_precision'],r['roc_auc']] for r in fold_results])+'''</details>
<figure>'''+inline_svg('fold_sizes.svg')+'''<figcaption>Figure 2. Fold 1 contains 4,273 items from a single client. Pooled metrics are therefore strongly influenced by one client, and model probabilities come from separately fitted pipelines.</figcaption></figure>
<h3>Errors matter more than a perfect top-ten number</h3>
<p>At threshold 0.5 there were 1,979 out-of-fold errors. Three confident false positives had zero early clicks from 2,061, 2,034, and 2,337 impressions. Their predicted probabilities were about 0.999, yet later CTR rose above the frozen benchmark. Zero early clicks did not guarantee persistent below-benchmark CTR. These are aggregate case descriptions, not identifiable pages.</p>
<p>On the original fit, clicks, impressions, and sessions had the largest absolute standardized coefficients. Clicks, impressions, and CTR are mathematically related, so individual coefficients do not provide causal explanations. We retain the baseline rather than claim a consistent model advantage.</p>''')
section('limits','05 / Limitations: what this evidence cannot establish', '''
<ul><li><strong>No intervention evidence.</strong> Later below-peer CTR is not a label for update benefit. No actual page reviews or content changes were performed.</li>
<li><strong>Restricted population.</strong> Requiring GSC and GA4 availability, early coverage, and later coverage selects a small subset. Missing rows are not zero activity.</li>
<li><strong>Shared reference information.</strong> Peer medians use all eligible early clients; broad buckets mix query, device, country, intent, and client differences.</li>
<li><strong>Short windows and unequal histories.</strong> Observation-day totals, weekday composition, low click counts, and reporting latency can affect results. Seasonal or long-term generalization was not tested.</li>
<li><strong>Exploratory validation.</strong> One large client dominates, pooled probabilities may differ in calibration, and no uncertainty intervals or significance tests were produced. Reusing already inspected March outcomes does not create a sealed test.</li>
<li><strong>Proxy alignment.</strong> The label reuses an early benchmark, so apparent predictive ability may reflect persistence of relative CTR rather than actionable content defects. The original editorial Precision@10 remains unmeasured.</li></ul>''')
section('actions','06 / A ranked queue becomes a human-review playbook', '''
<p>The historical queue contains <strong>3,280 review candidates</strong>, <strong>3,285 monitoring items</strong>, and <strong>42 items requiring more evidence</strong>. Every record remains NOT_REVIEWED. These are rule-defined categories, not learned content clusters.</p>'''+table(['Evidence / reason','Action'],[
['BELOW_PEER_CTR','Check measurement quality, search appearance, and intent; propose a specific change only after page inspection.'],
['NO_CTR_GAP','Monitor. Zero score does not establish page health or low business importance.'],
['INSUFFICIENT_DATA','Collect or verify evidence before assigning review priority.']])+'''
<details><summary>Top ten historical candidates, without identifiers</summary>'''+table(['Rank','Score','Impressions','Clicks','Position group'],[[r['rank'],f"{float(r['score']):.3f}",f"{int(float(r['impressions'])):,}",int(float(r['clicks'])),r['position_bucket']] for r in top_rows])+'''<p>All ten share BELOW_PEER_CTR and require human inspection. Sparse clicks, mismatched peers, query mix, or tracking issues could make any recommendation unsuitable.</p></details>
<h3>Review, approval, and cost</h3><p>Verify measurements and peer suitability, inspect authorized page content, document the issue, and assign APPROVE_SPECIFIC_CHANGE, INVESTIGATE_FURTHER, MONITOR, or REJECT_RECOMMENDATION. A content owner approves implementation. Never automate publishing, deletion, redirects, or indexing changes from this score.</p>
<p>A proposed pilot reviews ten candidates at an initial 15 minutes each (150 minutes). This is a planning assumption, not a measured cost. Record actual time and confirmed actionable issues; the provisional goal of seven actionable cases out of ten is not an achieved result.</p>
<h3>Monitoring and reconsideration</h3><p>Check coverage and schema before each weekly refresh. Pause on duplicates, missing keys, invalid measurements, or an unexplained &gt;20% relative change in eligible counts. Two pilot batches below the proposed seven-of-ten yield trigger investigation. Thresholds are provisional. For approved changes, inspect 30- and 60-day follow-up with comparable untreated pages where possible; a simple before/after change is not causal proof.</p>
<p>This baseline has no fitted model to retrain. Revisit rules through a documented validation process. Reconsider ML only when independent labels and repeated comparisons establish useful gains.</p>''')
section('reproduce','07 / Reproducibility and evidence trail', '''
<p>The figures and tables on this page are rebuilt from the saved executed outputs in the uploaded repository, not a new authenticated warehouse run. Source hashes and extracted metrics are recorded in <a href="'''+REPO+'''/blob/main/work/outputs/paper_metrics.json">paper_metrics.json</a>.</p>
<ol><li>Clone the repository and install <code>huggingface_hub duckdb pandas numpy scikit-learn matplotlib</code>.</li>
<li>For a fresh analysis, obtain the gated dataset permission and store a Read token as the Colab secret <code>HF_TOKEN</code>. Never put its value in code.</li>
<li>Run the data-contract, baseline, model, validation, and playbook notebooks in order, top to bottom. Raw queue CSVs remain outside GitHub.</li>
<li>Run <code>python work/scripts/build_research_paper.py</code> from the repository root, or run the capstone notebook, to rebuild this page from executed outputs.</li></ol>
<p>Seeds are 42 where randomization is used. The Week 5 output records scikit-learn 1.6.1; other analysis packages were installed without full version pins, so a bit-for-bit training environment was not captured. The report build has no gated-data dependency. Public artifact links follow.</p>
<div class="links">'''+''.join(link_notebook(n,l) for n,l in [('w03_data_contract','Data contract'),('w04_baseline_score','Baseline'),('w05_model','Model'),('w06_validation_audit','Validation audit'),('w07_action_playbook','Playbook'),('capstone','Capstone assembly')])+'''</div>
<p>Run receipts: <a href="'''+REPO+'''/blob/main/work/outputs/baseline_run_receipt.json">baseline</a> · <a href="'''+REPO+'''/blob/main/work/outputs/action_playbook_receipt.json">playbook</a>. <a href="'''+REPO+'''">Repository and source code</a>.</p>''')
section('credit','08 / Acknowledgments & data credit', '''
<p>Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>. The warehouse provides pseudonymized search and analytics observations for educational research. Access and reuse remain subject to its data-use terms.</p>
<p>Om Sawarkar completed the notebook workflow with AI assistance for code, explanations, and report drafting. The page reports saved notebook evidence and does not imply peer review, independent editorial validation, or endorsement by FlyRank.</p>
<p>Research context: <a href="flyrank-seo-research-march-2026.pdf">FlyRank, The State of AI-Driven SEO in Numbers (March 2026)</a>. Its portfolio is separate from this March warehouse experiment. Presentation references: <a href="https://distill.pub/2020/communicating-with-interactive-articles/">Distill’s interactive-article discussion</a> informed optional details and figure captions; <a href="https://joss.theoj.org/papers/10.21105/joss.00107">the JOSS ReproZip page</a> informed a clear evidence-and-code trail. These are design references, not evidence for the SEO results.</p>''')

css='''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;background:#f7f7f2;color:#182e39;font:17px/1.75 system-ui,-apple-system,Segoe UI,sans-serif}a{color:#076978;text-underline-offset:4px}a:focus-visible,summary:focus-visible{outline:3px solid #d56b36;outline-offset:5px}.top{border-bottom:1px solid #d9dfdc;padding:14px 5%;display:flex;justify-content:space-between;gap:20px;font-size:13px;letter-spacing:.06em;background:#f7f7f2}.wrap{max-width:1180px;margin:auto;padding:0 35px}.hero{padding:72px 0 48px;border-bottom:1px solid #c8d3cf}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.17em;color:#087e8b;font-weight:750}h1{font-family:Georgia,serif;font-size:clamp(38px,5vw,67px);line-height:1.08;letter-spacing:-.025em;max-width:950px;margin:20px 0 26px}.subtitle{font-size:21px;max-width:800px;color:#48616a}.meta{font-size:14px;color:#48616a}.tags{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}.tags span{border:1px solid #becfca;border-radius:30px;padding:5px 13px;font-size:12px}.layout{display:grid;grid-template-columns:190px minmax(0,1fr);gap:60px}aside{padding-top:50px}aside nav{position:sticky;top:28px;display:grid;gap:12px;font-size:13px}aside a{text-decoration:none;color:#46616b}article{min-width:0;padding-bottom:65px}section{padding:42px 0 10px;border-bottom:1px solid #d9dfdc}h2{font:700 30px/1.25 Georgia,serif;margin:0 0 24px}h3{font-size:18px;margin-top:28px}p{margin:14px 0}.abstract{background:#e9f0eb;padding:28px 30px;margin-top:40px;border-left:4px solid #087e8b}.abstract h2{font-size:24px}.callout{background:#173e45;color:#fff;padding:24px 30px;margin:26px 0}.callout p{margin:0}.table-wrap{overflow-x:auto;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.5}th{text-align:left;background:#e9efea;color:#23444b;padding:12px;white-space:nowrap}td{padding:12px;border-bottom:1px solid #d6dfda;vertical-align:top}tbody tr:nth-child(even){background:#f1f3ee}figure{margin:32px 0;background:#fff;border:1px solid #dbe1dc;padding:18px}figure svg{width:100%;height:auto;display:block}figcaption{font-size:13px;color:#51656b;padding:14px 4px 0;line-height:1.6}.formula{font:15px/1.65 ui-monospace,monospace;background:#e9efea;padding:20px;border-left:3px solid #087e8b;overflow-wrap:anywhere}code{font-size:.83em;overflow-wrap:anywhere}li{margin:10px 0}details{border:1px solid #c9d6d0;padding:14px 18px;margin:24px 0}summary{cursor:pointer;font-weight:650;font-size:15px}.links{display:flex;flex-wrap:wrap;gap:10px}.links a{border:1px solid #b6c9c3;padding:6px 11px;font-size:13px;text-decoration:none}footer{border-top:1px solid #c8d3cf;font-size:12px;padding:25px 0;color:#587079}@media(max-width:800px){.wrap{padding:0 20px}.hero{padding-top:45px}.layout{display:block}aside{padding-top:22px}aside nav{position:static;display:flex;overflow-x:auto;gap:18px;padding-bottom:10px}aside a{white-space:nowrap}.abstract{margin-top:20px;padding:22px}section{padding-top:32px}h2{font-size:26px}body{font-size:16px}.top{padding:12px 20px}}@media print{aside,.top{display:none}.layout{display:block}.hero{padding-top:0}body{background:white;font-size:11pt}details{display:block}figure,table{break-inside:avoid}a{color:inherit}.wrap{padding:0}}'''
nav=''.join(f'<a href="#{i}">{escape(t.split(" / ")[0])} · {escape(t.split(" / ")[1].split(":")[0])}</a>' for i,t,b in sections)
html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A client-grouped comparison of a peer CTR baseline and logistic regression, with an editorial action playbook."><title>Which pages should an editor review first? — Om Sawarkar</title><style>'''+css+'''</style></head><body><a href="#abstract" style="position:absolute;left:-10000px" onfocus="this.style.left='10px'" onblur="this.style.left='-10000px'">Skip to paper</a><header class="top"><span>OM SAWARKAR / RESEARCH NOTE</span><a href="'''+REPO+'''">View source ↗</a></header><main class="wrap"><header class="hero"><div class="eyebrow">Search intelligence · Internship capstone · October 2026</div><h1>Which pages should an editor review first?</h1><p class="subtitle">A simple CTR baseline, a learned ranking, and what broader validation changed.</p><p class="meta">Om Sawarkar · March 2026 data · Exploratory observational analysis</p><div class="tags"><span>9.84M source daily rows</span><span>6,531 comparison items</span><span>21 clients in validation</span><span>Human review required</span></div></header><div class="layout"><aside><nav aria-label="Paper sections">'''+nav+'''</nav></aside><article><div class="abstract" id="abstract"><h2>Abstract</h2><p>'''+abstract+'''</p></div><div class="callout"><p><strong>Decision:</strong> retain the transparent baseline. Broader validation did not establish a consistent model advantage for the review queue.</p></div>'''+''.join(f'<section id="{i}"><h2>{t}</h2>{b}</section>' for i,t,b in sections)+'''</article></div><footer>Historical research demonstration · No automated content changes · No private page data published</footer></main></body></html>'''
(DOCS/'index.html').write_text(html,encoding='utf-8')

report='# Which pages should an editor review first?\n\nOm Sawarkar\n\n## Abstract\n\n'+abstract+'\n\n'
for _,title,body in sections:
    # HTML is retained in this Markdown companion to preserve exact tables.
    report+='## '+title+'\n\n'+body+'\n\n'
(ROOT/'work/capstone_report.md').write_text(report,encoding='utf-8')
print('Paper rebuilt from saved executed outputs.')
print('Original comparison: 246 items; pooled comparison: 6,531 items.')
print('Wrote docs/index.html, work/capstone_report.md, two figures and paper_metrics.json.')
