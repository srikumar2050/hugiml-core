"""Shared HTML page shell for the HUGIML NLP example pages.

Both example pages (lending / card-default) use the same visual system; this
module holds the one copy of the CSS/JS shell and the small set of HTML
snippet builders (Q&A bubbles, metric chips, readout panels, tables) so
neither example script duplicates markup.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

CSS = """
  :root{
    --paper:#ECEFE6; --panel:#F8FAF3; --ink:#1C2420; --ink-soft:#5B6358; --rule:#C7CFC0;
    --teal:#2C6E63; --teal-soft:#E4EEEA; --amber:#B5651D; --amber-soft:#F3E7D8;
    --violet:#6B5CA0; --violet-soft:#ECE8F5;
  }
  *{box-sizing:border-box;}
  html{scroll-behavior:smooth;}
  body{margin:0; background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans', sans-serif; font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased;}
  @media (prefers-reduced-motion: reduce){ html{scroll-behavior:auto;} *{animation-duration:0.001ms !important; transition-duration:0.001ms !important;} }
  a{color:var(--teal);} :focus-visible{outline:2px solid var(--teal); outline-offset:2px;} .mono{font-family:'IBM Plex Mono', monospace;}

  header.masthead{padding:56px 24px 36px; max-width:880px; margin:0 auto; border-bottom:1px solid var(--rule);}
  .eyebrow{font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-soft); margin:0 0 14px;}
  .eyebrow .dot{color:var(--amber);}
  h1.title{font-family:'IBM Plex Serif', serif; font-weight:600; font-size:clamp(32px,5vw,46px); line-height:1.08; margin:0 0 16px; letter-spacing:-.01em;}
  .dek{font-size:17px; color:var(--ink-soft); max-width:620px; margin:0 0 22px;}
  .meta-row{display:flex; flex-wrap:wrap; gap:10px; font-family:'IBM Plex Mono', monospace; font-size:12.5px;}
  .meta-chip{border:1px solid var(--rule); padding:5px 10px; color:var(--ink-soft); background:var(--panel);}
  .meta-chip b{color:var(--ink); font-weight:600;}

  .stage-strip-wrap{position:sticky; top:0; z-index:50; background:var(--paper); border-bottom:1px solid var(--rule);}
  .stage-strip{max-width:880px; margin:0 auto; display:flex; overflow-x:auto; scrollbar-width:thin;}
  .stage-strip::-webkit-scrollbar{height:4px;} .stage-strip::-webkit-scrollbar-thumb{background:var(--rule);}
  .stage-tab{flex:0 0 auto; font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:.04em; color:var(--ink-soft); text-decoration:none; padding:11px 14px 9px; border-right:1px solid var(--rule); border-bottom:3px solid transparent; white-space:nowrap; transition:color .15s ease, border-color .15s ease, background .15s ease;}
  .stage-tab .num{color:var(--ink-soft); margin-right:6px;}
  .stage-tab:hover{background:var(--panel); color:var(--ink);}
  .stage-tab.active{color:var(--ink); border-bottom-color:var(--amber); background:var(--panel);}
  .stage-tab.active .num{color:var(--amber);}

  main{max-width:880px; margin:0 auto; padding:0 24px 100px;}
  section.stage{padding:52px 0 8px; border-bottom:1px solid var(--rule); scroll-margin-top:54px;}
  section.stage:last-of-type{border-bottom:none;}
  .stage-head{display:flex; align-items:baseline; gap:14px; margin-bottom:6px;}
  .stage-num{font-family:'IBM Plex Mono', monospace; font-size:13px; color:var(--amber); border:1px solid var(--amber); padding:2px 7px; flex:0 0 auto;}
  h2.stage-title{font-family:'IBM Plex Serif', serif; font-weight:600; font-size:24px; margin:0;}
  .stage-note{color:var(--ink-soft); font-size:14.5px; margin:6px 0 28px; max-width:600px;}

  .turn{margin-bottom:26px;}
  .bubble{display:flex; gap:14px; align-items:flex-start; margin-bottom:14px; opacity:0; transform:translateY(8px); animation:rise .5s ease forwards;}
  @keyframes rise{ to{opacity:1; transform:translateY(0);} }
  .tag{flex:0 0 auto; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono', monospace; font-size:12px; font-weight:600; border:1px solid currentColor; margin-top:2px;}
  .tag.q{color:var(--teal);} .tag.a{color:var(--amber);}
  .bubble-body{flex:1; min-width:0;}
  .bubble.q .bubble-body{background:var(--teal-soft); border-left:3px solid var(--teal); padding:12px 16px;}
  .bubble.a .bubble-body{background:var(--panel); border-left:3px solid var(--amber); padding:14px 16px;}
  .speaker-label{font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.06em; text-transform:uppercase; margin-bottom:5px;}
  .bubble.q .speaker-label{color:var(--teal);} .bubble.a .speaker-label{color:var(--amber);}
  .bubble p{margin:0 0 10px;} .bubble p:last-child{margin-bottom:0;}
  .bubble.a.boundary .bubble-body{background:var(--violet-soft); border-left:3px solid var(--violet);}
  .bubble.a.boundary .tag{color:var(--violet); border-color:var(--violet);}
  .bubble.a.boundary .speaker-label{color:var(--violet);}

  .readout{position:relative; border:1px solid var(--rule); background:#fff; margin:14px 0 4px; overflow-x:auto;}
  .readout-tag{position:absolute; top:-1px; left:-1px; font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:.06em; background:var(--ink); color:var(--paper); padding:3px 8px 2px;}
  .readout-tag.live::before{content:"\\25CF"; color:#7CC9A8; margin-right:5px;}
  .readout pre{margin:0; padding:26px 16px 14px; font-family:'IBM Plex Mono', monospace; font-size:12.5px; line-height:1.6; white-space:pre; overflow-x:auto;}
  table.readout-table{width:100%; border-collapse:collapse; font-family:'IBM Plex Mono', monospace; font-size:12.5px;}
  table.readout-table th, table.readout-table td{padding:7px 12px; border-bottom:1px solid var(--rule); text-align:right; white-space:nowrap;}
  table.readout-table th:first-child, table.readout-table td:first-child{text-align:left;}
  table.readout-table thead th{color:var(--ink-soft); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:.03em; border-bottom:1px solid var(--ink-soft);}
  table.readout-table tbody tr:last-child td{border-bottom:none;}
  table.readout-table tr.highlight td{background:var(--amber-soft);}
  table.readout-table tr.winner td{background:#E2EFE9; font-weight:600;}
  .pos{color:var(--teal);} .neg{color:#A1442E;}

  .chart-wrap{padding:30px 16px 18px;}
  .chart-caption{font-size:12px; color:var(--ink-soft); margin:10px 0 0; max-width:560px;}
  .chart-legend{display:flex; gap:16px; font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--ink-soft); margin-top:8px;}
  .chart-legend span{display:inline-flex; align-items:center; gap:6px;}
  .swatch{width:10px; height:10px; display:inline-block;}
  .swatch.amber{background:var(--amber);} .swatch.teal{background:var(--teal);} .swatch.ink{background:var(--ink);}

  .metric-row{display:flex; flex-wrap:wrap; gap:8px; margin:14px 0;}
  .metric{border:1px solid var(--rule); background:var(--panel); padding:8px 12px; font-family:'IBM Plex Mono', monospace; font-size:12.5px; min-width:96px;}
  .metric .k{display:block; color:var(--ink-soft); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px;}
  .metric .v{font-size:16px; font-weight:600;} .metric .v.up{color:var(--teal);} .metric .v.down{color:#A1442E;}

  .badge{display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:10.5px; letter-spacing:.04em; text-transform:uppercase; padding:2px 8px; border:1px solid var(--ink-soft); color:var(--ink-soft); margin-left:8px;}

  svg.chart{display:block; max-width:100%; height:auto; overflow:hidden;} svg.chart text{font-family:'IBM Plex Mono', monospace;}

  .findings{max-width:880px; margin:48px auto 0; padding:0 24px;}
  .findings-card{border:1px solid var(--ink-soft); background:var(--panel); padding:30px 28px;}
  .findings-card h3{font-family:'IBM Plex Serif', serif; font-size:21px; margin:0 0 6px;}
  .findings-card .findings-sub{color:var(--ink-soft); font-size:13.5px; margin:0 0 18px;}
  .findings-list{list-style:none; margin:0 0 18px; padding:0; display:grid; gap:12px;}
  .findings-list li{display:flex; gap:12px; align-items:baseline; font-size:14.5px; line-height:1.5;}
  .findings-list li::before{content:'\\2014'; color:var(--amber); font-weight:600; flex:0 0 auto;}
  .findings-close{margin:0; font-size:14px; color:var(--ink-soft); font-style:italic; border-top:1px solid var(--rule); padding-top:14px;}

  .closing{max-width:880px; margin:0 auto; padding:44px 24px 70px;}
  .closing-card{border:1px solid var(--rule); background:var(--panel); padding:26px 24px;}
  .closing-card h3{font-family:'IBM Plex Serif', serif; margin:0 0 10px; font-size:19px;}
  .chip-row{display:flex; flex-wrap:wrap; gap:10px; margin-top:16px;}
  .next-chip{border:1px solid var(--ink-soft); padding:8px 14px; font-size:13px; color:var(--ink); background:#fff;}

  @media (max-width:600px){
    header.masthead{padding:40px 18px 28px;} main{padding:0 18px 80px;} section.stage{padding:40px 0 8px;}
    .bubble{gap:10px;} .tag{width:24px; height:24px; font-size:11px;}
  }
"""

JS = """
(function(){
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.stage-tab'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('section.stage'));
  if(!('IntersectionObserver' in window) || tabs.length === 0) return;
  var map = {};
  sections.forEach(function(s){ map[s.id] = tabs.find(function(t){ return t.getAttribute('href') === '#' + s.id; }); });
  var current = null;
  var observer = new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        var tab = map[entry.target.id];
        if(tab && tab !== current){
          if(current) current.classList.remove('active');
          tab.classList.add('active'); current = tab;
          var stripEl = document.getElementById('stageStrip');
          stripEl.scrollTo({left: tab.offsetLeft - 24, behavior: 'smooth'});
        }
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
  sections.forEach(function(s){ observer.observe(s); });
})();
"""


def esc(value: object) -> str:
    return html.escape(str(value))


@dataclass
class Turn:
    """One question/answer pair. ``answer_html`` is pre-built HTML (may embed charts)."""

    question: str
    answer_html: str
    boundary: bool = False
    speaker_label: str = "Assistant"


@dataclass
class Stage:
    id: str
    num: str
    tab: str
    title: str
    note: str
    turns: list[Turn] = field(default_factory=list)


def metric_row(items: list[tuple[str, str, str]]) -> str:
    """items: (label, value, variant) where variant is '', 'up', or 'down'."""

    cells = "".join(
        f'<div class="metric"><span class="k">{esc(k)}</span>'
        f'<span class="v{(" " + v) if v else ""}">{esc(val)}</span></div>'
        for k, val, v in items
    )
    return f'<div class="metric-row">{cells}</div>'


def readout(tag: str, content_html: str, *, live: bool = True) -> str:
    live_cls = " live" if live else ""
    return f'<div class="readout"><span class="readout-tag{live_cls}">{esc(tag)}</span>{content_html}</div>'


def readout_pre(tag: str, text: str, *, live: bool = True) -> str:
    return readout(tag, f"<pre>{text}</pre>", live=live)


def readout_table(
    tag: str, headers: list[str], rows: list[list[str]], row_classes: list[str] | None = None
) -> str:
    row_classes = row_classes or [""] * len(rows)
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for cls, row in zip(row_classes, rows):
        cls_attr = f' class="{cls}"' if cls else ""
        body += f"<tr{cls_attr}>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    table = f'<table class="readout-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return readout(tag, table)


def chart_block(svg: str, caption: str, legend: str = "") -> str:
    return f'<div class="chart-wrap">{svg}{legend}<p class="chart-caption">{caption}</p></div>'


def render_turn(turn: Turn) -> str:
    boundary_cls = " boundary" if turn.boundary else ""
    return f"""
    <div class="turn">
      <div class="bubble q">
        <div class="tag q">Q</div>
        <div class="bubble-body"><div class="speaker-label">You</div><p>{esc(turn.question)}</p></div>
      </div>
      <div class="bubble a{boundary_cls}">
        <div class="tag a">A</div>
        <div class="bubble-body"><div class="speaker-label">{esc(turn.speaker_label)}</div>{turn.answer_html}</div>
      </div>
    </div>"""


def render_stage(stage: Stage) -> str:
    turns_html = "\n".join(render_turn(t) for t in stage.turns)
    return f"""
  <section class="stage" id="{stage.id}">
    <div class="stage-head"><span class="stage-num">{esc(stage.num)}</span><h2 class="stage-title">{esc(stage.title)}</h2></div>
    <p class="stage-note">{esc(stage.note)}</p>
{turns_html}
  </section>"""


def render_findings(heading_sub: str, bullets: list[str], closing: str) -> str:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""
<div class="findings">
  <div class="findings-card">
    <h3>Key findings from this session</h3>
    <p class="findings-sub">{esc(heading_sub)}</p>
    <ul class="findings-list">{items}</ul>
    <p class="findings-close">{closing}</p>
  </div>
</div>"""


def render_closing(chips: list[str]) -> str:
    chip_html = "".join(f'<span class="next-chip">{esc(c)}</span>' for c in chips)
    return f"""
<div class="closing">
  <div class="closing-card">
    <h3>Where to go from here</h3>
    <p style="color:var(--ink-soft); font-size:14px; max-width:560px; margin:0;">The same conversation works on any file you bring -- just point at the table and describe what your team needs to act on.</p>
    <div class="chip-row">{chip_html}</div>
  </div>
</div>"""


def render_page(
    *,
    title: str,
    eyebrow: str,
    dek: str,
    meta_chips: list[tuple[str, str]],
    stages: list[Stage],
    findings_html: str,
    closing_html: str,
) -> str:
    tabs = "".join(
        f'<a class="stage-tab" href="#{s.id}"><span class="num">{esc(s.num)}</span>{esc(s.tab)}</a>'
        for s in stages
    )
    sections = "\n".join(render_stage(s) for s in stages)
    meta = "".join(f'<span class="meta-chip">{esc(k)}: <b>{esc(v)}</b></span>' for k, v in meta_chips)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<header class="masthead">
  <p class="eyebrow">{eyebrow}</p>
  <h1 class="title">{esc(title)}</h1>
  <p class="dek">{dek}</p>
  <div class="meta-row">{meta}</div>
</header>

<div class="stage-strip-wrap">
  <nav class="stage-strip" id="stageStrip" aria-label="Session stages">{tabs}</nav>
</div>

<main>{sections}
</main>
{findings_html}
{closing_html}

<script>{JS}</script>

</body>
</html>
"""
