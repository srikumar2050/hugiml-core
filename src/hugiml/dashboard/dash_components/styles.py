"""Theme tokens and CSS for HUGIML Governance Studio."""

from __future__ import annotations

THEMES = ["Ocean", "Forest", "Dark"]

_TOKENS = {
    "Ocean": {
        "a": "#2563eb",
        "a2": "#06b6d4",
        "bg": "#ffffff",
        "sf": "#ffffff",
        "sa": "#f8faff",
        "tx": "#0f172a",
        "mu": "#64748b",
        "bd": "rgba(37,99,235,0.22)",
        "bs": "rgba(128,128,128,0.12)",
        "cb": "rgba(37,99,235,0.07)",
        "ha": "rgba(37,99,235,0.14)",
        "hb": "rgba(6,182,212,0.10)",
        "sh": "0 4px 20px rgba(14,116,144,0.08)",
        "gr": "#16a34a",
        "rd": "#dc2626",
        "am": "#d97706",
    },
    "Forest": {
        "a": "#059669",
        "a2": "#84cc16",
        "bg": "#ffffff",
        "sf": "#ffffff",
        "sa": "#f0fdf4",
        "tx": "#14532d",
        "mu": "#6b7280",
        "bd": "rgba(5,150,105,0.22)",
        "bs": "rgba(128,128,128,0.12)",
        "cb": "rgba(5,150,105,0.07)",
        "ha": "rgba(5,150,105,0.14)",
        "hb": "rgba(132,204,22,0.10)",
        "sh": "0 4px 20px rgba(5,122,85,0.08)",
        "gr": "#15803d",
        "rd": "#dc2626",
        "am": "#d97706",
    },
    "Dark": {
        "a": "#60a5fa",
        "a2": "#a78bfa",
        "bg": "#0f172a",
        "sf": "#1e293b",
        "sa": "#0f172a",
        "tx": "#f1f5f9",
        "mu": "#94a3b8",
        "bd": "rgba(96,165,250,0.28)",
        "bs": "rgba(148,163,184,0.15)",
        "cb": "rgba(30,41,59,0.85)",
        "ha": "rgba(15,23,42,0.90)",
        "hb": "rgba(30,41,59,0.75)",
        "sh": "0 6px 28px rgba(0,0,0,0.30)",
        "gr": "#4ade80",
        "rd": "#f87171",
        "am": "#fbbf24",
    },
}


def get_tokens(theme="Ocean"):
    return _TOKENS.get(theme, _TOKENS["Ocean"])


_THEME_STATE_CSS = (
    ".theme-dark"
    "{background:var(--bg)!important;color:var(--tx)!important}"
    ".theme-dark .hug-hdr,.theme-dark .hug-hero,"
    ".theme-dark .hug-ctrl,.theme-dark .hug-upl,"
    ".theme-dark .hug-tabrow"
    "{background:var(--sf)!important;border-color:var(--bd)!important}"
    ".theme-dark .card,.theme-dark .mc,"
    ".theme-dark .setup-card,.theme-dark .results-side-card"
    "{background:var(--sf)!important;border-color:var(--bd)!important;color:var(--tx)!important}"
    ".theme-dark .form-control,.theme-dark .form-select,"
    ".theme-dark .Select-control,.theme-dark .Select-menu-outer"
    "{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}"
    ".theme-dark .Select-value-label,.theme-dark .Select-placeholder,"
    ".theme-dark .Select-input>input,.theme-dark .VirtualizedSelectOption"
    "{color:var(--tx)!important}"
    ".theme-dark .VirtualizedSelectFocusedOption"
    "{background:var(--cb)!important;color:var(--tx)!important}"
    ".theme-dark .accordion-item,.theme-dark .accordion-button,"
    ".theme-dark .accordion-button:not(.collapsed),.theme-dark .tab-content"
    "{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}"
    ".theme-dark .text-muted{color:var(--mu)!important}"
    ".theme-dark .info-b,.theme-dark .sn,"
    ".theme-dark .parameter-pre,.theme-dark .tree-pre"
    "{background:var(--cb)!important;color:var(--tx)!important}"
    ".theme-dark .nav-tabs{border-bottom-color:var(--bd)!important}"
)


_STATIC_CSS = (
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');"
    "*{box-sizing:border-box}"
    "html,body{margin:0;padding:0;height:100%;font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--tx)}"
    ".hug-wrap{display:flex;flex-direction:column;height:100vh;overflow:hidden}"
    ".hug-hdr{flex-shrink:0;background:var(--sf);border-bottom:2px solid var(--bd);box-shadow:0 2px 10px rgba(0,0,0,0.06)}"
    ".hug-hero{position:relative;padding:14px 24px 12px;background:radial-gradient(ellipse 60% 55% at 0 0,var(--ha),transparent),radial-gradient(ellipse 40% 35% at 102% 0,var(--hb),transparent),var(--sf)}"
    ".hug-hero::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:var(--bs)}"
    ".hero-ey{font-size:.60rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--a);display:block;margin-bottom:4px}"
    ".hero-h1{font-size:clamp(1.2rem,1.7vw,1.6rem);font-weight:900;letter-spacing:-0.04em;line-height:1.1;color:var(--tx);margin:0 0 4px}"
    ".hero-p{margin:0;opacity:.70;line-height:1.55;font-size:.83rem;color:var(--tx)}"
    ".chip{display:inline-flex;align-items:center;background:var(--cb);border:1px solid var(--bd);border-radius:5px;padding:2px 7px;font-size:.61rem;font-weight:600;letter-spacing:.04em;font-family:'JetBrains Mono',monospace;color:var(--tx)}"
    ".chip-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}"
    ".hug-ctrl{display:flex;align-items:center;gap:8px;padding:7px 24px;background:var(--sa);border-bottom:1px solid var(--bs);min-height:44px}.primary-nav{flex-wrap:nowrap}.workspace-switch{display:flex;gap:8px;align-items:center}"
    ".ctrl-sep{width:1px;height:20px;background:var(--bs);flex-shrink:0}"
    ".hug-upl{display:flex;align-items:center;gap:10px;padding:7px 24px;background:var(--sf);border-bottom:1px solid var(--bs);flex-wrap:wrap}"
    ".hug-tabrow{display:flex;align-items:stretch;padding:0;background:var(--sf);border-bottom:2px solid var(--bs);overflow-x:auto;scrollbar-width:none;min-height:40px}.hug-tabrow>div{width:100%}#gov-nav-row{padding:0 20px}"
    ".hug-tabrow::-webkit-scrollbar{display:none}"
    ".tab-btn{font-size:.78rem;font-weight:600;padding:0 14px;cursor:pointer;color:var(--mu);border:none;border-bottom:2px solid transparent;background:transparent;white-space:nowrap;margin-bottom:-2px;transition:color .12s,border-color .12s;display:flex;align-items:center}"
    ".tab-btn:hover{color:var(--a)}"
    ".tab-btn.act{color:var(--a);border-bottom-color:var(--a);font-weight:700}"
    ".hug-content{flex:1;overflow-y:auto;padding:22px 28px}"
    ".ws-btn{font-size:.78rem;font-weight:600;padding:4px 13px;border-radius:7px;border:1px solid var(--bs);background:transparent;color:var(--tx);cursor:pointer;transition:all .12s;white-space:nowrap}"
    ".ws-btn:hover{background:var(--cb)}"
    ".ws-btn.act{background:var(--a);color:#fff;border-color:var(--a)}"
    ".theme-lbl{cursor:pointer;font-size:.66rem;font-weight:700;letter-spacing:.10em;text-transform:uppercase;color:var(--a);padding:3px 9px;border-radius:5px;border:1px solid var(--bd);background:var(--cb);user-select:none;white-space:nowrap}"
    ".theme-select{width:118px;min-width:118px;font-size:.72rem}.theme-select .Select-control{height:29px;min-height:29px;background:var(--cb);border-color:var(--bd)}.theme-select .Select-placeholder,.theme-select .Select-value{line-height:27px!important}.theme-select .Select-input{height:27px}.theme-select .Select-arrow-zone{padding-right:6px}"
    ".btn-fit{background:var(--a)!important;border:none!important;color:#fff!important;font-weight:700!important;border-radius:8px!important;padding:5px 16px!important;cursor:pointer;font-size:.83rem;white-space:nowrap;transition:opacity .15s}"
    ".btn-fit:hover{opacity:.88}"
    ".fit-badge{display:inline-flex;align-items:center;gap:5px;background:var(--cb);border:1px solid var(--bd);border-radius:20px;padding:3px 11px;font-size:.74rem;font-weight:600;color:var(--a);white-space:nowrap}"
    ".fit-dot{width:7px;height:7px;border-radius:50%;background:var(--gr);flex-shrink:0}"
    ".hug-spacer{flex:1}"
    ".mc{background:var(--sf);border:1px solid var(--bs);border-top:3px solid var(--a);border-radius:10px;padding:12px 14px 10px;min-height:72px}"
    ".mc-v{font-size:clamp(1.1rem,1.5vw,1.7rem);font-weight:900;letter-spacing:-0.03em;line-height:1.08;color:var(--tx)}"
    ".mc-l{font-size:.60rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--mu);margin-bottom:3px}"
    ".sn{border:1px solid var(--bs);border-left:3px solid var(--a);background:var(--cb);border-radius:0 8px 8px 0;padding:8px 14px;margin:4px 0 14px}"
    ".sn p{margin:0;font-size:.84rem;line-height:1.55;color:var(--tx);opacity:.82}"
    ".pg-h{font-size:1.10rem;font-weight:800;letter-spacing:-0.025em;color:var(--tx)}"
    ".info-b{background:var(--cb);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;font-size:.83rem;line-height:1.55;color:var(--tx)}"
    ".warn-b{background:rgba(217,119,6,.06);border:1px solid rgba(217,119,6,.25);border-radius:8px;padding:10px 14px;font-size:.83rem;color:#92400e}"
    ".err-b{background:rgba(220,38,38,.06);border:1px solid rgba(220,38,38,.20);border-radius:8px;padding:10px 14px;font-size:.83rem;color:#7f1d1d}"
    ".wf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:14px}"
    ".wf-step{background:var(--sf);border:1px solid var(--bs);border-radius:9px;padding:11px 13px}"
    ".wf-num{display:inline-flex;width:20px;height:20px;align-items:center;justify-content:center;border-radius:50%;background:var(--cb);border:1px solid var(--bd);color:var(--a);font-weight:900;font-size:.70rem;margin-bottom:6px}"
    ".card{background:var(--sf)!important;border:1px solid var(--bs)!important;border-radius:10px!important;box-shadow:var(--sh)!important}"
    ".form-label{font-size:.80rem;font-weight:600;color:var(--tx)}"
    ".form-control,.form-select{font-size:.83rem;border-color:var(--bs)}"
    ".form-control:focus,.form-select:focus{border-color:var(--a);box-shadow:0 0 0 2px var(--cb)}"
    ".nav-tabs .nav-link{font-size:.79rem;font-weight:600;padding:6px 13px;color:var(--mu);border:none;border-bottom:2px solid transparent;background:transparent}"
    ".nav-tabs .nav-link.active{color:var(--a);border-bottom-color:var(--a)}"
    ".nav-tabs{border-bottom:1px solid var(--bs)}"
    ".nav-pills .nav-link{font-size:.79rem;font-weight:600;padding:5px 12px;border-radius:6px;color:var(--mu)}"
    ".nav-pills .nav-link.active{background:var(--cb)!important;color:var(--a)!important;border:1px solid var(--bd)!important}"
    ".hero-topline{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:4px}.hero-topline .hero-ey{margin:0}.primary-nav{min-height:38px;padding-top:5px;padding-bottom:5px}.workbench-nav-row{display:flex;align-items:stretch;gap:0;flex-wrap:nowrap;width:auto!important;padding:0 20px}.hug-tabrow>div.workbench-nav-row{width:auto!important}.workbench-page{width:100%;max-width:1440px;margin:0 auto}.workbench-data-setup{margin-bottom:12px}.setup-command-grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(260px,.55fr);gap:12px}.setup-card{background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:14px 16px;box-shadow:var(--sh)}.setup-card-compact{padding:11px 13px}.experiment-setup-card{padding:0;background:transparent;border:0;box-shadow:none}.setup-field-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.role-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}.setup-field{min-width:0}.upload-roles{margin-top:10px}.upload-dropzone{border:1.5px dashed var(--bd);border-radius:8px;padding:10px 12px;text-align:center;background:var(--cb);cursor:pointer}.compact-upload{display:flex;justify-content:center;align-items:center;min-height:42px}.upload-title{font-size:.80rem;font-weight:700;color:var(--a)}.upload-subtitle{font-size:.70rem;color:var(--mu);margin-left:4px}.upload-status{font-size:.74rem;margin-top:5px}.compact-choice{font-size:.78rem;margin-bottom:6px}.compact-dropdown{font-size:.80rem}.setup-help{font-size:.74rem;line-height:1.45;color:var(--mu);margin-top:7px}.setup-config-grid{display:grid;grid-template-columns:minmax(280px,.72fr) minmax(0,1.28fr);gap:12px}.model-picker-card .mb-2{margin-bottom:.3rem!important}.model-picker-card .form-check{margin-right:9px}.model-config-card .card{box-shadow:none!important;margin-bottom:8px!important}.model-config-card .card-body{padding:10px 12px}.model-config-card .accordion-item{background:transparent;border-color:var(--bs)}.model-config-card .accordion-button{font-size:.78rem;font-weight:700;padding:8px 10px;background:var(--cb);color:var(--tx)}.model-config-card .accordion-body{padding:9px 10px}.setup-section{margin-bottom:10px}.setup-lbl{font-size:.60rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--mu);margin-bottom:6px;display:block}.setup-row{display:flex;align-items:center;gap:6px;margin-bottom:5px;flex-wrap:wrap}.setup-cat{font-size:.72rem;font-weight:700;color:var(--tx);min-width:120px;flex-shrink:0}.mini-input{font-size:.78rem!important;padding:3px 6px!important;border-radius:5px!important;border:1px solid var(--bs)!important;background:transparent;color:var(--tx)}.mini-label{font-size:.73rem;color:var(--mu);white-space:nowrap}.run-action-bar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:12px;padding:11px 13px;background:var(--sf);border:1px solid var(--bs);border-radius:10px;box-shadow:var(--sh)}.run-action-controls{display:flex;align-items:center;justify-content:flex-end;gap:10px;min-width:300px}.run-primary-action{min-width:190px;padding:7px 18px!important}.results-view-menu>.nav-tabs{gap:2px;flex-wrap:wrap}.results-view-menu .tab-content{padding-top:12px}.results-side-card{height:100%;background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:9px 11px}.results-card-title{font-size:.80rem;font-weight:700;margin:2px 0 8px}.compact-leaderboard-chart{width:100%;max-width:100%}.compact-leaderboard-chart .js-plotly-plot,.compact-leaderboard-chart .plot-container{width:100%!important}.artifact-control-grid{display:grid;grid-template-columns:minmax(170px,.7fr) minmax(220px,1fr) minmax(220px,1fr);gap:10px;margin-bottom:12px}.inspect-selector{max-width:520px;margin-bottom:12px}.inspect-heading{display:flex;align-items:center;gap:9px;margin-bottom:10px}.inspect-title{font-size:1rem;font-weight:800}.inspect-run-id{font-size:.72rem;color:var(--mu);margin-right:auto}.inspect-feature-chart{max-width:900px}.parameter-pre,.tree-pre{font-family:'JetBrains Mono',monospace;font-size:.71rem;background:var(--cb);border-radius:6px;padding:9px;white-space:pre-wrap;max-height:320px;overflow:auto}.empty-results{padding:48px 20px;text-align:center;background:var(--sf);border:1px dashed var(--bd);border-radius:10px}.empty-results-title{font-size:.95rem;font-weight:800;margin-bottom:4px}.promotion-controls{display:grid;grid-template-columns:minmax(260px,1fr) auto;gap:10px;align-items:center}.promotion-select{min-width:260px}.workbench-page .tab-content{padding-top:12px}.candidate-help{margin:0 0 9px;padding:7px 9px;background:var(--cb);border:1px solid var(--bd);border-radius:7px}.candidate-grid{display:grid;gap:9px}.candidate-grid-two{grid-template-columns:repeat(2,minmax(0,1fr))}.candidate-grid-three{grid-template-columns:repeat(3,minmax(0,1fr))}.candidate-grid-four{grid-template-columns:repeat(4,minmax(0,1fr))}.rpte-result-tabs>.nav-tabs{margin-top:4px}.rpte-result-tabs .tab-content{padding-top:10px}.rpte-tree-pre{max-height:440px}.config-compare-chart .js-plotly-plot,.config-compare-chart .plot-container{width:100%!important}@media(max-width:1000px){.setup-command-grid,.setup-config-grid{grid-template-columns:1fr}.role-grid{grid-template-columns:1fr 1fr}.artifact-control-grid,.candidate-grid-four,.candidate-grid-three{grid-template-columns:1fr 1fr}.hug-content{padding:16px}.hero-p{display:none}}@media(max-width:650px){.role-grid,.setup-field-grid,.candidate-grid-two,.candidate-grid-three,.candidate-grid-four{grid-template-columns:1fr}.run-action-bar,.run-action-controls{align-items:stretch;flex-direction:column}.run-action-controls{min-width:0;width:100%}.run-primary-action{width:100%}.promotion-controls{grid-template-columns:1fr}.hug-ctrl{padding-left:14px;padding-right:14px}.ws-btn{padding:5px 10px}.hug-hero{padding-left:16px;padding-right:16px}.workbench-nav-row,#gov-nav-row{padding-left:10px;padding-right:10px}}"
)


_PROFILE_CSS = (
    ".data-profile-section{margin-top:10px}.profile-preview-card{background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:11px 13px;box-shadow:var(--sh)}.profile-preview-label-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:7px}.profile-preview-label-row .setup-lbl{margin-bottom:0}.profile-preview-hint{font-size:.68rem;color:var(--mu)}.data-profile-accordion{margin-top:9px}.data-profile-accordion .accordion-item{border:1px solid var(--bs);border-radius:10px!important;overflow:hidden;background:var(--sf);box-shadow:var(--sh)}"
    ".data-profile-accordion .accordion-button{padding:10px 14px;font-size:.80rem;font-weight:800;background:var(--cb);color:var(--tx)}.data-profile-accordion .accordion-button:not(.collapsed){background:var(--cb);color:var(--a);box-shadow:none}.data-profile-accordion .accordion-body{padding:12px 14px 15px}"
    ".profile-preview-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:8px}.profile-preview-title{font-size:.88rem;font-weight:800;color:var(--tx)}.profile-preview-meta{font-size:.70rem;color:var(--mu)}.profile-empty{font-size:.76rem;color:var(--mu);padding:8px 0}"
    ".profile-toolbar{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0 6px}.profile-control-block{background:var(--cb);border:1px solid var(--bd);border-radius:8px;padding:8px 10px}.profile-scope-choice{font-size:.75rem}.profile-scope-choice label{margin-right:14px}.profile-scope-note{font-size:.70rem;color:var(--mu);margin:6px 0 10px}.profile-view-summary{background:var(--cb);border:1px solid var(--bd);border-radius:8px;padding:9px 11px;margin:7px 0}.profile-view-help{font-size:.72rem;color:var(--tx);line-height:1.45}.profile-view-footnote,.profile-excluded-empty{font-size:.66rem;color:var(--mu);margin-top:7px}.profile-excluded-title{font-size:.63rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--mu);margin-top:8px}.profile-excluded-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:5px}.profile-excluded-chip{display:inline-flex;align-items:center;gap:5px;flex-wrap:wrap;background:var(--sf);border:1px solid var(--bs);border-radius:999px;padding:4px 7px}.profile-excluded-name{font-size:.68rem;font-weight:800;color:var(--tx)}.profile-role-badge{font-size:.55rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:var(--a);background:var(--cb);border:1px solid var(--bd);border-radius:999px;padding:1px 5px}.profile-role-sensitive{color:var(--am)}"
    ".profile-tabs>.nav-tabs{margin-top:5px}.profile-card-grid{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin:10px 0 12px}.profile-card{background:var(--sf);border:1px solid var(--bs);border-top:2px solid var(--a);border-radius:8px;padding:9px 11px}.profile-card-label{font-size:.59rem;text-transform:uppercase;letter-spacing:.09em;color:var(--mu);font-weight:800}.profile-card-value{font-size:1.15rem;font-weight:900;color:var(--tx);margin-top:2px}"
    ".profile-two-col{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-items:start}.profile-relationship-grid{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.35fr);gap:12px;align-items:start}.profile-panel{background:var(--sf);border:1px solid var(--bs);border-radius:9px;padding:10px 12px;min-height:280px}.profile-panel-title{font-size:.78rem;font-weight:800;color:var(--tx);margin-bottom:8px}"
    ".profile-findings{display:grid;gap:6px;max-height:430px;overflow:auto}.profile-finding{border-left:3px solid var(--bd);background:var(--cb);border-radius:0 7px 7px 0;padding:7px 9px}.profile-finding-warning{border-left-color:var(--rd)}.profile-finding-attention{border-left-color:var(--am)}.profile-finding-ok{border-left-color:var(--gr)}.profile-finding-title{font-size:.72rem;font-weight:800;color:var(--tx)}.profile-finding-detail{font-size:.68rem;color:var(--mu);line-height:1.4;margin-top:1px}"
    ".profile-variable-grid{display:grid;grid-template-columns:minmax(360px,.85fr) minmax(0,1.65fr);gap:12px}.profile-variable-list,.profile-variable-detail{background:var(--sf);border:1px solid var(--bs);border-radius:9px;padding:10px 12px;min-width:0}.profile-variable-list .Select-control{margin-bottom:8px}.profile-heading-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.profile-type-badge{font-size:.62rem;font-weight:700;color:var(--a);background:var(--cb);border:1px solid var(--bd);border-radius:10px;padding:2px 7px}"
    ".profile-mini-grid{display:grid;grid-template-columns:repeat(4,minmax(100px,1fr));gap:7px;margin-bottom:8px}.profile-mini-card{background:var(--cb);border:1px solid var(--bd);border-radius:7px;padding:7px 8px}.profile-mini-label{font-size:.57rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mu);font-weight:800}.profile-mini-value{font-size:.80rem;font-weight:800;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
    ".theme-dark .data-profile-accordion .accordion-item,.theme-dark .profile-panel,.theme-dark .profile-variable-list,.theme-dark .profile-variable-detail,.theme-dark .profile-card{background:var(--sf)!important;border-color:var(--bd)!important}.theme-dark .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,.theme-dark .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bs)!important}"
    "@media(max-width:1000px){.profile-variable-grid{grid-template-columns:1fr}.profile-card-grid{grid-template-columns:repeat(2,minmax(120px,1fr))}}@media(max-width:700px){.profile-toolbar,.profile-two-col,.profile-relationship-grid{grid-template-columns:1fr}.profile-card-grid,.profile-mini-grid{grid-template-columns:repeat(2,minmax(100px,1fr))}.profile-preview-head{align-items:flex-start;flex-direction:column}}"
)


def _theme_class_css():
    blocks = []
    for name, tokens in _TOKENS.items():
        values = "".join(f"--{key}:{value};" for key, value in tokens.items())
        blocks.append(f".theme-{name.lower()}{{{values}}}")
    return "".join(blocks)


def build_css(theme="Ocean"):
    t = get_tokens(theme)
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in t.items()) + "}"
    return root + _theme_class_css() + _STATIC_CSS + _PROFILE_CSS + _THEME_STATE_CSS
