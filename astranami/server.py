"""Local-only HTTP server and the discovery page."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

from .util import log

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aftercredits</title>
<style>
:root{color-scheme:dark;--bg:#0e0f16;--panel:#171924;--line:#2c3144;--muted:#a7adc2;--accent:#d7b07a;--ink:#f4f1ea}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 500px at 10% -10%,#2a2418,transparent),var(--bg);color:var(--ink);font:16px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:32px 20px 64px}
header{display:flex;justify-content:space-between;gap:16px;align-items:end}
.brand{letter-spacing:.28em;font-size:13px;font-weight:700}.brand b{color:var(--accent)}
h1{font-size:clamp(34px,6vw,62px);line-height:1.02;letter-spacing:-.04em;margin:28px 0 8px;max-width:16ch}
h1 span{color:var(--accent)} p.lead,p{color:var(--muted)}
.panel{background:rgba(23,25,36,.92);border:1px solid var(--line);border-radius:18px;padding:18px}
label{display:block;margin:12px 0 6px;color:#ddd6c8;font-size:13px}
input,select,textarea,button{font:inherit} input,select,textarea{width:100%;background:#10131c;color:var(--ink);border:1px solid #3a4158;border-radius:10px;padding:10px}
textarea{min-height:84px;resize:vertical} button{border:0;border-radius:10px;background:var(--accent);color:#1c140c;font-weight:700;padding:10px 14px;cursor:pointer}
button.secondary{background:#2a3144;color:#efe7da;font-weight:600}
button:disabled{opacity:.55;cursor:wait}
.row{display:grid;grid-template-columns:1.4fr repeat(3,1fr);gap:12px}
.sliders{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.lookup{display:flex;gap:8px}.lookup input{flex:1}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{background:#2c261c;color:#f0e2c8;border-radius:999px;padding:6px 10px;font-size:13px;border:0;cursor:pointer}
.checks{display:flex;gap:18px;align-items:center;margin-top:8px}.checks label{display:flex;gap:8px;align-items:center;margin:0}.checks input{width:auto}
.submit{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:16px}
#status{min-height:28px;margin:18px 0} #results{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;display:flex;flex-direction:column;gap:8px}
.card h2{font-size:22px;line-height:1.2;margin:0}.meta{color:var(--accent);font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.tags{color:#d9c7a4;font-size:13px}.reason{border-top:1px solid var(--line);padding-top:8px;color:#ddd6c8}
.bars{display:grid;gap:4px}.bar{display:grid;grid-template-columns:64px 1fr 36px;gap:6px;align-items:center;font-size:12px;color:var(--muted)}
.bar i{display:block;height:7px;border-radius:99px;background:#2a3144;overflow:hidden}.bar i b{display:block;height:100%;background:var(--accent)}
.matches{display:grid;gap:6px;margin-top:8px}.match{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid #2a3144}
footer{margin-top:36px;color:var(--muted);font-size:12px} a{color:var(--accent)}
@media(max-width:860px){.row,.sliders,#results{grid-template-columns:1fr}.match,.submit{flex-direction:column;align-items:stretch}}
</style></head><body><main>
<header><div class="brand">AFTER<b>CREDITS</b></div><div id="catalog">Local ranker</div></header>
<h1>Same feeling.<br><span>Different title.</span></h1>
<p class="lead">Like a story. The ranker matches plot, tone, themes, creators, and what fans of it actually recommend.</p>
<section class="panel">
<label for="title">Find a title</label>
<div class="lookup"><input id="title" maxlength="200" placeholder="Monster, The Wire, Arrival"><button id="find" class="secondary" type="button">Find</button></div>
<div id="matches" class="matches"></div>
<div id="preferences" class="chips"></div>
<label for="query">Or describe the mood</label>
<textarea id="query" maxlength="1000" placeholder="Slow psychological mystery, moral gray, quiet dread"></textarea>
<div class="row">
<div><label for="kind">Format</label><select id="kind"><option value="all">Any format</option><option value="anime">Anime</option><option value="tv">TV</option><option value="movie">Movies</option></select></div>
<div><label for="year_min">After year</label><input id="year_min" type="number" min="0" max="9999" placeholder="Any"></div>
<div><label for="min_rating">Min rating</label><input id="min_rating" type="number" min="0" max="10" step="0.5" value="0"></div>
<div><label for="diversity">Diversity <span id="diversity_val">0.12</span></label><input id="diversity" type="range" min="0" max="0.6" step="0.02" value="0.12"></div>
</div>
<div class="sliders">
<div><label>Plot <span id="plot_val">1</span></label><input id="plot" type="range" min="0" max="3" step="0.1" value="1"></div>
<div><label>Themes <span id="themes_val">1</span></label><input id="themes" type="range" min="0" max="3" step="0.1" value="1"></div>
<div><label>Tone <span id="tone_val">1</span></label><input id="tone" type="range" min="0" max="3" step="0.1" value="1"></div>
<div><label>Creators <span id="creators_val">1</span></label><input id="creators" type="range" min="0" max="3" step="0.1" value="1"></div>
<div><label>Hidden gems <span id="gems_val">0</span></label><input id="hidden_gems" type="range" min="0" max="1" step="0.05" value="0"></div>
</div>
<label for="exclude_tags">Exclude tags</label><input id="exclude_tags" placeholder="horror, gore">
<div class="checks"><label><input id="cross_format" type="checkbox" checked>Allow other formats</label><label><input id="explain" type="checkbox">Show why each title ranked</label></div>
<div class="submit"><span>Liked franchises are held out, so sequels do not crowd the list.</span><button id="discover" type="button">Rank the next watch</button></div>
</section>
<div id="status">Like a title or describe a mood.</div>
<section id="results"></section>
<footer><p id="source-note"></p><div id="sources"></div></footer>
</main><script>
const el=id=>document.getElementById(id);
let state={liked:{},disliked:{},seen:{}};
try{const saved=JSON.parse(localStorage.getItem("aftercredits-v2"));if(saved)state=saved;}catch(e){}
function node(tag,text,cls){const item=document.createElement(tag);if(text!==undefined)item.textContent=text;if(cls)item.className=cls;return item;}
function save(){localStorage.setItem("aftercredits-v2",JSON.stringify(state));renderPrefs();}
function choose(kind,item){for(const key of ["liked","disliked","seen"])delete state[key][item.id];state[kind][item.id]=item.title;save();}
function renderPrefs(){const box=el("preferences");box.replaceChildren();for(const key of ["liked","disliked","seen"])for(const [id,title] of Object.entries(state[key])){const chip=node("button",key+": "+title+" ×","chip");chip.onclick=()=>{delete state[key][id];save();};box.append(chip);}}
function actions(item){const box=node("div",undefined,"chips");for(const [key,label] of [["liked","Like"],["disliked","Not for me"],["seen","Watched"]]){const button=node("button",label,"secondary");button.onclick=()=>choose(key,item);box.append(button);}return box;}
async function api(path,body){const response=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(data.error||"Request failed");return data;}
async function lookup(){const data=await api("/api/titles?q="+encodeURIComponent(el("title").value));const box=el("matches");box.replaceChildren();if(!data.length){box.textContent="No titles found.";return;}for(const item of data){const row=node("div",undefined,"match");row.append(node("span",item.title+" · "+item.kind+" · "+(item.year||"")),actions(item));box.append(row);}}
function bars(aspects){const box=node("div",undefined,"bars");for(const [name,value] of Object.entries(aspects||{})){const row=node("div",undefined,"bar");const meter=node("i");const fill=node("b");fill.style.width=Math.round(Math.max(0,Math.min(1,value))*100)+"%";meter.append(fill);row.append(node("span",name),meter,node("span",value.toFixed(2)));box.append(row);}return box;}
function inspector(data){const box=node("details",undefined,"inspect");box.append(node("summary","Why this rank"));const lines=[];lines.push("Scorer: "+data.scorer+" · rank before diversity: "+data.rank_before_diversity+" · franchise entries: "+data.franchise_size);if((data.found_by||[]).length)lines.push("Found by: "+data.found_by.join(", "));const parts=Object.entries(data.score_parts||{}).map(([name,value])=>name.replace("_"," ")+" "+(value>=0?"+":"")+value.toFixed(2));if(parts.length)lines.push("Score: "+parts.join(" · "));for(const liked of data.liked||[])lines.push("vs "+liked.title+": "+liked.score.toFixed(2));for(const text of lines)box.append(node("p",text));const table=node("div",undefined,"bars");for(const feature of (data.features||[]).slice(0,8)){const row=node("div",undefined,"bar");row.append(node("span",feature.label),node("span",""),node("span",(feature.contribution>=0?"+":"")+feature.contribution.toFixed(2)));table.append(row);}box.append(table);return box;}
el("find").onclick=lookup;
el("title").onkeydown=event=>{if(event.key==="Enter")lookup();};
for(const id of ["diversity","plot","themes","tone","creators","hidden_gems"])el(id).oninput=()=>{el(id==="hidden_gems"?"gems_val":id==="diversity"?"diversity_val":id+"_val").textContent=el(id).value;};
el("discover").onclick=async()=>{el("discover").disabled=true;el("status").textContent="Ranking…";el("results").replaceChildren();try{
const request={query:el("query").value,kind:el("kind").value,year_min:Number(el("year_min").value||0),year_max:9999,min_rating:Number(el("min_rating").value||0),diversity:Number(el("diversity").value),hidden_gems:Number(el("hidden_gems").value),cross_format:el("cross_format").checked,exclude_tags:el("exclude_tags").value.split(",").map(text=>text.trim()).filter(Boolean),explain:el("explain").checked,aspect_weights:{plot:Number(el("plot").value),themes:Number(el("themes").value),tone:Number(el("tone").value),creators:Number(el("creators").value)}};
for(const key of ["liked","disliked","seen"])request[key]=Object.keys(state[key]);
const data=await api("/api/recommend",request);
el("status").textContent=data.results.length+" titles. "+data.message;
for(const item of data.results){const card=node("article",undefined,"card");card.append(node("div",item.kind+" · "+(item.year||"")+" · "+(item.rating?Number(item.rating).toFixed(1):"unrated"),"meta"),node("h2",item.title),node("div",(item.tags||[]).slice(0,8).join(" · "),"tags"),bars(item.aspects),node("p",(item.synopsis||"No synopsis.").slice(0,320)),node("p",(item.because?"Because you liked "+item.because.title+": ":"")+(item.explanation||""),"reason"),actions(item));if(item.inspect)card.append(inspector(item.inspect));if(item.url){const link=node("a","Source");link.href=item.url;link.target="_blank";link.rel="noopener";card.append(link);}el("results").append(card);}}
catch(error){el("status").textContent=error.message;}finally{el("discover").disabled=false;}};
renderPrefs();
api("/api/stats").then(data=>{el("catalog").textContent=Number(data.count).toLocaleString()+" titles";el("source-note").textContent=data.note||"";for(const source of data.sources||[]){const row=node("p");const link=node("a",source.name);link.href=source.url;link.target="_blank";link.rel="noopener";row.append(link,document.createTextNode(" — "+source.terms));el("sources").append(row);}}).catch(error=>{el("status").textContent=error.message;});
</script></body></html>'''


def serve(engine, port: int = 8765, ready=None) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log("%s - %s" % (self.address_string(), fmt % args))

        def reply(self, status, value, content_type="application/json; charset=utf-8"):
            body = value.encode() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "img-src https: data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def trusted_host(self) -> bool:
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.trusted_host():
                self.reply(403, {"error": "Untrusted Host header."})
                return
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.reply(200, PAGE, "text/html; charset=utf-8")
                elif parsed.path == "/api/stats":
                    self.reply(200, engine.stats())
                elif parsed.path == "/api/titles":
                    query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
                    if len(query) > 200:
                        self.reply(400, {"error": "Title query exceeds 200 characters."})
                    else:
                        self.reply(200, engine.lookup(query))
                elif parsed.path == "/api/similar":
                    params = urllib.parse.parse_qs(parsed.query)
                    identifier = params.get("id", [""])[0]
                    limit = int(params.get("limit", ["12"])[0])
                    self.reply(200, engine.recommend({"liked": [identifier], "limit": limit}))
                else:
                    self.reply(404, {"error": "Not found."})
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Request failed."})
                raise

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if not self.trusted_host() or (origin and origin not in allowed):
                self.reply(403, {"error": "Cross-origin requests are not accepted."})
                return
            if self.path != "/api/recommend":
                self.reply(404, {"error": "Not found."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, {"error": "Expected application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request body must contain 1-65536 bytes.")
                payload = json.loads(self.rfile.read(length))
                self.reply(200, engine.recommend(payload))
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Recommendation failed."})
                raise

    httpd = HTTPServer(("127.0.0.1", port), Handler)
    log(f"Aftercredits listening on http://127.0.0.1:{port}")
    if ready:
        ready(httpd)
    httpd.serve_forever()
