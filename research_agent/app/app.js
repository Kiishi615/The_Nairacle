/**
 * NAIRACLE — Telegram Mini App
 * Purple Nebula theme · Session management · Telegram SDK · SSE Streaming
 */
(function(){
"use strict";
const API="/api",TG=!!(window.Telegram&&window.Telegram.WebApp),tg=TG?window.Telegram.WebApp:null;
const st={sessions:[],activeId:null,messages:[],loading:false,userId:null};
let anonId=localStorage.getItem("anon_user_id");
if(!anonId){anonId=Math.floor(Math.random()*2147483647).toString();localStorage.setItem("anon_user_id",anonId);}
const $=s=>document.querySelector(s);
const el={
  sidebar:$("#sidebar"),overlay:$("#sidebar-overlay"),btnMenu:$("#btn-menu"),
  btnNew:$("#btn-new-chat"),sList:$("#session-list"),
  welcome:$("#welcome"),chatArea:$("#chat-area"),msgs:$("#messages"),typing:$("#typing"),
  input:$("#msg-input"),btnSend:$("#btn-send")
};

/* Telegram */
function initTG(){if(!tg)return;tg.ready();tg.expand();if(tg.initDataUnsafe?.user)st.userId=tg.initDataUnsafe.user.id}
function haptic(s){tg?.HapticFeedback?.impactOccurred(s)}

/* API */
async function api(p,o={}){
  const h={"Content-Type":"application/json",...o.headers};
  if(TG&&tg.initData)h["X-Telegram-Init-Data"]=tg.initData;
  else h["X-Anonymous-User-Id"]=anonId;
  try{const r=await fetch(API+p,{...o,headers:h});if(!r.ok)throw new Error(r.status);return await r.json()}
  catch(e){console.warn("API:",e.message);return null}
}

function apiHeaders(){
  const h={"Content-Type":"application/json"};
  if(TG&&tg.initData)h["X-Telegram-Init-Data"]=tg.initData;
  else h["X-Anonymous-User-Id"]=anonId;
  return h;
}

async function loadSessions(){
  const d=await api("/sessions");
  st.sessions=d?.sessions||demoSessions();
  renderSessions();
}

async function loadMessages(id){
  const d=await api(`/sessions/${id}/messages`);
  st.messages=d?.messages||demoMessages();
  renderMessages();
}

async function newSession(){
  const d=await api("/sessions",{method:"POST"});
  if(d?.id){st.activeId=d.id;await loadSessions();st.messages=[];renderMessages()}
  else{const id="d-"+Date.now();st.sessions.unshift({id,title:"New research",created_at:new Date().toISOString(),message_count:0});st.activeId=id;st.messages=[];renderSessions();renderMessages()}
  closeSidebar();haptic("medium");
}

async function switchTo(id){st.activeId=id;await loadMessages(id);renderSessions();closeSidebar();haptic("light")}

async function delSession(id){
  await api(`/sessions/${id}`,{method:"DELETE"});
  st.sessions=st.sessions.filter(s=>s.id!==id);
  if(st.activeId===id){st.activeId=null;st.messages=[];renderMessages()}
  renderSessions();haptic("medium");
}

/* Demo */
function demoSessions(){return[
  {id:"d1",title:"NGX All-Share Index analysis",created_at:new Date().toISOString(),message_count:4},
  {id:"d2",title:"CBN interest rate decision",created_at:new Date(Date.now()-864e5).toISOString(),message_count:6},
  {id:"d3",title:"GTBank Q3 earnings",created_at:new Date(Date.now()-1728e5).toISOString(),message_count:3}
]}
function demoMessages(){return[
  {role:"user",content:"What are the top-performing stocks on NGX today?",created_at:new Date(Date.now()-6e4).toISOString()},
  {role:"assistant",content:"Based on the latest data from the Nigerian Exchange (NGX), here are today's top performers:\n\n📈 **Top Gainers:**\n1. **Dangote Cement (DANGCEM)** — ₦412.50 (+5.2%)\n2. **BUA Foods (BUAFOODS)** — ₦298.00 (+3.8%)\n3. **Airtel Africa (AIRTELAFRI)** — ₦1,850.00 (+2.9%)\n4. **GTCO Holdings (GTCO)** — ₦52.80 (+2.5%)\n5. **Zenith Bank (ZENITHBANK)** — ₦41.30 (+2.1%)\n\nThe All-Share Index is currently at **105,234.67**, up **1.3%** from yesterday's close. Trading volume stands at ₦8.2 billion across 312 million shares.\n\nThe banking sector is leading gains today, driven by positive Q3 earnings expectations.",created_at:new Date().toISOString()}
]}

/* Render Sessions */
function renderSessions(){
  const l=el.sList;
  if(!st.sessions.length){l.innerHTML='<div class="sidebar__empty">No sessions yet</div>';return}
  let h="";
  for(const s of st.sessions){
    const on=s.id===st.activeId;
    h+=`<button class="session-item${on?" active":""}" data-id="${s.id}">${esc(s.title)}</button>`;
  }
  l.innerHTML=h;
  l.querySelectorAll(".session-item").forEach(el=>{el.addEventListener("click",()=>switchTo(el.dataset.id))});
}

/* Render Messages */
function renderMessages(){
  if(!st.messages.length){el.welcome.style.display="";el.chatArea.style.display="none";return}
  el.welcome.style.display="none";el.chatArea.style.display="";
  let h="";
  for(const m of st.messages){
    if(m.role==="user"){
      h+=`<div class="msg msg--user">
        <div class="msg__avatar-wrap msg__avatar-wrap--user"></div>
        <div class="msg__bubble nebula-card msg__bubble--user"><div class="msg__content">${fmt(m.content)}</div></div>
      </div>`;
    }else{
      h+=`<div class="msg msg--ai">
        <div class="msg__avatar-wrap nebula-icon-bg"><img src="profile-photo.png" class="msg__avatar-img" alt=""/></div>
        <div class="msg__bubble nebula-card msg__bubble--ai">
          <div class="msg__content">${fmt(m.content)}</div>
          <div class="msg__actions">
            <button class="msg__action-btn copy-btn" title="Copy" data-copy="${escA(m.content)}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
          </div>
        </div>
      </div>`;
    }
  }
  el.msgs.innerHTML=h;
  el.msgs.querySelectorAll(".copy-btn").forEach(b=>{b.addEventListener("click",()=>{
    navigator.clipboard.writeText(b.dataset.copy).then(()=>{b.style.color="#4ade80";setTimeout(()=>b.style.color="",1500);haptic("light")})
  })});
  scrollDown();
}

/* Sidebar */
function toggleSidebar(){
  if(window.innerWidth>=1024){
    el.sidebar.classList.toggle("collapsed");
  }else{
    if(el.sidebar.classList.contains("open")) closeSidebar();
    else openSidebar();
  }
  haptic("light");
}
function openSidebar(){
  if(window.innerWidth>=1024){
    el.sidebar.classList.remove("collapsed");
  }else{
    el.sidebar.classList.add("open");
    el.overlay.classList.add("open");
  }
  haptic("light");
}
function closeSidebar(){
  if(window.innerWidth>=1024){
    el.sidebar.classList.add("collapsed");
  }else{
    el.sidebar.classList.remove("open");
    el.overlay.classList.remove("open");
  }
}

/* Input */
function setupInput(){
  el.input.addEventListener("input",()=>{
    el.input.style.height="auto";el.input.style.height=Math.min(el.input.scrollHeight, window.innerHeight * 0.5)+"px";
    el.btnSend.disabled=!el.input.value.trim();
  });
  el.input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});
  el.btnSend.addEventListener("click",send);
}

/* ------------------------------------------------------------------ */
/* Streaming send — SSE with progressive token rendering              */
/* ------------------------------------------------------------------ */

async function send(){
  const txt=el.input.value.trim();if(!txt||st.loading)return;

  // Auto-create session if none active
  if(!st.activeId){
    const d=await api("/sessions",{method:"POST"});
    if(d?.id){st.activeId=d.id;st.sessions.unshift({id:d.id,title:txt.slice(0,40),created_at:new Date().toISOString(),message_count:0})}
    else{st.activeId="d-"+Date.now();st.sessions.unshift({id:st.activeId,title:txt.slice(0,40),created_at:new Date().toISOString(),message_count:0})}
    renderSessions();
  }

  // Add user message
  st.messages.push({role:"user",content:txt,created_at:new Date().toISOString()});
  el.input.value="";el.input.style.height="auto";el.btnSend.disabled=true;
  renderMessages();
  st.loading=true;

  // Create a live AI bubble for streaming
  const liveBubble = createLiveAiBubble();
  let fullText = "";
  let doneText = "";  // authoritative final from server's done event
  let streamOk = false;

  try {
    const resp = await fetch(API+`/sessions/${st.activeId}/messages/stream`, {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify({content: txt}),
    });

    if (!resp.ok || !resp.body) throw new Error("stream-unavailable");

    streamOk = true;
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";  // keep incomplete line in buffer

      let eventType = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ") && eventType) {
          try {
            const data = JSON.parse(line.slice(6));
            handleSSE(eventType, data, liveBubble,
              (t) => { fullText += t; },
              (t) => { doneText = t; }
            );
          } catch(e) { /* skip malformed */ }
          eventType = "";
        }
      }
    }
  } catch(e) {
    console.warn("Streaming failed, falling back:", e.message);
    if (!streamOk) {
      // Fallback to non-streaming endpoint
      removeLiveAiBubble(liveBubble);
      showTyping(true);
      const resp = await api(`/sessions/${st.activeId}/messages`, {
        method: "POST",
        body: JSON.stringify({content: txt}),
      });
      showTyping(false);
      if (resp?.message) {
        st.messages.push(resp.message);
      } else {
        st.messages.push({role:"assistant",content:"I couldn't reach the server right now. Please try again.",created_at:new Date().toISOString()});
      }
      st.loading=false;
      renderMessages();
      haptic("medium");
      return;
    }
  }

  // Finalize: prefer accumulated tokens, fall back to server's done event
  const finalContent = fullText || doneText;
  if (finalContent) {
    st.messages.push({role:"assistant",content:finalContent,created_at:new Date().toISOString()});
  } else {
    st.messages.push({role:"assistant",content:"I wasn't able to generate a response. Please try again.",created_at:new Date().toISOString()});
  }
  st.loading=false;
  renderMessages();
  haptic("medium");
}


/* --- SSE helpers --- */

function createLiveAiBubble(){
  el.welcome.style.display="none";
  el.chatArea.style.display="";
  const wrapper = document.createElement("div");
  wrapper.className = "msg msg--ai msg--streaming";
  wrapper.innerHTML = `
    <div class="msg__avatar-wrap nebula-icon-bg"><img src="profile-photo.png" class="msg__avatar-img" alt=""/></div>
    <div class="msg__bubble nebula-card msg__bubble--ai">
      <div class="msg__tool-status" style="display:none;"></div>
      <div class="msg__content">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  el.msgs.appendChild(wrapper);
  scrollDown();
  return wrapper;
}

function removeLiveAiBubble(bubble){
  if(bubble && bubble.parentNode) bubble.parentNode.removeChild(bubble);
}

function handleSSE(type, data, bubble, onText, onDone){
  const contentEl = bubble.querySelector(".msg__content");
  const toolEl = bubble.querySelector(".msg__tool-status");

  switch(type){
    case "token":
      if(data.content){
        onText(data.content);
        
        // Remove typing dots if they are still there
        const dots = contentEl.querySelector(".typing-dots");
        if(dots) dots.remove();

        // Remove cursor, append text, re-add cursor
        const cursor = contentEl.querySelector(".streaming-cursor");
        if(cursor) cursor.remove();
        contentEl.insertAdjacentText("beforeend", data.content);
        const newCursor = document.createElement("span");
        newCursor.className = "streaming-cursor";
        contentEl.appendChild(newCursor);
        scrollDown();
      }
      break;

    case "tool_start":
      toolEl.style.display = "";
      toolEl.textContent = `🔍 ${data.label || data.tool}…`;
      scrollDown();
      break;

    case "tool_end":
      toolEl.style.display = "none";
      toolEl.textContent = "";
      break;

    case "done":
      // Authoritative final content from the server.
      // If we missed tokens during streaming, use this instead.
      if(data.content){
        onDone(data.content);
      }
      // Remove streaming cursor
      {
        const c = contentEl.querySelector(".streaming-cursor");
        if(c) c.remove();
      }
      break;

    case "error":
      contentEl.innerHTML = `<em>${esc(data.message||"An error occurred.")}</em>`;
      const cursor = contentEl.querySelector(".streaming-cursor");
      if(cursor) cursor.remove();
      break;
  }
}


function showTyping(on){el.typing.style.display=on?"":"none";if(on)scrollDown()}
function scrollDown(){requestAnimationFrame(()=>{el.chatArea.scrollTop=el.chatArea.scrollHeight})}

/* Suggestion cards */
function setupCards(){
  document.querySelectorAll(".suggestion-card").forEach(c=>{
    c.addEventListener("click",()=>{el.input.value=c.dataset.q;el.input.dispatchEvent(new Event("input"));el.input.focus()})
  });
}

/* Helpers */
function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML}
function escA(s){return s.replace(/"/g,"&quot;").replace(/</g,"&lt;")}
function fmt(t){
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true });
    return marked.parse(t);
  }
  let h=esc(t);h=h.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>");h=h.replace(/\n/g,"<br>");return h;
}

/* Boot */
function boot(){
  initTG();setupInput();setupCards();
  el.btnMenu.addEventListener("click",toggleSidebar);
  el.overlay.addEventListener("click",closeSidebar);
  
  // Retreat when user clicks the chat box on small screens
  el.chatArea.addEventListener("click", () => {
    if(window.innerWidth<1024 && el.sidebar.classList.contains("open")) closeSidebar();
  });
  el.input.addEventListener("focus", () => {
    if(window.innerWidth<1024 && el.sidebar.classList.contains("open")) closeSidebar();
  });
  el.btnNew.addEventListener("click",newSession);
  loadSessions();
  setTimeout(()=>{if(st.sessions.length&&!st.activeId){st.activeId=st.sessions[0].id;loadMessages(st.sessions[0].id);renderSessions()}},300);
}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot);else boot();
})();
