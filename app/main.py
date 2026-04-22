# app/main.py
"""
FastAPI 主应用 — 含流式 SSE 接口 + 完整管理界面
"""
import os
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    Query,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from app.config import get_settings, BASE_DIR
from app.models import (
    ChatRequest,
    ChatResponse,
    KnowledgeBaseCreate,
    APIResponse,
    HealthResponse,
)
from app.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from app.vector_store_manager import VectorStoreManager
from app.chat_engine import ChatEngine

# ===== 创建应用 =====
app = FastAPI(
    title="🧠 智能知识库系统",
    description="基于 LangChain + FastAPI 的 RAG 知识库（支持流式输出）",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
doc_processor = DocumentProcessor()
vector_manager = VectorStoreManager()
chat_engine = ChatEngine()
upload_status: dict = {}


@app.on_event("startup")
async def startup_event():
    kb_list = vector_manager.list_knowledge_bases()
    if not any(kb["name"] == "default" for kb in kb_list):
        vector_manager.create_knowledge_base("default", description="默认知识库")


# ===== 后台处理 =====
async def process_uploaded_file(task_id, file_path, kb_name, filename):
    try:
        upload_status[task_id]["status"] = "processing"
        upload_status[task_id]["message"] = "正在处理文档..."
        chunks, info = doc_processor.process_file(file_path)
        add_result = vector_manager.add_documents(kb_name, chunks, filename)
        upload_status[task_id]["status"] = "completed"
        upload_status[task_id]["message"] = "处理完成"
        upload_status[task_id]["info"] = {**info, **add_result, "knowledge_base": kb_name}
    except Exception as e:
        upload_status[task_id]["status"] = "failed"
        upload_status[task_id]["message"] = f"处理失败: {str(e)}"
        upload_status[task_id]["info"] = {"error": str(e)}


# ================================================================
#                         API 路由
# ================================================================

@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "version": "1.1.0"}


# ---------- 知识库管理 ----------
@app.get("/api/kb/list", tags=["知识库"])
async def list_knowledge_bases():
    return {"code": 200, "data": vector_manager.list_knowledge_bases()}


@app.post("/api/kb/create", tags=["知识库"])
async def create_knowledge_base(req: KnowledgeBaseCreate):
    try:
        vector_manager.create_knowledge_base(req.name, req.description)
        return {"code": 200, "message": "创建成功"}
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@app.delete("/api/kb/{kb_name}", tags=["知识库"])
async def delete_knowledge_base(kb_name: str):
    if kb_name == "default":
        raise HTTPException(400, detail="不能删除默认知识库")
    if vector_manager.delete_knowledge_base(kb_name):
        return {"code": 200, "message": "已删除"}
    raise HTTPException(404, detail="不存在")


# ---------- 文档上传 ----------
@app.post("/api/upload", tags=["文档"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_base: str = Form("default"),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, detail=f"不支持 {ext}，支持: {list(SUPPORTED_EXTENSIONS.keys())}")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.max_upload_size:
        raise HTTPException(400, detail="文件超过50MB限制")

    kb_list = vector_manager.list_knowledge_bases()
    if not any(kb["name"] == knowledge_base for kb in kb_list):
        raise HTTPException(404, detail=f"知识库 {knowledge_base} 不存在")

    task_id = str(uuid.uuid4())
    filename = f"{task_id}_{file.filename}"
    file_path = Path(settings.documents_dir) / filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    upload_status[task_id] = {
        "filename": file.filename, "file_type": ext,
        "file_size": file_size, "status": "pending",
        "message": "等待处理...", "upload_time": datetime.now().isoformat(), "info": None,
    }
    background_tasks.add_task(process_uploaded_file, task_id, str(file_path), knowledge_base, file.filename)
    return {"code": 200, "data": {"task_id": task_id, "filename": file.filename}}


@app.get("/api/upload/status/{task_id}", tags=["文档"])
async def get_upload_status(task_id: str):
    if task_id not in upload_status:
        raise HTTPException(404, detail="任务不存在")
    return {"code": 200, "data": upload_status[task_id]}


# ---------- 智能问答 ----------

# 非流式接口（保留）
@app.post("/api/chat", tags=["问答"])
async def chat(req: ChatRequest):
    try:
        result = chat_engine.chat(
            question=req.question,
            knowledge_base=req.knowledge_base,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            conversation_id=req.conversation_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ⭐ 流式接口（新增）
@app.post("/api/chat/stream", tags=["问答"])
async def chat_stream(req: ChatRequest):
    """
    流式问答接口 — 返回 SSE (Server-Sent Events) 流

    SSE 事件类型:
      - sources:  引用来源信息（JSON）
      - token:    文本片段 {"content": "..."}
      - done:     结束 {"conversation_id": "...", "full_answer": "..."}
      - error:    错误 {"error": "..."}
    """
    return StreamingResponse(
        chat_engine.chat_stream(
            question=req.question,
            knowledge_base=req.knowledge_base,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            conversation_id=req.conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",       # Nginx 禁用缓冲
        },
    )


# ---------- 语义搜索 ----------
@app.get("/api/search", tags=["搜索"])
async def search_documents(
    q: str = Query(...),
    kb: str = Query("default"),
    top_k: int = Query(5),
    threshold: float = Query(0.5),
):
    results = vector_manager.similarity_search(kb, q, top_k, threshold)
    return {"code": 200, "data": {
        "query": q,
        "results": [{"content": d.page_content, "score": round(s, 4), "metadata": d.metadata} for d, s in results]
    }}


# ================================================================
#                      完整前端界面（含流式）
# ================================================================

@app.get("/ui", response_class=HTMLResponse, tags=["界面"])
async def ui():
    return HTMLResponse(content=UI_HTML)


# ===== 前端 HTML（完整内联）=====
# ===== 前端 HTML（含 Markdown 渲染 + 格式化显示）=====
UI_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧠 智能知识库系统</title>
<!-- 引入 Marked.js 用于 Markdown 渲染 -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root{--pri:#4f46e5;--pri-h:#4338ca;--bg:#f9fafb;--bdr:#e5e7eb;--tx:#111827;--tx2:#6b7280;--code-bg:#1e1e1e;--code-tx:#d4d4d4}
*{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
body{height:100vh;display:flex;background:var(--bg);color:var(--tx)}

/* 侧边栏 */
.sidebar{width:320px;background:#fff;border-right:1px solid var(--bdr);display:flex;flex-direction:column;flex-shrink:0}
.sb-hd{padding:20px;border-bottom:1px solid var(--bdr);display:flex;justify-content:space-between;align-items:center}
.sb-hd h2{font-size:18px}
.btn{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;transition:.2s}
.btn-p{background:var(--pri);color:#fff}.btn-p:hover{background:var(--pri-h)}
.btn-d{background:#fee2e2;color:#dc2626}.btn-d:hover{background:#fecaca}
.btn-c{background:#f3f4f6;color:var(--tx)}.btn-c:hover{background:#e5e7eb}
.btn-s{padding:4px 10px;font-size:12px}
.kb-list{flex:1;overflow-y:auto;padding:8px}
.kb-item{padding:12px;border-radius:8px;cursor:pointer;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;border:2px solid transparent}
.kb-item:hover{background:#f3f4f6}
.kb-item.act{background:#eef2ff;border-color:var(--pri)}
.kb-info h4{font-size:14px;margin-bottom:2px}
.kb-info p{font-size:12px;color:var(--tx2)}
.upload-zone{margin:12px;padding:20px;border:2px dashed var(--bdr);border-radius:8px;text-align:center;background:#fafafa;transition:.2s}
.upload-zone.over{border-color:var(--pri);background:#eef2ff}
.upload-zone p{font-size:13px;color:var(--tx2);margin-top:6px}
.upload-zone input{display:none}
.st-list{max-height:160px;overflow-y:auto;padding:0 12px 12px}
.st-item{font-size:12px;padding:6px 8px;background:#f9fafb;border-radius:4px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center}
.st-item.ok{color:#16a34a}.st-item.err{color:#dc2626}.st-item.ing{color:var(--pri)}

/* 主区域 */
.main{flex:1;display:flex;flex-direction:column}
.m-hd{padding:14px 24px;background:#fff;border-bottom:1px solid var(--bdr);display:flex;justify-content:space-between;align-items:center}
.m-hd h3{font-size:16px;display:flex;align-items:center;gap:8px}
.chat-box{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:20px}
.msg{display:flex;gap:12px;max-width:85%}.msg.u{align-self:flex-end;flex-direction:row-reverse}
.av{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;background:#e5e7eb}
.msg.u .av{background:var(--pri);color:#fff}
.bbl{padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.8;background:#fff;border:1px solid var(--bdr);box-shadow:0 1px 3px rgba(0,0,0,.05);word-break:break-word}
.msg.u .bbl{background:var(--pri);color:#fff;border:none}

/* ========== Markdown 格式化样式 ========== */
.bbl h1,.bbl h2,.bbl h3,.bbl h4{margin:16px 0 8px;font-weight:600;color:var(--tx)}
.bbl h1{font-size:20px;border-bottom:1px solid var(--bdr);padding-bottom:6px}
.bbl h2{font-size:18px}
.bbl h3{font-size:16px}
.bbl p{margin:8px 0}
.bbl ul,.bbl ol{margin:8px 0;padding-left:24px}
.bbl li{margin:4px 0}
.bbl strong{color:var(--pri);font-weight:600}
.bbl blockquote{margin:12px 0;padding:8px 16px;border-left:4px solid var(--pri);background:#f0f1ff;color:var(--tx2);border-radius:0 6px 6px 0}
.bbl a{color:var(--pri);text-decoration:underline}

/* 行内代码 */
.bbl code{background:#f3f4f6;padding:2px 6px;border-radius:4px;font-size:13px;color:#c7254e;font-family:'Menlo','Monaco','Courier New',monospace}

/* 代码块 */
.bbl pre{margin:12px 0;border-radius:8px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,.05)}
.bbl pre code{display:block;padding:16px;background:var(--code-bg);color:var(--code-tx);font-size:13px;line-height:1.6;overflow-x:auto;white-space:pre;font-family:'Menlo','Monaco','Courier New',monospace}

/* 表格 */
.bbl table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
.bbl th,.bbl td{border:1px solid var(--bdr);padding:8px 12px;text-align:left}
.bbl th{background:#f9fafb;font-weight:600}
.bbl tr:nth-child(even){background:#fafafa}

/* 分割线 */
.bbl hr{border:none;border-top:1px solid var(--bdr);margin:16px 0}

/* ========== 引用来源卡片样式 ========== */
.src-cards{margin-top:16px;padding-top:12px;border-top:1px dashed #d1d5db}
.src-title{font-size:13px;font-weight:600;color:var(--tx2);margin-bottom:8px;display:flex;align-items:center;gap:4px}
.src-card{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#f9fafb;border:1px solid var(--bdr);border-radius:6px;margin-bottom:6px;font-size:12px;transition:.2s}
.src-card:hover{background:#eef2ff;border-color:var(--pri)}
.src-left{display:flex;align-items:center;gap:8px;color:var(--tx)}
.src-icon{background:var(--pri);color:#fff;width:20px;height:20px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:11px}
.src-score{font-weight:600;color:var(--pri);background:#eef2ff;padding:2px 8px;border-radius:10px}

.inp-area{padding:16px 24px;background:#fff;border-top:1px solid var(--bdr);display:flex;gap:12px}
.inp-area textarea{flex:1;border:1px solid var(--bdr);border-radius:8px;padding:12px;resize:none;height:48px;outline:none;font-family:inherit;font-size:14px}
.inp-area textarea:focus{border-color:var(--pri)}
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--tx2)}
.empty h2{font-size:28px;margin-bottom:8px;color:var(--tx)}
.empty p{font-size:14px}

/* 光标闪烁 */
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.cursor-blink::after{content:'▌';animation:blink .8s infinite;color:var(--pri);margin-left:2px}

/* 模态框 */
.modal-bg{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.45);display:none;justify-content:center;align-items:center;z-index:1000}
.modal-bg.show{display:flex}
.modal{background:#fff;padding:24px;border-radius:12px;width:400px;box-shadow:0 20px 60px rgba(0,0,0,.15)}
.modal h3{margin-bottom:20px;font-size:18px}
.fg{margin-bottom:16px}
.fg label{display:block;margin-bottom:6px;font-size:14px;font-weight:500}
.fg input{width:100%;padding:10px 12px;border:1px solid var(--bdr);border-radius:6px;outline:none;font-size:14px}
.fg input:focus{border-color:var(--pri)}
.modal-acts{display:flex;justify-content:flex-end;gap:10px;margin-top:24px}
</style>
</head>
<body>

<!-- 侧边栏 -->
<div class="sidebar">
  <div class="sb-hd">
    <h2>📁 知识库</h2>
    <button class="btn btn-p btn-s" onclick="showModal()">＋ 新建</button>
  </div>
  <div class="kb-list" id="kbList"></div>
  <div class="upload-zone" id="upZone">
    <input type="file" id="fInput" multiple accept=".pdf,.txt,.md,.docx,.csv,.json" onchange="handleFiles(this.files)">
    <p style="font-size:24px;margin:0">⬆️</p>
    <p>拖拽文件到此处，或 <span style="color:var(--pri);cursor:pointer" onclick="document.getElementById('fInput').click()">点击上传</span></p>
    <p style="font-size:11px;margin-top:4px;color:#9ca3af">PDF / TXT / MD / DOCX / CSV / JSON</p>
  </div>
  <div class="st-list" id="stList"></div>
</div>

<!-- 主区域 -->
<div class="main">
  <div class="m-hd">
    <h3 id="kbTitle">📂 默认知识库</h3>
    <div style="display:flex;gap:8px;align-items:center">
      <label style="font-size:12px;color:var(--tx2);display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="streamToggle" checked style="accent-color:var(--pri)"> 流式输出
      </label>
      <button class="btn btn-c btn-s" onclick="clearChat()">🗑️ 清空对话</button>
    </div>
  </div>
  <div class="chat-box" id="chatBox">
    <div class="empty" id="emptyState">
      <h2>🧠 智能知识库助手</h2>
      <p>在左侧选择知识库并上传文档，然后开始提问</p>
    </div>
  </div>
  <div class="inp-area">
    <textarea id="qInput" placeholder="输入问题… (Enter发送, Shift+Enter换行)"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
    <button class="btn btn-p" onclick="send()" id="sendBtn">发送</button>
  </div>
</div>

<!-- 新建模态框 -->
<div class="modal-bg" id="modal">
  <div class="modal">
    <h3>🆕 新建知识库</h3>
    <div class="fg"><label>名称（英文/数字/下划线/中文）</label><input id="nkName" placeholder="例如：产品手册"></div>
    <div class="fg"><label>描述（可选）</label><input id="nkDesc" placeholder="例如：公司内部产品文档"></div>
    <div class="modal-acts">
      <button class="btn btn-c" onclick="hideModal()">取消</button>
      <button class="btn btn-p" onclick="createKB()">创建</button>
    </div>
  </div>
</div>

<script>
const API = location.origin;
let curKB = 'default';
let convId = null;
let sending = false;

marked.setOptions({ breaks: true, gfm: true });

// ========== 知识库管理（无变化） ==========
async function loadKBs() {
  try {
    const r = await fetch(API+'/api/kb/list');
    const j = await r.json();
    renderKBs(j.data || []);
  } catch(e) { console.error(e); }
}

function renderKBs(kbs) {
  document.getElementById('kbList').innerHTML = kbs.map(k => `
    <div class="kb-item ${k.name===curKB?'act':''}" onclick="switchKB('${k.name}','${k.description||''}')">
      <div class="kb-info"><h4>📂 ${k.name}</h4><p>${k.description||'无描述'} · ${k.document_count}篇文档</p></div>
      <div class="kb-actions" onclick="event.stopPropagation()">
        ${k.name!=='default'?`<button class="btn btn-d btn-s" onclick="deleteKB('${k.name}')">🗑</button>`:''}
      </div>
    </div>`).join('');
}

function switchKB(name, desc) {
  curKB = name; convId = null;
  document.getElementById('kbTitle').innerHTML = `📂 ${name}${desc?' · '+desc:''}`;
  document.getElementById('chatBox').innerHTML = `
    <div class="empty"><h2>已切换到 ${name}</h2><p>开始提问或继续上传文档</p></div>`;
  loadKBs();
}

function showModal() { document.getElementById('modal').classList.add('show'); }
function hideModal() { document.getElementById('modal').classList.remove('show'); }

async function createKB() {
  const name = document.getElementById('nkName').value.trim();
  const desc = document.getElementById('nkDesc').value.trim();
  if (!name) return alert('请输入名称');
  try {
    const r = await fetch(API+'/api/kb/create', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, description:desc})
    });
    const d = await r.json();
    if (d.code===200) { hideModal(); loadKBs(); switchKB(name, desc); }
    else alert(d.message||'创建失败');
  } catch(e) { alert('请求失败'); }
}

async function deleteKB(name) {
  if (!confirm(`确定删除「${name}」？不可恢复！`)) return;
  await fetch(API+'/api/kb/'+name, {method:'DELETE'});
  loadKBs();
  if (curKB===name) switchKB('default','默认知识库');
}

// ========== 文件上传（无变化） ==========
const upZone = document.getElementById('upZone');
upZone.addEventListener('dragover', e=>{e.preventDefault();upZone.classList.add('over')});
upZone.addEventListener('dragleave', ()=>upZone.classList.remove('over'));
upZone.addEventListener('drop', e=>{e.preventDefault();upZone.classList.remove('over');handleFiles(e.dataTransfer.files)});

async function handleFiles(files) {
  if (!files.length) return;
  for (const f of files) await uploadFile(f);
  document.getElementById('fInput').value='';
  setTimeout(loadKBs, 2500);
}

async function uploadFile(file) {
  const fd = new FormData(); fd.append('file',file); fd.append('knowledge_base',curKB);
  const tid = Date.now().toString();
  addStUI(tid, file.name, 'ing', '上传中…');
  try {
    const r = await fetch(API+'/api/upload', {method:'POST', body:fd});
    const d = await r.json();
    if (d.code===200 && d.data.task_id) {
      updStUI(tid, file.name, 'ing', '处理中…');
      pollSt(d.data.task_id, tid, file.name);
    } else { updStUI(tid, file.name, 'err', d.message||'失败'); }
  } catch(e) { updStUI(tid, file.name, 'err', '网络错误'); }
}

function pollSt(apiTid, uiTid, fname) {
  const iv = setInterval(async ()=>{
    try {
      const r = await fetch(API+'/api/upload/status/'+apiTid);
      const d = await r.json();
      if (d.data.status==='completed') { clearInterval(iv); updStUI(uiTid, fname, 'ok', `✓ ${d.data.info.final_chunks}块`); }
      else if (d.data.status==='failed') { clearInterval(iv); updStUI(uiTid, fname, 'err', d.data.message); }
    } catch(e) { clearInterval(iv); updStUI(uiTid, fname, 'err', '查询失败'); }
  }, 1500);
}

function addStUI(id,name,type,msg) {
  const el=document.createElement('div'); el.id='st-'+id; el.className='st-item '+type;
  const short = name.length>18 ? name.slice(0,18)+'…' : name;
  el.innerHTML=`<span>📄 ${short}</span><span>${msg}</span>`;
  document.getElementById('stList').prepend(el);
}
function updStUI(id,name,type,msg) {
  const el=document.getElementById('st-'+id);
  if(!el)return; el.className='st-item '+type;
  const short = name.length>18 ? name.slice(0,18)+'…' : name;
  el.innerHTML=`<span>📄 ${short}</span><span>${msg}</span>`;
}

// ========== 对话功能（核心修复区） ==========
function rmEmpty(){const e=document.getElementById('emptyState');if(e)e.remove()}

function buildSourcesHtml(sources) {
  if (!sources || !sources.length) return '';
  let cards = sources.map(s => {
    const pageStr = s.page ? ` · 第${s.page}页` : '';
    const scoreVal = parseFloat(s.score) || 0;
    let scoreColor = '#dc2626';
    if (scoreVal >= 0.8) scoreColor = '#16a34a';
    else if (scoreVal >= 0.6) scoreColor = '#d97706';
    return `<div class="src-card">
      <div class="src-left"><div class="src-icon">📄</div><span>${s.filename}${pageStr}</span></div>
      <div class="src-score" style="color:${scoreColor};background:${scoreColor}15">${scoreVal}</div>
    </div>`;
  }).join('');
  return `<div class="src-cards"><div class="src-title">📎 引用来源 (${sources.length})</div>${cards}</div>`;
}

function addMsg(role, content, isHtml = false) {
  rmEmpty();
  const c = document.getElementById('chatBox');
  const d = document.createElement('div'); 
  d.className = 'msg ' + (role==='user' ? 'u' : 'a');
  const displayContent = isHtml ? content : `<p style="white-space:pre-wrap;word-break:break-word">${content.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</p>`;
  d.innerHTML = `<div class="av">${role==='user'?'👤':'🤖'}</div><div class="bbl">${displayContent}</div>`;
  c.appendChild(d); c.scrollTop = c.scrollHeight;
}

// ⭐⭐⭐ 重写：健壮的流式解析器 ⭐⭐⭐
async function sendStream(question) {
  rmEmpty();
  const c = document.getElementById('chatBox');

  // 1. 创建 DOM 骨架，【不使用 id】，直接用变量持有引用，彻底避免多次提问的 ID 冲突
  const wrapper = document.createElement('div');
  wrapper.className = 'msg a';
  const bubble = document.createElement('div');
  bubble.className = 'bbl cursor-blink';
  wrapper.innerHTML = `<div class="av">🤖</div>`;
  wrapper.appendChild(bubble);
  c.appendChild(wrapper); 
  c.scrollTop = c.scrollHeight;

  let fullText = '';
  let sources = [];

  try {
    const resp = await fetch(API + '/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question, knowledge_base: curKB, top_k: 5, conversation_id: convId }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      // 追加原始字节流到缓冲区
      buffer += decoder.decode(value, { stream: true });

      // 按 "\n\n" 切割出【完整的 SSE 事件】
      // 这样无论 TCP 怎么拆包粘包，都能保证拿到完整的 JSON
      let boundary = buffer.indexOf('\n\n');
      
      while (boundary !== -1) {
        const eventBlock = buffer.substring(0, boundary);
        buffer = buffer.substring(boundary + 2); // 剩余部分放回 buffer

        // 解析当前事件块的 event 和 data
        let eventName = '';
        let dataStr = '';
        const lines = eventBlock.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventName = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            // 兼容多行 data
            dataStr += line.substring(6);
          }
        }

        // 分发处理
        if (eventName === 'sources') {
          const data = JSON.parse(dataStr);
          sources = data.sources || [];
          convId = data.conversation_id;
        } 
        else if (eventName === 'token') {
          const data = JSON.parse(dataStr);
          if (data.content) {
            fullText += data.content;
            // 实时渲染 Markdown
            bubble.innerHTML = marked.parse(fullText);
            c.scrollTop = c.scrollHeight;
          }
        } 
        else if (eventName === 'done') {
          const data = JSON.parse(dataStr);
          convId = data.conversation_id;
        } 
        else if (eventName === 'error') {
          const data = JSON.parse(dataStr);
          bubble.innerHTML = `<span style="color:red">❌ ${data.error}</span>`;
          fullText = ''; // 标记失败，最后不追加来源
        }

        // 继续查看 buffer 里还有没有下一个完整的 \n\n
        boundary = buffer.indexOf('\n\n');
      }
    }

    // 流正常结束
    bubble.classList.remove('cursor-blink');
    if (fullText) {
      bubble.innerHTML = marked.parse(fullText) + buildSourcesHtml(sources);
    }

  } catch(e) {
    bubble.classList.remove('cursor-blink');
    bubble.innerHTML = `<span style="color:red">❌ 网络或解析错误: ${e.message}</span>`;
  }
}

// 非流式发送（备用）
async function sendNormal(question) {
  addMsg('user', question);
  const c = document.getElementById('chatBox');
  const ld = document.createElement('div'); ld.className='msg a'; ld.id='ld';
  ld.innerHTML=`<div class="av">🤖</div><div class="bbl">思考中…</div>`;
  c.appendChild(ld); c.scrollTop=c.scrollHeight;
  try {
    const r = await fetch(API+'/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, knowledge_base:curKB, top_k:5, conversation_id:convId})
    });
    const d = await r.json();
    document.getElementById('ld')?.remove();
    if (d.answer) {
      convId = d.conversation_id;
      addMsg('assistant', marked.parse(d.answer) + buildSourcesHtml(d.sources), true);
    } else { addMsg('assistant', '❌ '+(d.detail||'错误')); }
  } catch(e) { document.getElementById('ld')?.remove(); addMsg('assistant','❌ '+e.message); }
}

// 统一发送入口
async function send() {
  const inp = document.getElementById('qInput');
  const q = inp.value.trim();
  if (!q || sending) return;
  
  sending = true;
  document.getElementById('sendBtn').disabled = true;
  inp.value = '';

  addMsg('user', q);

  try {
    if (document.getElementById('streamToggle').checked) {
      await sendStream(q);
    } else {
      await sendNormal(q);
    }
  } finally {
    // 用 finally 确保无论成功还是报错，都能解锁发送按钮
    sending = false;
    document.getElementById('sendBtn').disabled = false;
  }
}

function clearChat() {
  convId = null;
  document.getElementById('chatBox').innerHTML = `
    <div class="empty" id="emptyState"><h2>🧠 对话已清空</h2><p>请输入新问题</p></div>`;
}

window.onload = () => loadKBs();
</script>

</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
