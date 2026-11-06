"""Optional device landing and token-scoped callback, independent of the API key."""

import json
import re
import secrets

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.dependencies import get_session
from app.core.config import get_settings
from app.services.nextcloud.device_collection import capture_computer, collection_status, public_base, reject

router = APIRouter()
HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"}

_LANDING = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AI Support — имя компьютера</title><style>body{font:18px system-ui;max-width:650px;margin:4em auto;padding:1em;line-height:1.5}a{display:inline-block;padding:.6em 1em;background:#173b64;color:white;border-radius:6px}</style>
<h1>Определение рабочего компьютера</h1><p>Будет передано только имя этого компьютера в ваше текущее обращение технической поддержки. Документы, пароли и другие сведения не собираются.</p>
<p id="status">Проверяем ссылку…</p><a id="launch" hidden>Определить имя компьютера</a>
<p>Браузер может запросить разрешение открыть установленный корпоративный помощник. Разрешение даёте вы. После определения компьютера вернитесь в чат.</p>
<script nonce="__NONCE__">(()=>{"use strict";
const match=/^#token=([A-Za-z0-9_-]{43})$/.exec(location.hash);
history.replaceState(null,"",location.pathname);
const status=document.getElementById("status"),button=document.getElementById("launch");
if(!match){status.textContent="Ссылка недействительна. Вернитесь в чат за новой ссылкой.";return;}
const token=match[1],endpoint=location.pathname.replace(/\\/device\\/collect\\/?$/,"/api/device-collection");
button.href="ai-support://collect/"+token;
let finished=false,attempted=false,checks=0;
async function check(){
 if(finished||checks++>=150){return;}
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);
 try{
  const response=await fetch(endpoint,{headers:{Authorization:"Bearer "+token},cache:"no-store",credentials:"omit",redirect:"error",referrerPolicy:"no-referrer",signal:controller.signal});
  if(!response.ok){finished=true;button.hidden=true;button.removeAttribute("href");status.textContent="Ссылка истекла или отменена. Вернитесь в чат за новой ссылкой.";return;}
  const value=await response.json();
  if(value.status==="complete"){finished=true;button.hidden=true;button.removeAttribute("href");status.textContent="Имя компьютера получено. Вернитесь в чат технической поддержки.";return;}
  if(value.status!=="pending"){throw Error();}
  button.hidden=false;status.textContent="Если помощник не открылся, нажмите кнопку ниже и подтвердите запрос браузера.";
  if(!attempted){attempted=true;try{location.assign(button.href);}catch(_){}}
 }catch(_){status.textContent="Не удалось проверить ответ. Можно повторить открытие помощника кнопкой или вернуться в чат.";}
 finally{clearTimeout(timer);if(!finished){setTimeout(check,2000);}}
}
check();})();</script></html>"""


def _bearer(request):
    if request.url.query:
        reject()
    header = request.headers.get("authorization", "")
    match = re.fullmatch(r"Bearer ([A-Za-z0-9_-]{43})", header, re.I)
    if match is None:
        reject()
    return match[1]


def _capture_object(pairs):
    if len(pairs) != 1 or pairs[0][0] != "computer_name":
        raise ValueError("Invalid capture shape")
    return dict(pairs)


@router.get("/device/collect", response_class=HTMLResponse)
def landing(settings=Depends(get_settings)):
    public_base(settings)
    nonce = secrets.token_urlsafe(18)
    return HTMLResponse(_LANDING.replace("__NONCE__", nonce), headers={**HEADERS,
        "Content-Security-Policy": "default-src 'none'; script-src 'nonce-" + nonce
        + "'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
    })


@router.get("/api/device-collection")
def status(request: Request, session=Depends(get_session), settings=Depends(get_settings)):
    return JSONResponse(collection_status(session, settings, _bearer(request)), headers=HEADERS)


@router.post("/api/device-collection")
async def capture(request: Request, session=Depends(get_session), settings=Depends(get_settings)):
    token = _bearer(request)
    raw = bytearray()
    async for part in request.stream():
        raw.extend(part)
        if len(raw) > 2048:
            reject()
    try:
        data = json.loads(raw, object_pairs_hook=_capture_object)
    except (ValueError, UnicodeError):
        reject()
    if not isinstance(data, dict) or set(data) != {"computer_name"}:
        reject()
    return JSONResponse(capture_computer(session, settings, token, data["computer_name"]), headers=HEADERS)
