import logging
import os
import secrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import interest
import profile
import reply
from errors import APIError
from llm import LLMClient, get_llm
from schemas import (
    InterestRequest, InterestResponse, ProfileRequest, ProfileResponse, ReplyRequest, ReplyResponse,
)

log = logging.getLogger("koikatsu")

app = FastAPI(title="恋活AI API", version="0.1.0", description="プロフィール生成 / 返信生成 / 脈あり度分析")


def new_id(prefix: str = "gen") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """APP_API_TOKEN が設定されているときだけBearer認証を要求する。(TODO: 本番はSupabase等のJWT検証に置き換え)"""
    token = os.getenv("APP_API_TOKEN", "")
    if not token:
        return
    supplied = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    if not secrets.compare_digest(supplied, token):
        raise APIError(401, "unauthorized", "認証に失敗しました。")


api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_auth)])


@api.post("/generate/profile", response_model=ProfileResponse, response_model_exclude_none=True)
async def generate_profile(req: ProfileRequest, llm: Optional[LLMClient] = Depends(get_llm)):
    result = await profile.generate_profile(req, llm)
    return ProfileResponse(id=new_id(), result=result)


@api.post("/generate/reply", response_model=ReplyResponse, response_model_exclude_none=True)
async def generate_reply(req: ReplyRequest, llm: Optional[LLMClient] = Depends(get_llm)):
    if req.mode == "reply" and not req.messages:
        raise APIError(400, "invalid_input", "mode=reply のときは messages が必要です。")
    result = await reply.generate_reply(req, llm)
    return ReplyResponse(id=new_id(), result=result)


@api.post("/analyze/interest", response_model=InterestResponse, response_model_exclude_none=True)
async def analyze_interest(req: InterestRequest, llm: Optional[LLMClient] = Depends(get_llm)):
    result = await interest.analyze_interest(req, llm)
    return InterestResponse(id=new_id("ana"), result=result)


app.include_router(api)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "llm_backend": os.getenv("LLM_BACKEND", "mock")}


# ---------- エラー封筒 {error:{code,message}} ----------
def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return _err(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    details = jsonable_encoder([{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()])
    return _err(400, "invalid_input", "入力内容を確認してください。", details)


@app.exception_handler(StarletteHTTPException)
async def http_handler(_: Request, exc: StarletteHTTPException):
    return _err(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("unhandled error")  # 会話本文などの入力はログに出さない
    return _err(500, "internal_error", "サーバーでエラーが発生しました。")
