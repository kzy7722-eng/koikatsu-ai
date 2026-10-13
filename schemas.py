from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER_DRAFT = "下書きです。事実と違う点がないかご自身で確認し、送信はご自身で行ってください。"
DISCLAIMER_INTEREST = "文面からの推測であり、相手の気持ちを断定するものではありません。"


# ---------- 共通 ----------
class ChatMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sender: Literal["partner", "me"] = Field(alias="from", description="partner=相手 / me=自分")
    text: str = Field(min_length=1, max_length=2000)
    at: Optional[datetime] = Field(default=None, description="送信時刻(任意。あると返信速度を見る)")


# ---------- プロフィール生成 ----------
class MyProfile(BaseModel):
    age_range: Optional[str] = Field(default=None, max_length=20, examples=["30代前半"])
    occupation: Optional[str] = Field(default=None, max_length=40)
    hobbies: list[str] = Field(default_factory=list, max_length=5)
    values: list[str] = Field(default_factory=list, max_length=5)
    relationship_goal: Optional[str] = Field(default=None, max_length=60)
    area: Optional[str] = Field(default=None, max_length=30)


class ProfileRequest(BaseModel):
    app_name: Optional[str] = Field(default=None, max_length=30)
    max_chars: int = Field(default=200, ge=50, le=800)
    tone: Literal["sincere", "bright", "intellectual"] = "sincere"
    me: MyProfile


class ProfileDraft(BaseModel):
    tone: Literal["sincere", "bright", "intellectual"]
    tone_label: str
    text: str
    char_count: int
    intent: Optional[str] = None


class ProfileResult(BaseModel):
    drafts: list[ProfileDraft]
    source: Literal["llm", "template"]
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_DRAFT


class ProfileResponse(BaseModel):
    id: str
    type: Literal["profile"] = "profile"
    result: ProfileResult


# ---------- 返信生成 ----------
class ReplyRequest(BaseModel):
    mode: Literal["first_message", "reply"] = "reply"
    partner_profile_text: Optional[str] = Field(default=None, max_length=3000)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=100)
    me: Optional[MyProfile] = None
    # 上限設定: 数値が大きいほど砕ける/温度が高い/絵文字が多い。未指定=自動
    keigo_max: Optional[int] = Field(default=None, ge=0, le=3, description="0=かたい敬語 1=丁寧語 2=軽い敬語混じり 3=タメ口可")
    emoji_max: Optional[int] = Field(default=None, ge=0, le=3)
    temperature_max: Optional[int] = Field(default=None, ge=0, le=4)
    allow_confession: bool = False
    dates_count: int = Field(default=0, ge=0, le=50)
    date_scheduled: bool = False
    interest_level: Optional[Literal["low", "mid", "high"]] = None
    count: int = Field(default=3, ge=1, le=5)
    now: Optional[datetime] = Field(default=None, description="現在時刻(テスト/送信タイミング用。未指定=サーバ時刻JST)")


class ReplyState(BaseModel):
    distance: int
    keigo: int
    temperature: int
    emoji: int
    pace: int


class ReplyDraft(BaseModel):
    label: str
    text: str
    char_count: int
    intent: Optional[str] = None
    common_point: Optional[str] = None


class SendTiming(BaseModel):
    label: str
    note: Optional[str] = None


class ReplyResult(BaseModel):
    mode: Literal["first_message", "normal", "closing", "low_energy"]
    state: ReplyState
    drafts: list[ReplyDraft]
    send_timing: SendTiming
    cautions: list[str] = Field(default_factory=list)
    source: Literal["llm", "template"]
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_DRAFT


class ReplyResponse(BaseModel):
    id: str
    type: Literal["reply"] = "reply"
    result: ReplyResult


# ---------- 脈あり度分析 ----------
class InterestRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)


class MetricOut(BaseModel):
    score: Optional[int] = None
    status: Literal["ok", "not_applicable"] = "ok"
    note: Optional[str] = None


class Evidence(BaseModel):
    metric: str
    text: str
    quote: Optional[str] = None


class InterestResult(BaseModel):
    status: Literal["scored", "insufficient", "warning"]
    score: Optional[int] = None
    level: Optional[Literal["low", "mid", "high"]] = None
    band: Optional[str] = None
    confidence: Optional[Literal["low", "mid", "high"]] = None
    metrics: dict[str, MetricOut] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    next_action: Optional[str] = None
    message: Optional[str] = None
    source: Literal["llm", "rules"] = "rules"
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_INTEREST


class InterestResponse(BaseModel):
    id: str
    type: Literal["interest"] = "interest"
    result: InterestResult
