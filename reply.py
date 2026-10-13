"""返信生成。会話から D(距離)/K(敬語)/T(温度)/E(絵文字)/R(返信頻度) を決め、制約付きで下書きを作る。

数値の向き: K 0=かたい敬語 1=丁寧語 2=軽い敬語混じり 3=タメ口可 / T 0=淡々 〜 4=好意を明言 /
E 0=なし 〜 3=多め / R 0=半日以上空ける 〜 3=すぐ返す。
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .errors import LLMError
from .llm import LLMClient, ask_json
from .schemas import ReplyDraft, ReplyRequest, ReplyResult, ReplyState, SendTiming
from .text_utils import (
    AFFECTION, CASUAL, COMPLIMENT, CONFESSION, CONTACT, EMOJI_RE, HARD_FORMAL, JST, NG_WORDS,
    PERSONAL_INFO, POSTPONE, REFUSAL, WORRY, MEET, casual_ratio, char_count, expressive_count,
    is_question, mask_pii, median, merge_utterances, reply_intervals, to_jst_naive,
)

CAP_K = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3}
CAP_T = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
CAP_E = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
HOBBIES = ["コーヒー", "カフェ", "映画", "旅行", "料理", "ランニング", "ヨガ", "読書", "音楽", "ライブ", "キャンプ",
           "登山", "ゲーム", "写真", "アニメ", "散歩", "ドライブ", "筋トレ", "サウナ", "お酒", "ワイン", "ラーメン"]
LABELS = {
    "first_message": ["丁寧", "親しみやすい", "落ち着いた"],
    "normal": ["素直に返す", "話題を広げる", "短く返す"],
    "closing": ["丁寧に区切る", "短くお礼"],
    "low_energy": ["短く返す", "軽い一言"],
}
DELAY = {
    3: ("5〜30分後", "相手がすぐ返してくれているため、近いペースで大丈夫です。"),
    2: ("30分〜3時間後", "相手の返信間隔に合わせています。"),
    1: ("1〜6時間後", "相手の返信間隔に合わせています。"),
    0: ("半日〜翌日", "相手の返信間隔に合わせ、急がなくて大丈夫です。"),
}


@dataclass
class Policy:
    D: int
    K: int
    T: int
    E: int
    R: int
    mode: str
    must_answer: bool = False
    allow_question: bool = True
    max_chars: int = 100
    flags: list[str] = field(default_factory=list)
    hobby: Optional[str] = None
    topic: Optional[str] = None


def find_hobby(text: str) -> Optional[str]:
    return next((h for h in HOBBIES if h in text), None)


def estimate_policy(req: ReplyRequest) -> Policy:
    utts = merge_utterances(req.messages)
    P = [u for u in utts if u.sender == "partner"]
    last5 = [u.text for u in P[-5:]]
    first = req.mode == "first_message" or not req.messages

    # D 距離
    n = len(req.messages)
    D = 0 if first else 1 if n < 6 else 2 if n < 20 else 3
    if req.dates_count >= 1:
        D = max(D, 3)
    if req.dates_count >= 3 and req.interest_level != "low":
        D = 4
    if req.interest_level == "low":
        D = min(D, 2)

    # K 敬語(相手に合わせる。距離の上限を超えない)
    if first or not last5:
        pk = 1
    elif sum(1 for t in last5 if HARD_FORMAL.search(t)) / len(last5) >= 0.4:
        pk = 0
    else:
        cr = casual_ratio(last5)
        pk = 3 if cr >= 0.6 else 2 if cr >= 0.3 else 1
    K = min(CAP_K[D], pk, req.keigo_max if req.keigo_max is not None else 3)

    # E 絵文字
    if first or not P:
        E = 0
    else:
        avg = sum(expressive_count(t) for t in last5) / len(last5)
        pe = 0 if avg == 0 else 1 if avg < 1 else 2 if avg < 2 else 3
        E = min(CAP_E[D], pe, req.emoji_max if req.emoji_max is not None else 3)

    # T 温度感(T4は本人が明示的に選び、距離が十分なときだけ)
    if first or not P:
        T = 1
    else:
        warm = sum(
            (2 if AFFECTION.search(t) else 0) + (1 if expressive_count(t) else 0) + (1 if is_question(t) else 0)
            for t in last5
        )
        pt = 0 if warm == 0 else 1 if warm <= 2 else 2 if warm <= 5 else 3
        cap = CAP_T[D]
        if req.temperature_max == 4 and req.allow_confession and D >= 3:
            cap = 4
        T = min(cap, pt + (1 if req.interest_level == "high" else 0))
    if req.temperature_max is not None:
        T = min(T, req.temperature_max)

    # R 返信頻度
    waits = reply_intervals(utts, "partner")
    if waits:
        m = median(waits)
        R = 3 if m <= 15 else 2 if m <= 120 else 1 if m <= 720 else 0
    else:
        R = 1

    # 分岐
    flags: list[str] = []
    last_text = P[-1].text if P and utts and utts[-1].sender == "partner" else ""
    mode = "first_message" if first else "normal"
    must_answer = bool(last_text) and is_question(last_text)
    allow_q = True
    if any(REFUSAL.search(u.text) for u in P[-2:]):
        mode, allow_q = "closing", False
        flags.append("clear_refusal")
    else:
        recent_me_meet = any(u.sender == "me" and MEET.search(u.text) for u in utts[-5:])
        if recent_me_meet and any(POSTPONE.search(u.text) for u in P[-2:]):
            flags.append("vague_decline")
        last3 = P[-3:]
        if len(last3) == 3 and all(char_count(u.text) <= 10 and not is_question(u.text) for u in last3):
            mode, allow_q = "low_energy", False
            flags.append("cold_streak")
    if last_text:
        if COMPLIMENT.search(last_text):
            flags.append("compliment")
        if WORRY.search(last_text):
            flags.append("worry")
        if PERSONAL_INFO.search(last_text):
            flags.append("personal_info")
    now = to_jst_naive(req.now) or datetime.now(JST).replace(tzinfo=None)
    if now.hour >= 23 or now.hour < 7:
        flags.append("late_night")

    avg_len = sum(char_count(u.text) for u in P) / len(P) if P else 60
    max_chars = 120 if first else 60 if mode == "closing" else max(30, min(140, round(avg_len * 1.3)))
    src = (req.partner_profile_text or "") + " " + last_text
    return Policy(D, K, T, E, R, mode, must_answer, allow_q, max_chars, flags, find_hobby(src), last_text.split("\n")[0][:30] or None)


# ---------- 下書きの検査 ----------
def check_draft(text: str, p: Policy) -> list[str]:
    v: list[str] = []
    if len(EMOJI_RE.findall(text)) > p.E:
        v.append(f"絵文字は{p.E}個まで")
    if NG_WORDS.search(text):
        v.append("押しつけ・断定・攻撃的な表現を含む")
    if p.T < 4 and CONFESSION.search(text):
        v.append("好意の明言は不可")
    if not p.allow_question and re.search(r"[?？]|ですか|ますか", text):
        v.append("質問は入れない")
    if char_count(text) > p.max_chars * 1.3:
        v.append(f"{p.max_chars}字程度に収める")
    if p.K <= 1 and CASUAL.search(text):
        v.append("タメ口は不可(丁寧語で)")
    if p.mode == "first_message" and CONTACT.search(text):
        v.append("初回で連絡先・他SNSの話題は不可")
    if mask_pii(text) != text:
        v.append("連絡先・URLを含めない")
    return v


# ---------- テンプレート(LLMなしのフォールバック) ----------
def _emoji(E: int) -> str:
    return ["", " 😊", " 😊✨", " 😊✨🎶"][E]


def template_drafts(p: Policy, n: int) -> list[tuple[str, str, Optional[str]]]:
    c = p.K >= 3
    e = _emoji(p.E)
    h = p.hobby
    out: list[tuple[str, str, Optional[str]]] = []
    if p.mode == "first_message":
        if h:
            out = [
                (f"はじめまして。プロフィールを拝見して、{h}がお好きなんだなと思い、ご連絡しました。最近よく行かれる場所はありますか?", "共通点から入る", h),
                (f"はじめまして!{h}が好きなんですね。私も興味があるので、おすすめがあれば教えてください{e}", "親しみを出す", h),
                (f"はじめまして。{h}のお話が気になってメッセージしました。最近印象に残ったことはありますか?", "落ち着いて質問する", h),
            ]
        else:
            out = [
                ("はじめまして。プロフィールを拝見して、お話ししてみたいなと思いご連絡しました。休日はどのように過ごされていますか?", "無難に始める", None),
                (f"はじめまして!プロフィール拝見しました。よろしくお願いします。気軽にお話しできたら嬉しいです{e}", "親しみを出す", None),
                ("はじめまして。プロフィールを読んで、落ち着いた雰囲気の方だなと思いました。よければお話ししてみませんか?", "落ち着いて始める", None),
            ]
    elif p.mode == "closing":
        out = [
            ("お返事ありがとうございます。お気持ち、わかりました。お話しできて楽しかったです。ありがとうございました。", "区切りをつける", None),
            ("ご連絡ありがとうございました。お元気でお過ごしください。", "短く終える", None),
        ]
    elif p.mode == "low_energy":
        out = [
            (("ありがとう。無理せずね" if c else "ありがとうございます。お時間のあるときで大丈夫です。"), "負担をかけない", None),
            (("了解!" if c else "了解です。また気が向いたときに。"), "軽く返す", None),
        ]
    else:
        lead = "お返事ありがとうございます。" if not c else "返信ありがとう。"
        if "worry" in p.flags:
            lead = "それは大変でしたね。無理せず休んでくださいね。" if not c else "それは大変だったね。無理せず休んでね。"
        elif "compliment" in p.flags:
            lead = "ありがとうございます、そう言っていただけて嬉しいです。" if not c else "ありがとう、そう言ってもらえて嬉しい。"
        if "personal_info" in p.flags:
            lead += "個人的な情報は、もう少しお互いを知ってからお伝えしたいです。" if not c else "個人的なことはもう少し仲良くなってからにしたいな。"
        ans = "【ご質問への答えをここに書いてください】" if p.must_answer else ""
        topic = h if (h := p.hobby) else None
        ask_topic = (f"{topic}について、もう少し聞かせてください。" if not c else f"{topic}のこと、もう少し聞きたいな。") if topic else (
            "最近はどんなふうに過ごしていますか?" if not c else "最近どう過ごしてる?")
        out = [
            (lead + ans + e, "お礼と回答", None),
            (lead + ans + (ask_topic if p.allow_question else "") + e, "話題を広げる", topic),
            (("ありがとう!" if c else "ありがとうございます。") + ans + e, "短く返す", None),
        ]
    return out[:n]


# ---------- LLM ----------
def build_system(p: Policy) -> str:
    kd = {0: "かたい敬語", 1: "丁寧語(です・ます)", 2: "軽い敬語混じり(ですます+くだけた語尾)", 3: "タメ口可"}[p.K]
    td = {0: "淡々と", 1: "穏やかに", 2: "やや親しく", 3: "親しく", 4: "好意を伝えてよい"}[p.T]
    rules = [
        f"敬語: {kd}。温度感: {td}。絵文字は{p.E}個まで(0なら使わない)。1通は{p.max_chars}字程度。",
        "相手の最新メッセージに質問があれば、まず答える。" if p.must_answer else "",
        "質問を入れない。短く、相手に負担をかけない。" if not p.allow_question else "",
        "相手が断っている。追わず、お礼と区切りだけを伝える。" if p.mode == "closing" else "",
        "相手の個人情報の質問には、軽く保留する(嘘はつかない)。" if "personal_info" in p.flags else "",
        "相手の不調には、まず気遣う。" if "worry" in p.flags else "",
        "初回メッセージ。連絡先・他SNSの話題と、外見への言及は避け、プロフィールの共通点を1つ拾って質問を1つだけ入れる。" if p.mode == "first_message" else "",
    ]
    return (
        "あなたは日本のマッチングアプリの返信下書きを作るアシスタントです。送信はユーザー自身が行います。\n"
        "守ること: 駆け引き・試し行動・罪悪感を使った誘導をしない。相手の気持ちを断定しない。事実を創作しない"
        "(不明な答えは【】で空欄にする)。連絡先やURLを書かない。\n"
        + "\n".join(r for r in rules if r)
        + '\n出力はJSONのみ: {"drafts":[{"label":"..","text":"..","intent":"..","common_point":null}]}'
    )


async def llm_drafts(llm: LLMClient, req: ReplyRequest, p: Policy, n: int, extra: str = "") -> list[tuple[str, str, Optional[str]]]:
    labels = LABELS[p.mode][:n]
    conv = "\n".join(f"[{'相手' if m.sender == 'partner' else '自分'}] {mask_pii(m.text)}" for m in req.messages[-20:])
    user = (
        f"相手のプロフィール:\n{mask_pii(req.partner_profile_text or '(なし)')}\n\n会話:\n{conv or '(まだなし)'}\n\n"
        f"{len(labels)}案を次のラベルで作ってください: {', '.join(labels)}\n{extra}"
    )

    def check(d: dict) -> None:
        if not isinstance(d["drafts"], list) or not d["drafts"]:
            raise ValueError("draftsが空")
        for x in d["drafts"]:
            if not isinstance(x["text"], str) or not x["text"].strip():
                raise ValueError("textが空")

    data = await ask_json(llm, build_system(p), user, check=check, max_tokens=1200)
    return [(x["text"].strip(), x.get("intent") or None, x.get("common_point") or None) for x in data["drafts"][:n]]


# ---------- 本体 ----------
def _timing(p: Policy) -> SendTiming:
    label, note = DELAY[p.R]
    if "late_night" in p.flags:
        note = "23時〜7時は、翌朝の送信をおすすめします。"
    return SendTiming(label=label, note=note)


def _cautions(p: Policy) -> list[str]:
    c: list[str] = []
    if p.mode == "first_message":
        c.append("初回でLINE交換などの話題は避けるのがおすすめです。")
    if "clear_refusal" in p.flags:
        c.append("相手のお断りの意思が読み取れます。追いかけず、区切りをつけるのがおすすめです。")
    if "vague_decline" in p.flags:
        c.append("誘いを濁されています。しばらく誘いは控えるのがおすすめです。")
    if "cold_streak" in p.flags:
        c.append("短い返事が続いています。質問を重ねず、相手のペースに合わせましょう。")
    if "personal_info" in p.flags:
        c.append("個人情報(住所・勤務先・連絡先など)は、信頼できると感じるまで伝えなくて大丈夫です。")
    return c


async def generate_reply(req: ReplyRequest, llm: Optional[LLMClient]) -> ReplyResult:
    p = estimate_policy(req)
    n = min(req.count, len(LABELS[p.mode]))
    labels = LABELS[p.mode]
    notes: list[str] = []
    drafts: list[tuple[str, str, Optional[str]]] = []
    source = "template"

    if llm is not None:
        try:
            cand = await llm_drafts(llm, req, p, n)
            good = [d for d in cand if not check_draft(d[0], p)]
            if len(good) < max(1, len(cand) // 2):
                viol = sorted({v for d in cand for v in check_draft(d[0], p)})
                cand = await llm_drafts(llm, req, p, n, extra="前回の違反: " + " / ".join(viol) + "。必ず守ってください。")
                good = [d for d in cand if not check_draft(d[0], p)]
            if good:
                drafts, source = good, "llm"
        except LLMError:
            notes.append("AIが使えなかったため、定型文で作成しています。")
    else:
        notes.append("AI未接続のため、定型文で作成しています。【】は自分で書き換えてください。")

    if not drafts:
        drafts = template_drafts(p, n)
    out = [
        ReplyDraft(label=labels[i], text=d[0], char_count=char_count(d[0]), intent=d[1], common_point=d[2])
        for i, d in enumerate(drafts[: len(labels)])
    ]
    return ReplyResult(
        mode=p.mode, state=ReplyState(distance=p.D, keigo=p.K, temperature=p.T, emoji=p.E, pace=p.R),
        drafts=out, send_timing=_timing(p), cautions=_cautions(p), source=source, notes=notes,
    )
