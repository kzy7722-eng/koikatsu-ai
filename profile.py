"""プロフィール生成。LLMがあれば3案を生成して検証、なければテンプレートで組み立てる。"""
from typing import Optional

from .errors import LLMError
from .llm import LLMClient, ask_json
from .schemas import MyProfile, ProfileDraft, ProfileRequest, ProfileResult
from .text_utils import char_count, mask_pii

TONES = ["sincere", "bright", "intellectual"]
TONE_LABEL = {"sincere": "誠実系", "bright": "明るい系", "intellectual": "知的系"}
TONE_INTENT = {
    "sincere": "仕事への姿勢と休日の過ごし方を並べて、誠実さが伝わるようにしています",
    "bright": "好きなことを前向きに伝えて、話しかけやすい雰囲気にしています",
    "intellectual": "関心のあることと考え方を落ち着いて伝え、会話の入口を作っています",
}
GREET = {
    "sincere": "はじめまして。プロフィールをご覧いただきありがとうございます。",
    "bright": "はじめまして!ご覧いただきありがとうございます。",
    "intellectual": "はじめまして。プロフィールを読んでいただきありがとうございます。",
}
CLOSE = {
    "sincere": "まずはメッセージのやり取りから、ゆっくり仲良くなれたら嬉しいです。",
    "bright": "気軽にメッセージくださいね!",
    "intellectual": "お互いの考えを話せる関係を築けたら嬉しいです。",
}


def ja_join(items: list[str]) -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]}や{items[1]}"
    return "、".join(items[:-1]) + f"や{items[-1]}"


def sentences(me: MyProfile, tone: str) -> list[str]:
    s = [GREET[tone]]
    if me.occupation and me.area:
        s.append(f"{me.area}で{me.occupation}として働いています。")
    elif me.occupation:
        s.append(f"{me.occupation}として働いています。")
    elif me.area:
        s.append(f"{me.area}に住んでいます。")
    if me.hobbies:
        h = ja_join(me.hobbies)
        s.append({
            "sincere": f"休日は{h}を楽しんでいます。",
            "bright": f"{h}が大好きで、休日はよく出かけています。",
            "intellectual": f"{h}に関心があり、最近も少しずつ深めています。",
        }[tone])
    if me.values:
        s.append(f"大切にしているのは、{ja_join(me.values)}です。")
    if me.relationship_goal:
        s.append(f"{me.relationship_goal}を希望しています。")
    s.append(CLOSE[tone])
    return s


def fit_sentences(parts: list[str], max_chars: int) -> str:
    """順番を保ったまま、文字数上限に収まる文だけを採用する(締めの一文を優先して残す)。"""
    close = parts[-1]
    body = parts[:-1]
    out: list[str] = []
    budget = max_chars - len(close)
    for p in body:
        if len("".join(out)) + len(p) <= budget:
            out.append(p)
    text = "".join(out) + close
    if len(text) > max_chars:  # 締めだけでも超える場合
        text = "".join(out) or close
        text = text[: max_chars - 1] + "。" if len(text) > max_chars else text
    return text


def template_profile(req: ProfileRequest) -> list[ProfileDraft]:
    order = [req.tone] + [t for t in TONES if t != req.tone]
    drafts = []
    for t in order:
        text = fit_sentences(sentences(req.me, t), req.max_chars)
        drafts.append(ProfileDraft(tone=t, tone_label=TONE_LABEL[t], text=text, char_count=len(text), intent=TONE_INTENT[t]))
    return drafts


SYSTEM = """あなたは日本のマッチングアプリの自己紹介文を書くアシスタントです。
守ること:
- 入力された事実だけを使い、年収・学歴・外見・経歴などを創作しない。誇張しない。
- 連絡先、SNSのID、URLは書かない。
- 「真面目です」のような自己評価の断定より、具体的な行動や好きなことで人柄が伝わるようにする。
- 文字数は指定の上限以内(空白を除く)。
- 3つの印象タイプ(sincere=誠実系, bright=明るい系, intellectual=知的系)を1案ずつ書き分ける。
出力はJSONのみ: {"drafts":[{"tone":"sincere","text":"..","intent":"この案の狙いを1文で"}, ...3件]}"""


def _facts(req: ProfileRequest) -> str:
    me = req.me
    rows = {
        "年代": me.age_range, "職業": me.occupation, "エリア": me.area,
        "趣味": "、".join(me.hobbies), "大切にしていること": "、".join(me.values), "探している関係": me.relationship_goal,
    }
    return "\n".join(f"{k}: {mask_pii(v)}" for k, v in rows.items() if v)


async def generate_profile(req: ProfileRequest, llm: Optional[LLMClient]) -> ProfileResult:
    notes: list[str] = []
    if llm is not None:
        def check(d: dict) -> None:
            tones = [x["tone"] for x in d["drafts"]]
            if sorted(tones) != sorted(TONES):
                raise ValueError("3つのtoneを1案ずつ")
            for x in d["drafts"]:
                n = char_count(x["text"])
                if not (0 < n <= req.max_chars):
                    raise ValueError(f"{x['tone']}が{n}字(上限{req.max_chars}字)")
                if mask_pii(x["text"]) != x["text"]:
                    raise ValueError("連絡先やURLを含めない")

        user = f"文字数上限: {req.max_chars}字\n特に重視する印象: {TONE_LABEL[req.tone]}\n使うアプリ: {req.app_name or '指定なし'}\n\n事実:\n{_facts(req)}"
        try:
            data = await ask_json(llm, SYSTEM, user, check=check, max_tokens=1500)
            by = {x["tone"]: x for x in data["drafts"]}
            order = [req.tone] + [t for t in TONES if t != req.tone]
            drafts = [
                ProfileDraft(tone=t, tone_label=TONE_LABEL[t], text=by[t]["text"].strip(),
                             char_count=char_count(by[t]["text"]), intent=by[t].get("intent") or TONE_INTENT[t])
                for t in order
            ]
            return ProfileResult(drafts=drafts, source="llm", notes=notes)
        except LLMError:
            notes.append("AIが使えなかったため、入力内容からテンプレートで作成しています。")
    else:
        notes.append("AI未接続のため、入力内容からテンプレートで作成しています。")
    if not (req.me.hobbies or req.me.occupation or req.me.values):
        notes.append("職業・趣味・大切にしていることのどれかを入力すると、内容が具体的になります。")
    return ProfileResult(drafts=template_profile(req), source="template", notes=notes)
