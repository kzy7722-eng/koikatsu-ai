"""脈あり度分析。5指標を数値化し、0〜100にまとめる(仕様書「脈あり度判定ロジック」準拠)。"""
import re
from typing import Optional

from .errors import LLMError
from .llm import LLMClient, ask_json
from .schemas import Evidence, InterestRequest, InterestResult, MetricOut
from .text_utils import (
    AFFECTION, CIRCUMSTANCE, DAILY, EMPATHY, MEET, POSTPONE, REFUSAL, SCAM,
    DATE_SPEC, Utterance, char_count, expressive_count, is_concrete_date,
    is_question, mask_pii, median, merge_utterances, reply_intervals,
)

METRIC_W = {"reply_speed": 0.15, "questions": 0.20, "emoji": 0.10, "length": 0.20, "interest": 0.35}
CHECK_W = {"meet_intent": 4, "self_disclosure": 3, "expand_topic": 3, "remembers": 3, "empathy": 2, "daily_share": 2}
CHECK_LABEL = {
    "meet_intent": "会う意向", "self_disclosure": "自己開示", "expand_topic": "話題を広げる",
    "remembers": "過去の話を覚えている", "empathy": "共感", "daily_share": "日常の共有",
}
BAND = {"high": "高め", "mid": "中くらい", "low": "低め"}
NEXT_ACTION = {
    "high": "日時と場所の候補を2つほど出して、軽く誘ってみる",
    "mid": "相手が話していた話題を1〜2往復深めてから、軽く誘ってみる",
    "low": "返信を急がず、相手のペースに合わせる。質問を重ねすぎない",
}
NOTE_LLM_FAIL = "AIによる関心度の判定が使えなかったため、ルールベースの簡易判定にしています。"


# ---------- 指標A〜D(ルールで計算) ----------
def metric_speed(utts: list[Utterance], circumstances: bool) -> MetricOut:
    waits = reply_intervals(utts, "partner")
    if len(waits) < 2:
        return MetricOut(status="not_applicable", note="送信時刻が足りないため、計算から除いています。")
    m = median(waits)
    for limit, sc in ((30, 90), (120, 75), (360, 60), (720, 45), (1440, 30)):
        if m <= limit:
            score = sc
            break
    else:
        score = 15
    note = f"相手の返信までの中央値は約{round(m)}分(夜間・平日日中は割り引いて計算)"
    if circumstances and score < 50:
        score = 50 + (score - 50) // 2
        note += "。仕事や体調などの事情が見えるため、低めの点を和らげています"
    return MetricOut(score=score, note=note)


def metric_questions(P: list[Utterance]) -> MetricOut:
    q = [u for u in P if is_question(u.text)]
    r = len(q) / len(P)
    score = 15 if not q else min(100, round(r * 150))
    if any(MEET.search(u.text) for u in q):
        score = min(100, score + 10)
    return MetricOut(score=score, note=f"相手の質問は{len(P)}発話中{len(q)}つ")


def metric_emoji(P: list[Utterance]) -> MetricOut:
    counts = [expressive_count(u.text) for u in P]
    if sum(counts) == 0:
        return MetricOut(status="not_applicable", note="絵文字・感情表現を使わない相手のため、計算から除いています。")
    rate = sum(counts) / len(P)
    return MetricOut(score=min(100, round(40 + rate * 40)), note=f"1発話あたり平均{rate:.1f}個")


def metric_length(P: list[Utterance], M: list[Utterance]) -> MetricOut:
    avg_p = sum(char_count(u.text) for u in P) / len(P)
    avg_m = sum(char_count(u.text) for u in M) / len(M) if M else 0
    if avg_m > 0:
        ratio = avg_p / avg_m
        score = 80 if ratio >= 1.0 else 65 if ratio >= 0.7 else 55 if ratio >= 0.5 else 40 if ratio >= 0.3 else 25
        note = f"相手の平均{avg_p:.0f}字 / 自分の平均{avg_m:.0f}字"
    else:
        score = 70 if avg_p >= 40 else 55 if avg_p >= 20 else 35
        note = f"相手の平均{avg_p:.0f}字"
    return MetricOut(score=score, note=note)


# ---------- 指標E(関心度チェックリスト) ----------
def _bigrams(s: str) -> set[str]:
    chars = re.findall(r"[一-龥ァ-ヴー]", s)
    return {a + b for a, b in zip(chars, chars[1:])}


def _count_to_v(n: int) -> int:
    return 2 if n >= 2 else 1 if n == 1 else 0


def heuristic_checks(utts: list[Utterance]) -> dict[str, dict]:
    """LLMなしのチェックリスト判定。{name: {value, quote}}"""
    res: dict[str, dict] = {k: {"value": 0, "quote": None} for k in CHECK_W}

    def first(pred) -> Optional[str]:
        for u in utts:
            if u.sender == "partner" and pred(u):
                return u.text.split("\n")[0][:40]
        return None

    p_texts = [u for u in utts if u.sender == "partner"]
    # 会う意向
    if any(MEET.search(u.text) and DATE_SPEC.search(u.text) for u in p_texts):
        res["meet_intent"] = {"value": 2, "quote": first(lambda u: bool(MEET.search(u.text)))}
    elif any(MEET.search(u.text) for u in p_texts):
        res["meet_intent"] = {"value": 1, "quote": first(lambda u: bool(MEET.search(u.text)))}
    # 自己開示
    sd = [u for u in p_texts if char_count(u.text) >= 50 and re.search(r"(私|自分|仕事|休日|普段|最近|好き)", u.text)]
    res["self_disclosure"] = {"value": _count_to_v(len(sd)), "quote": sd[0].text.split("\n")[0][:40] if sd else None}
    # 話題を広げる(質問が直前の自分の発話の語を拾っている)
    ex = []
    for prev, cur in zip(utts, utts[1:]):
        if cur.sender == "partner" and prev.sender == "me" and is_question(cur.text):
            if len(_bigrams(prev.text) & _bigrams(cur.text)) >= 1:
                ex.append(cur)
    res["expand_topic"] = {"value": _count_to_v(len(ex)), "quote": ex[0].text.split("\n")[0][:40] if ex else None}
    # 覚えている
    rem = [u for u in p_texts if re.search(r"(前に|言ってた|話してた|って言ってた|覚えて)", u.text)]
    res["remembers"] = {"value": _count_to_v(len(rem)), "quote": rem[0].text.split("\n")[0][:40] if rem else None}
    # 共感
    emp = [u for u in p_texts if EMPATHY.search(u.text) or AFFECTION.search(u.text)]
    res["empathy"] = {"value": _count_to_v(len(emp)), "quote": emp[0].text.split("\n")[0][:40] if emp else None}
    # 日常の共有
    day = [u for u in p_texts if DAILY.search(u.text)]
    res["daily_share"] = {"value": _count_to_v(len(day)), "quote": day[0].text.split("\n")[0][:40] if day else None}
    return res


def checks_to_score(checks: dict[str, dict]) -> int:
    got = sum(CHECK_W[k] * checks[k]["value"] for k in CHECK_W)
    return round(100 * got / (2 * sum(CHECK_W.values())))


SYSTEM_INTEREST = """あなたは日本語のチャット文面を読み、「相手(partner)が会話にどれだけ関心を示しているか」を観察項目で判定する分析器です。
相手の気持ちを断定してはいけません。文面に実際に現れている行動だけを根拠にします。
各項目を 0(見られない) / 1(ややある) / 2(はっきりある) で判定し、根拠になった相手の発言を一字一句そのまま quote に入れてください(なければ null)。
項目: meet_intent=会う意向を示す / self_disclosure=自分のことを自分から話す / expand_topic=こちらの話題を拾って広げる / remembers=過去の話を覚えている / empathy=共感や気遣い / daily_share=日常の出来事を共有する
出力はJSONのみ: {"checks":{"meet_intent":{"value":0,"quote":null}, ...6項目すべて}}"""


def _check_llm(data: dict) -> None:
    c = data["checks"]
    for k in CHECK_W:
        if int(c[k]["value"]) not in (0, 1, 2):
            raise ValueError(f"{k}.value は0/1/2")


async def llm_checks(llm: LLMClient, utts: list[Utterance]) -> dict[str, dict]:
    lines = [f"{i + 1}. [{'相手' if u.sender == 'partner' else '自分'}] {mask_pii(u.text)}" for i, u in enumerate(utts)]
    data = await ask_json(llm, SYSTEM_INTEREST, "会話:\n" + "\n".join(lines), check=_check_llm, max_tokens=900)
    out: dict[str, dict] = {}
    partner_text = "\n".join(u.text for u in utts if u.sender == "partner")
    for k in CHECK_W:
        v = int(data["checks"][k]["value"])
        q = data["checks"][k].get("quote")
        masked_text = mask_pii(partner_text)
        out[k] = {"value": v, "quote": q if isinstance(q, str) and q and q in masked_text else None}
    return out


# ---------- 本体 ----------
async def analyze_interest(req: InterestRequest, llm: Optional[LLMClient]) -> InterestResult:
    utts = merge_utterances(req.messages)
    P = [u for u in utts if u.sender == "partner"]
    M = [u for u in utts if u.sender == "me"]

    # 1) 詐欺・勧誘の兆候 → 点数を出さず注意
    scam = next((u for u in P if SCAM.search(u.text)), None)
    if scam:
        return InterestResult(
            status="warning", flags=["scam_suspected"],
            message="金銭や投資の話、外部サービスへの誘導は、マッチングアプリでの詐欺によくある手口と似ています。"
                    "やり取りを続ける前に、利用中のアプリの運営への通報も検討してください。",
            evidence=[Evidence(metric="warning", text="金銭・勧誘に関する話題が含まれています", quote=scam.text.split("\n")[0][:40])],
        )

    # 2) 情報不足
    if len(P) < 3:
        return InterestResult(status="insufficient", message="相手の発言が3通以上になると診断できます。", flags=["insufficient_data"])

    notes: list[str] = []
    circumstances = any(CIRCUMSTANCE.search(u.text) for u in P)

    metrics: dict[str, MetricOut] = {
        "reply_speed": metric_speed(utts, circumstances),
        "questions": metric_questions(P),
        "emoji": metric_emoji(P),
        "length": metric_length(P, M),
    }

    # 指標E
    source = "rules"
    if llm is not None:
        try:
            checks = await llm_checks(llm, utts)
            source = "llm"
        except LLMError:
            checks = heuristic_checks(utts)
            notes.append(NOTE_LLM_FAIL)
    else:
        checks = heuristic_checks(utts)
    metrics["interest"] = MetricOut(score=checks_to_score(checks), note="関心を示す行動6項目のチェックから算出")

    # 総合:対象外指標を除いて重みを再配分 → 発言数が少ないほど50に寄せる
    used = {k: m.score for k, m in metrics.items() if m.status == "ok" and m.score is not None}
    wsum = sum(METRIC_W[k] for k in used)
    raw = sum(METRIC_W[k] * s for k, s in used.items()) / wsum
    c = min(1.0, len(P) / 10)
    score = 50 + (raw - 50) * c

    # フラグ補正
    flags: list[str] = []
    evidence: list[Evidence] = []
    if circumstances:
        flags.append("circumstances")
    refusal = next((u for u in P[-3:] if REFUSAL.search(u.text)), None)
    postpone_n = sum(1 for u in P if POSTPONE.search(u.text))
    concrete = None
    for prev, cur in zip(utts, utts[1:]):
        if cur.sender == "partner" and is_concrete_date(cur.text, prev.text if prev.sender == "me" else None):
            concrete = cur
    if refusal:
        score = min(score, 20)
        flags.append("clear_refusal")
        evidence.append(Evidence(metric="interest", text="お断りの意思が読み取れる発言があります", quote=refusal.text.split("\n")[0][:40]))
    else:
        if postpone_n >= 2:
            score = min(score, 55)
            flags.append("vague_decline")
            evidence.append(Evidence(metric="interest", text=f"先延ばしの表現が{postpone_n}回あります"))
        if concrete:
            score = max(score, 70)
            flags.append("concrete_date")
            evidence.append(Evidence(metric="interest", text="日時を含む具体的な誘い・了承があります", quote=concrete.text.split("\n")[0][:40]))
    score_i = max(0, min(100, round(score)))
    level = "high" if score_i >= 70 else "mid" if score_i >= 50 else "low"

    # 根拠(質問・関心度)
    evidence.append(Evidence(metric="questions", text=metrics["questions"].note or ""))
    for k in sorted(CHECK_W, key=lambda x: -CHECK_W[x] * checks[x]["value"])[:2]:
        if checks[k]["value"] > 0:
            evidence.append(Evidence(metric="interest", text=f"{CHECK_LABEL[k]}の様子が見られます", quote=checks[k]["quote"]))

    # 信頼度
    speed_ok = metrics["reply_speed"].status == "ok"
    confidence = "high" if len(P) >= 15 and speed_ok else "mid" if len(P) >= 6 else "low"

    action = NEXT_ACTION[level]
    if "clear_refusal" in flags:
        action = "追いかけず、お礼を一言伝えて区切りをつけるのがおすすめです"
    elif "vague_decline" in flags:
        action = "しばらく誘いは控えて、相手が話しやすい話題で様子を見る"
    elif "concrete_date" in flags:
        action = "日時・場所・集合方法を確認して、約束を具体的に固める"

    return InterestResult(
        status="scored", score=score_i, level=level, band=BAND[level], confidence=confidence,
        metrics=metrics, evidence=evidence, flags=flags, next_action=action, source=source, notes=notes,
    )
