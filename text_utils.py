"""日本語テキストの共通ルール(正規表現・PIIマスク・発話の結合・実効待ち時間)。"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

JST = timezone(timedelta(hours=9))

# ---------- 正規表現 ----------
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")
KAOMOJI_RE = re.compile(r"[\(（][^\(\)（）\s]{0,8}[´`ωΘ∀Д▽\^＾><＞＜][^\(\)（）\s]{0,8}[\)）]")
LAUGH_RE = re.compile(r"(w{2,}|ｗ{2,}|[\(（]笑[\)）]|笑(?=[!！。\s]|$)|草)")
QUESTION_RE = re.compile(r"[?？]|(?:です|ます|でしょう|ました|ません)か|かな[。\s]*$|教えて(?:ください|ほしい|もらえ)")

HARD_FORMAL = re.compile(r"(でございます|ございました|ございません|おります|存じ|拝見|恐れ入り|お願い申し|いただけ(ます|れば))")
FORMAL = re.compile(r"(です|ます|でした|ました|ません|ください)")
CASUAL = re.compile(r"(だよ|だね|じゃん|だよね|だった|してる|ない[?？]|だわ|だろ|かな$|よ[!！]|ね[!！])")

REFUSAL = re.compile(
    r"(お付き合い|交際).{0,6}(できません|難しい|考えられません)"
    r"|会う(つもり|気)(は)?(ありません|ない)"
    r"|これ以上.{0,6}(連絡|メッセージ).{0,6}(やめ|控え|ご遠慮)"
    r"|ブロック|ごめんなさい.{0,10}(会えません|無理)|ご縁がなかった"
)
POSTPONE = re.compile(r"(また今度|落ち着いたら|そのうち|機会があれば|予定が合えば|忙しくて)")
MEET = re.compile(r"(会い(たい|ましょう|ませんか)|会って|会える|会えたら|会おう|ご飯|ランチ|ディナー|飲みに|飲みませ|お茶|行きましょう|行きませんか|行こう|デート)")
DATE_SPEC = re.compile(r"(\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|[月火水木金土日]曜|来週|今週末|週末|明日|あさって|\d{1,2}時|\d{1,2}:\d{2})")
ACCEPT = re.compile(r"(大丈夫|いいですね|行きたい|空いて|お願いします|ぜひ|楽しみ)")
SCAM = re.compile(r"(投資|仮想通貨|暗号資産|FX取引|副業|儲か|儲け|送金|振り込|振込|ギフトカード|プリペイド|必ず稼|テレグラム|telegram)", re.I)
COMPLIMENT = re.compile(r"(素敵|かわいい|可愛い|きれい|綺麗|かっこいい|優しい|魅力的|素晴らしい|すごい)(です|ね|!|！|$)")
WORRY = re.compile(r"(疲れ|しんどい|つらい|辛い|落ち込|不安|悩み|体調|風邪|熱が)")
PERSONAL_INFO = re.compile(r"(本名|フルネーム|住所|どこに住|最寄り駅|勤務先|会社名|電話番号|連絡先|年収)")
EMPATHY = re.compile(r"(わかります|分かります|そうなんですね|そうなんだ|たしかに|確かに|いいですね|いいな|素敵ですね|大変でしたね|お疲れ様)")
DAILY = re.compile(r"(今日|さっき|昨日|今朝|今夜)")
AFFECTION = re.compile(r"(会いたい|楽しみ|嬉しい|うれしい|好き|また話したい)")
CIRCUMSTANCE = re.compile(r"(仕事が忙しい|繁忙期|出張|試験|体調|残業|締め切り|締切)")
CONTACT = re.compile(r"(LINE|ライン|連絡先|電話|インスタ|Instagram|ID)", re.I)
NG_WORDS = re.compile(r"(絶対|必ず|運命|会わないと|返事(を)?ください|なんで返信|既読|彼女になって|付き合ってください|愛して|結婚してください|ブス|デブ)")
CONFESSION = re.compile(r"(好きです|付き合って|愛して)")


# ---------- 文字列ユーティリティ ----------
def char_count(text: str) -> int:
    return len(re.sub(r"\s", "", text))


def is_question(text: str) -> bool:
    return bool(QUESTION_RE.search(text))


def expressive_count(text: str) -> int:
    """絵文字+顔文字+笑い表現の数(「絵文字・感情表現」指標用)。"""
    return len(EMOJI_RE.findall(text)) + len(KAOMOJI_RE.findall(text)) + len(LAUGH_RE.findall(text))


def emoji_count(text: str) -> int:
    return len(EMOJI_RE.findall(text))


def is_concrete_date(text: str, prev_other_text: Optional[str] = None) -> bool:
    """会う話が日時つきで具体化しているか(自分から誘う/誘いを受ける)。"""
    if MEET.search(text) and DATE_SPEC.search(text):
        return True
    if prev_other_text and MEET.search(prev_other_text) and DATE_SPEC.search(text) and ACCEPT.search(text):
        return True
    return False


def casual_ratio(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    n = sum(1 for t in texts if CASUAL.search(t) or not FORMAL.search(t))
    return n / len(texts)


def mask_pii(text: str) -> str:
    """LLMに渡す前に連絡先等を伏せ字にする。"""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[メール]", text)
    text = re.sub(r"https?://\S+", "[URL]", text)

    def _phone(m: "re.Match[str]") -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[電話番号]" if 10 <= len(digits) <= 11 else m.group(0)

    text = re.sub(r"(?<!\d)0\d{1,4}[-ー ]?\d{1,4}[-ー ]?\d{3,4}(?!\d)", _phone, text)
    text = re.sub(r"(?i)(?:line|ライン)\s*(?:id|アイディー)?\s*[:：は]?\s*[A-Za-z0-9._-]{4,}", "[ID]", text)
    return text


# ---------- 発話の結合・待ち時間 ----------
@dataclass
class Utterance:
    sender: str
    text: str
    at: Optional[datetime] = None
    end: Optional[datetime] = None
    count: int = 1


def to_jst_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(JST).replace(tzinfo=None)


def merge_utterances(msgs: Iterable, gap_min: int = 5) -> list[Utterance]:
    """同一送信者の連投(gap_min分以内)を1発話にまとめる。"""
    out: list[Utterance] = []
    for m in msgs:
        at = to_jst_naive(m.at)
        if (
            out
            and out[-1].sender == m.sender
            and at is not None
            and out[-1].end is not None
            and 0 <= (at - out[-1].end).total_seconds() <= gap_min * 60
        ):
            u = out[-1]
            u.text += "\n" + m.text
            u.end = at
            u.count += 1
        else:
            out.append(Utterance(m.sender, m.text, at, at))
    return out


def effective_wait_minutes(start: datetime, end: datetime) -> float:
    """実効待ち時間(分)。23:00-7:00は0、平日9-18時は0.5倍で数える(5分刻み)。"""
    if end <= start:
        return 0.0
    end = min(end, start + timedelta(days=14))
    total = 0.0
    t = start
    step = timedelta(minutes=5)
    while t < end:
        nxt = min(t + step, end)
        frac = (nxt - t).total_seconds() / 300
        h = t.hour
        if h >= 23 or h < 7:
            w = 0.0
        elif t.weekday() < 5 and 9 <= h < 18:
            w = 0.5
        else:
            w = 1.0
        total += w * 5 * frac
        t = nxt
    return total


def reply_intervals(utts: Sequence[Utterance], responder: str) -> list[float]:
    """responder が相手の発話に返すまでの実効待ち時間(分)の一覧。"""
    waits: list[float] = []
    for prev, cur in zip(utts, utts[1:]):
        if cur.sender == responder and prev.sender != responder and prev.end and cur.at:
            waits.append(effective_wait_minutes(prev.end, cur.at))
    return waits


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
