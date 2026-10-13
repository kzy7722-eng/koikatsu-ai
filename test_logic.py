import asyncio
from datetime import datetime

from app import interest, profile, reply
from app.schemas import ChatMessage, InterestRequest, MyProfile, ProfileRequest, ReplyRequest
from app.text_utils import effective_wait_minutes, mask_pii


def run(coro):
    return asyncio.run(coro)


def M(sender, text, at=None):
    return ChatMessage(**{"from": sender, "text": text, "at": datetime.fromisoformat(at) if at else None})


# ---------- text_utils ----------
def test_effective_wait_skips_night():
    # 月曜22:30 → 火曜07:05 : 22:30-23:00 の30分 + 07:00-07:05 の5分
    assert effective_wait_minutes(datetime(2026, 10, 12, 22, 30), datetime(2026, 10, 13, 7, 5)) == 35


def test_effective_wait_weekday_daytime_halved():
    assert effective_wait_minutes(datetime(2026, 10, 12, 10, 0), datetime(2026, 10, 12, 11, 0)) == 30


def test_mask_pii():
    s = mask_pii("連絡は 090-1234-5678 か a.b@example.com、https://x.example/y まで")
    assert "090" not in s and "@" not in s and "http" not in s


# ---------- interest ----------
HIGH = [
    M("me", "休日はカフェ巡りをしています", "2026-10-10 10:00"),
    M("partner", "いいですね!私もカフェ好きです😊 最近はどこに行きましたか?", "2026-10-10 10:10"),
    M("me", "先週、蔵前のお店に行きました", "2026-10-10 11:00"),
    M("partner", "蔵前いいですよね!今日も実は散歩してきました。普段は仕事が忙しいので、休日はゆっくりしたくて(笑)", "2026-10-10 11:12"),
    M("me", "わかります、のんびりしたいですよね", "2026-10-10 12:00"),
    M("partner", "そうなんです😊 蔵前のお店、おすすめはありますか?", "2026-10-10 12:08"),
    M("me", "コーヒーが美味しいお店があります", "2026-10-10 13:00"),
    M("partner", "気になります!今度一緒に行きませんか?来週の土曜日なら空いています", "2026-10-10 13:10"),
]


def test_interest_concrete_date_floor():
    r = run(interest.analyze_interest(InterestRequest(messages=HIGH), None))
    assert r.status == "scored"
    assert r.score >= 70 and r.level == "high"
    assert "concrete_date" in r.flags
    assert r.metrics["reply_speed"].status == "ok"


def test_interest_insufficient():
    r = run(interest.analyze_interest(InterestRequest(messages=HIGH[:3]), None))
    assert r.status == "insufficient" and r.score is None


def test_interest_scam_warning():
    msgs = [M("me", "こんにちは"), M("partner", "こんにちは"), M("me", "お仕事は?"),
            M("partner", "いい投資の話があるので、このサイトに登録してみませんか"), M("me", "えっと"), M("partner", "絶対儲かります")]
    r = run(interest.analyze_interest(InterestRequest(messages=msgs), None))
    assert r.status == "warning" and r.score is None


def test_interest_refusal_cap():
    msgs = [M("me", "週末どうですか"), M("partner", "ごめんなさい。会うつもりはありません"), M("me", "そうですか"),
            M("partner", "はい"), M("me", "わかりました"), M("partner", "すみません")]
    r = run(interest.analyze_interest(InterestRequest(messages=msgs), None))
    assert r.status == "scored" and r.score <= 20


def test_interest_emoji_not_applicable():
    msgs = [M("me", "こんにちは"), M("partner", "こんにちは。よろしくお願いします。"), M("me", "趣味は?"),
            M("partner", "読書です。あなたは?"), M("me", "映画です"), M("partner", "いいですね。最近は何を観ましたか?")]
    r = run(interest.analyze_interest(InterestRequest(messages=msgs), None))
    assert r.metrics["emoji"].status == "not_applicable"


# ---------- reply ----------
def test_first_message_policy_and_draft():
    req = ReplyRequest(mode="first_message", partner_profile_text="趣味はコーヒー巡りです。映画も好き。")
    res = run(reply.generate_reply(req, None))
    assert res.mode == "first_message"
    assert (res.state.distance, res.state.keigo, res.state.emoji) == (0, 1, 0)
    assert len(res.drafts) == 3
    assert "コーヒー" in res.drafts[0].text
    assert "LINE" not in "".join(d.text for d in res.drafts)


def test_closing_on_refusal():
    req = ReplyRequest(messages=[M("me", "今度会いませんか?"), M("partner", "ごめんなさい、会うつもりはありません")])
    res = run(reply.generate_reply(req, None))
    assert res.mode == "closing"
    assert all("?" not in d.text and "?" not in d.text for d in res.drafts)


def test_cold_streak_no_questions():
    msgs = [M("me", "週末は何してましたか?"), M("partner", "家です"), M("me", "のんびりですね"), M("partner", "はい"),
            M("me", "映画とか観ます?"), M("partner", "たまに")]
    res = run(reply.generate_reply(ReplyRequest(messages=msgs), None))
    assert res.mode == "low_energy"
    assert all("?" not in d.text and "?" not in d.text for d in res.drafts)


def test_emoji_max_respected():
    msgs = [M("me", "こんにちは😊"), M("partner", "こんにちは😊✨ よろしくね!"), M("me", "趣味は?"),
            M("partner", "ライブ行くの好き🎶 そっちは?"), M("me", "映画かな"), M("partner", "いいね〜😊 おすすめある?")]
    res = run(reply.generate_reply(ReplyRequest(messages=msgs, emoji_max=0), None))
    assert res.state.emoji == 0
    assert all(not reply.EMOJI_RE.search(d.text) for d in res.drafts)


def test_keigo_capped_at_early_distance():
    msgs = [M("me", "はじめまして"), M("partner", "はじめまして!よろしくね。趣味は映画だよ"), M("me", "僕もです"),
            M("partner", "そうなんだ!嬉しいな")]
    res = run(reply.generate_reply(ReplyRequest(messages=msgs), None))
    assert res.state.distance == 1 and res.state.keigo <= 1


def test_late_night_timing():
    req = ReplyRequest(messages=[M("partner", "おやすみなさい")], now=datetime(2026, 10, 13, 23, 30))
    res = run(reply.generate_reply(req, None))
    assert "翌朝" in (res.send_timing.note or "")


# ---------- profile ----------
def test_profile_fits_max_chars():
    me = MyProfile(age_range="30代前半", occupation="エンジニア", hobbies=["コーヒー", "映画", "ランニング"],
                   values=["誠実さ", "対話"], relationship_goal="真剣な交際", area="松山")
    for mc in (50, 100, 200, 400):
        res = run(profile.generate_profile(ProfileRequest(max_chars=mc, tone="bright", me=me), None))
        assert [d.tone for d in res.drafts][0] == "bright" and len(res.drafts) == 3
        assert all(0 < d.char_count <= mc for d in res.drafts)
