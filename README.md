# 恋活AI API (FastAPI)

プロフィール生成 / 返信生成 / 脈あり度分析の3エンドポイント。
LLMなし(`LLM_BACKEND=mock`)でも、ルールとテンプレートで動作します。

## 起動

```bash
pip install -r requirements.txt
cp .env.example .env        # 必要なら編集(環境変数として読み込んでください)
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/docs
pytest
```

Claudeを使う場合: `LLM_BACKEND=anthropic` と `ANTHROPIC_API_KEY` を設定(モデルは `ANTHROPIC_MODEL`、既定 `claude-sonnet-5-5`)。
`APP_API_TOKEN` を設定すると `Authorization: Bearer <token>` が必須になります(未設定=認証なし)。

## エンドポイント

| メソッド | パス | 内容 |
|---|---|---|
| POST | `/api/v1/generate/profile` | プロフィール3案(誠実系/明るい系/知的系)。文字数上限を必ず守る |
| POST | `/api/v1/generate/reply` | 初回メッセージ/返信の下書き。距離感・敬語・温度感・絵文字・返信頻度を会話から判定 |
| POST | `/api/v1/analyze/interest` | 脈あり度 0–100(5指標+根拠+次の一手) |
| GET | `/healthz` | 死活確認 |

```bash
curl -X POST localhost:8000/api/v1/analyze/interest -H 'content-type: application/json' -d '{
  "messages": [
    {"from":"me","text":"休日はカフェ巡りをしています","at":"2026-10-10T10:00:00+09:00"},
    {"from":"partner","text":"いいですね!私もカフェ好きです😊 最近はどこに行きましたか?","at":"2026-10-10T10:10:00+09:00"}
  ]}'
```

`from` は `partner`(相手)/ `me`(自分)。`at`(送信時刻)は任意で、あると返信速度を見ます。
成功時は `{id, type, result}`、失敗時は `{error:{code,message}}`(`invalid_input` 400 / `unauthorized` 401 / `ai_unavailable` 503)。

## 設計のポイント

- 返信: `app/reply.py` が D(距離) K(敬語) T(温度) E(絵文字) R(頻度) を決め、LLMの下書きを `check_draft` で検査(絵文字数・タメ口・質問禁止・NGワード・連絡先)。違反は1回だけ再生成し、だめならテンプレートに切り替えます。
- 脈あり度: 返信速度/質問/絵文字/文章量/関心度を計算し、発言数が少ないほど50に寄せます。詐欺の兆候は点数を出さず注意を返します。「断りが明確→20以下」「具体的な日時→70以上」等の補正入り。
- LLM失敗時は落とさず、ルール/テンプレートで返し `notes` に理由を入れます。LLMへ渡す前に連絡先・URLは伏せ字にします。
- 下書きのみ(自動送信なし)、相手の気持ちの断定なし、駆け引きの助言なし。

## 未実装(次にやること)

DB保存(履歴・利用回数制限 `limit_exceeded`)、`Idempotency-Key`、JWT認証(`require_auth` を差し替え)、スクリーンショット入力、`content_blocked` の本番向け判定、レート制限。
