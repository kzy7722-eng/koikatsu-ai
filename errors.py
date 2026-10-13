class APIError(Exception):
    """エラー封筒 {error:{code,message}} で返すための例外。"""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


class AIUnavailable(APIError):
    def __init__(self, message: str = "AI機能を利用できません。時間をおいて再度お試しください。"):
        super().__init__(503, "ai_unavailable", message)


class LLMError(Exception):
    """LLM呼び出し/出力検証の失敗(内部用。多くの場合ルール/テンプレートにフォールバックする)。"""
