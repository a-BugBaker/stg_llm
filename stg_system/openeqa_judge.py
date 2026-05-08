from __future__ import annotations

"""OpenEQA 问答结果裁判器。

第一优先级使用 LLM 裁判做语义一致性判断；
如果未启用 LLM，则回退到简单的归一化字符串匹配，便于离线调试。
"""

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import LLMConfig


@dataclass(slots=True)
class OpenEQAJudge:
    llm_config: LLMConfig

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        extra_answers: list[str] | None,
        pred_answer: str,
    ) -> dict[str, Any]:
        acceptable_answers = [gold_answer] + [item for item in (extra_answers or []) if str(item).strip()]
        if not self._available():
            return self._fallback_judge(acceptable_answers=acceptable_answers, pred_answer=pred_answer)

        prompt = (
            "你是一个 OpenEQA 问答裁判。\n"
            "请根据问题、标准答案、可接受备选答案、模型预测答案，判断预测是否语义正确。\n"
            "只输出严格 JSON，格式为：\n"
            "{\n"
            '  "score": 0 or 1,\n'
            '  "correct": true or false,\n'
            '  "reason": "简洁中文理由"\n'
            "}\n\n"
            f"question={json.dumps(question, ensure_ascii=False)}\n"
            f"gold_answer={json.dumps(gold_answer, ensure_ascii=False)}\n"
            f"extra_answers={json.dumps(acceptable_answers[1:], ensure_ascii=False)}\n"
            f"pred_answer={json.dumps(pred_answer, ensure_ascii=False)}\n"
            "判断规则：如果预测答案与标准答案或任一备选答案语义等价，则 score=1；否则 score=0。"
        )
        data = self._chat_json(prompt)
        if data is None:
            fallback = self._fallback_judge(acceptable_answers=acceptable_answers, pred_answer=pred_answer)
            fallback["reason"] = f"LLM 裁判失败，回退规则匹配。{fallback['reason']}"
            return fallback

        score = 1 if int(data.get("score", 0)) == 1 else 0
        correct = bool(data.get("correct", score == 1))
        return {
            "score": score,
            "correct": correct,
            "reason": str(data.get("reason", "")).strip(),
        }

    def _available(self) -> bool:
        return self.llm_config.enabled and bool(self.llm_config.api_key)

    def _chat_json(self, prompt: str) -> dict[str, Any] | None:
        url = self.llm_config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict OpenEQA evaluator. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.llm_config.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                content = "".join(parts)
            return json.loads(content)
        except Exception:
            return None

    def _fallback_judge(self, *, acceptable_answers: list[str], pred_answer: str) -> dict[str, Any]:
        pred_norm = _normalize_text(pred_answer)
        answer_norms = [_normalize_text(item) for item in acceptable_answers if _normalize_text(item)]
        correct = pred_norm in answer_norms if pred_norm else False
        return {
            "score": 1 if correct else 0,
            "correct": correct,
            "reason": "使用归一化字符串精确匹配作为回退裁判。",
        }


def _normalize_text(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
