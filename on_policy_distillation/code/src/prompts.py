"""Single source of truth for prompt format and answer extraction.

Both the trainer and the eval import `build_prompt` / `extract_answer` from here, so the
student is trained in exactly the format it is evaluated in (including the ``#### <number>``
marker the scorer looks for). If you change a prompt, you change it once.

TASK SPECS
----------
`TASK_SPECS` carries one entry per benchmark. Choose the one with real headroom for the
student: `gsm8k` for a 0.8B or 2B, `math500` for a 4B (GSM8K leaves it little to gain).
`gsm8k_native` is GSM8K in Qwen's zero-shot \boxed{} format.

THINKING MODE
-------------
`build_prompt(..., enable_thinking=)` is threaded all the way through because a reasoning
model's chat template injects a ``<think>`` block. If the teacher is conditioned with
thinking on and the student generates with it off (or vice versa), the two token streams
diverge *structurally* and the per-token reverse KL between them is meaningless. Pin it
identically on both sides; the default is off, which also keeps rollouts short enough to
score affordably.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Literal

AnswerStyle = Literal["marker", "boxed"]

# ---------------------------------------------------------------------------
# Answer-format instructions
# ---------------------------------------------------------------------------

MARKER_PREAMBLE = (
    "Solve the grade school math problem. Reason step by step, then give the final "
    "answer on its own line in the form '#### <number>'.\n\n"
)

BOXED_PREAMBLE = (
    "Solve the mathematics problem. Reason step by step, then put the final answer "
    "inside \\boxed{}.\n\n"
)

# ---------------------------------------------------------------------------
# Few-shot pools -- worked examples in the exact format we ask the model to produce
# ---------------------------------------------------------------------------

GSM8K_FEWSHOT: list[tuple[str, str]] = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April and May?",
        "In April Natalia sold 48 clips.\nIn May she sold 48 / 2 = 24 clips.\n"
        "Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
        "babysitting. How much did she earn?",
        "Per minute Weng earns 12 / 60 = $0.2.\nFor 50 minutes she earned "
        "50 * 0.2 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of "
        "the money she needs. Her parents decided to give her $15 for that purpose, and "
        "her grandparents twice as much as her parents. How much more money does Betty "
        "need to buy the wallet?",
        "Betty has 100 / 2 = $50.\nHer grandparents gave her 15 * 2 = $30.\n"
        "In total she has 50 + 15 + 30 = $95.\nShe still needs 100 - 95 = $5.\n#### 5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages "
        "does he write a year?",
        "Each time James writes 3 * 2 = 6 pages.\nTwice a week that is "
        "6 * 2 = 12 pages.\nOver a year that is 12 * 52 = 624 pages.\n#### 624",
    ),
    (
        "Mark has a garden with flowers. He planted plants of three different colors in "
        "it. Ten of them are yellow, and there are 80% more of those in purple. There are "
        "only 25% as many green flowers as there are yellow and purple flowers. How many "
        "flowers does Mark have in his garden?",
        "There are 10 yellow flowers.\nPurple: 10 + 80% of 10 = 10 + 8 = 18.\n"
        "Yellow and purple together: 10 + 18 = 28.\nGreen: 25% of 28 = 7.\n"
        "Total: 28 + 7 = 35.\n#### 35",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts "
        "in total does it take?",
        "Blue fiber: 2 bolts.\nWhite fiber: 2 / 2 = 1 bolt.\nTotal: 2 + 1 = 3 bolts.\n#### 3",
    ),
    (
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in "
        "$50,000 in repairs. This increased the value of the house by 150%. How much profit "
        "did he make?",
        "Josh spent 80000 + 50000 = $130,000.\nThe new value is "
        "80000 * (1 + 1.5) = $200,000.\nProfit is 200000 - 130000 = $70,000.\n#### 70000",
    ),
    (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She "
        "gives 15 cups in the morning and 25 cups in the afternoon. If Wendi has 20 "
        "chickens, how many cups does she need to give in the final meal of the day?",
        "20 chickens need 20 * 3 = 60 cups per day.\nShe has already given "
        "15 + 25 = 40 cups.\nThe final meal needs 60 - 40 = 20 cups.\n#### 20",
    ),
]

MATH_FEWSHOT: list[tuple[str, str]] = [
    (
        "What is the value of $\\sqrt{36+64}-\\sqrt{25-16}$?",
        "First, $36 + 64 = 100$, so $\\sqrt{100} = 10$.\n"
        "Next, $25 - 16 = 9$, so $\\sqrt{9} = 3$.\n"
        "Therefore the value is $10 - 3 = 7$.\n"
        "The final answer is $\\boxed{7}$.",
    ),
    (
        "If $2x - 5 = 11$, what is the value of $3x + 2$?",
        "From $2x - 5 = 11$ we get $2x = 16$, so $x = 8$.\n"
        "Then $3x + 2 = 3(8) + 2 = 24 + 2 = 26$.\n"
        "The final answer is $\\boxed{26}$.",
    ),
    (
        "A fair coin is flipped 3 times. What is the probability that all three flips "
        "are heads? Express your answer as a common fraction.",
        "Each flip is heads with probability $\\frac{1}{2}$, and the flips are independent.\n"
        "So the probability is $\\left(\\frac{1}{2}\\right)^3 = \\frac{1}{8}$.\n"
        "The final answer is $\\boxed{\\frac{1}{8}}$.",
    ),
    (
        "What is the degree measure of an interior angle of a regular pentagon?",
        "The interior angles of an $n$-gon sum to $180(n-2)$ degrees.\n"
        "For $n = 5$ that is $180 \\cdot 3 = 540$ degrees.\n"
        "A regular pentagon has 5 equal angles, so each is $540 / 5 = 108$ degrees.\n"
        "The final answer is $\\boxed{108}$.",
    ),
    (
        "Simplify $\\sqrt{50} + \\sqrt{18}$. Express your answer in simplest radical form.",
        "Factor each radical: $\\sqrt{50} = \\sqrt{25 \\cdot 2} = 5\\sqrt{2}$ and "
        "$\\sqrt{18} = \\sqrt{9 \\cdot 2} = 3\\sqrt{2}$.\n"
        "Adding, $5\\sqrt{2} + 3\\sqrt{2} = 8\\sqrt{2}$.\n"
        "The final answer is $\\boxed{8\\sqrt{2}}$.",
    ),
    (
        "For what values of $x$ is $x^2 - 5x + 6 < 0$? Express your answer in interval notation.",
        "Factor: $x^2 - 5x + 6 = (x-2)(x-3)$.\n"
        "The product is negative exactly when one factor is positive and the other negative, "
        "which happens for $2 < x < 3$.\n"
        "The final answer is $\\boxed{(2, 3)}$.",
    ),
    (
        "Find the ordered pair $(x, y)$ satisfying $2x + y = 7$ and $x - y = 2$.",
        "Add the two equations: $(2x + y) + (x - y) = 7 + 2$, so $3x = 9$ and $x = 3$.\n"
        "Substitute into the second equation: $3 - y = 2$, so $y = 1$.\n"
        "The final answer is $\\boxed{(3, 1)}$.",
    ),
    (
        "What is the remainder when $2^{10}$ is divided by $7$?",
        "Compute powers of $2$ modulo $7$: $2^1 \\equiv 2$, $2^2 \\equiv 4$, $2^3 \\equiv 1 \\pmod 7$.\n"
        "So $2^{10} = 2^{3 \\cdot 3 + 1} = (2^3)^3 \\cdot 2 \\equiv 1^3 \\cdot 2 \\equiv 2 \\pmod 7$.\n"
        "The final answer is $\\boxed{2}$.",
    ),
]


# ---------------------------------------------------------------------------
# Task specs
# ---------------------------------------------------------------------------


def _gsm8k_gold(record: dict[str, Any]) -> str:
    """GSM8K stores the gold after the '####' marker in the reference solution."""
    return record["answer"].split("####")[-1].strip()


def _math_gold(record: dict[str, Any]) -> str:
    """MATH-500 ships a pre-extracted `answer` column; fall back to the boxed solution."""
    answer = (record.get("answer") or "").strip()
    if answer:
        return answer
    return extract_boxed(record.get("solution", "")) or ""


@dataclass(frozen=True)
class TaskSpec:
    """Everything that distinguishes one benchmark from another, in one place."""

    key: str
    dataset: str
    dataset_name: str | None
    train_split: str
    test_split: str
    question_field: str
    preamble: str
    answer_style: AnswerStyle
    fewshot_pool: list[tuple[str, str]]
    gold_fn: Callable[[dict[str, Any]], str]
    # Sequences that mean "the model has started a new problem" -- used as stop strings so
    # a few-shot completion does not run on into a hallucinated next question.
    stop_seqs: tuple[str, ...] = ("\nProblem:", "\n\nProblem:", "\nQuestion:", "\n\nQuestion:")
    # Header words used for both the few-shot prefix and the live prompt. Empty strings drop
    # the header entirely -- what a zero-shot "native" prompt wants (see gsm8k_native).
    q_label: str = "Question"
    a_label: str = "Answer"
    # Appended AFTER the question: the Qwen model card puts the format instruction here rather
    # than in a preamble (see gsm8k_native).
    suffix: str = ""


TASK_SPECS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        key="gsm8k",
        dataset="openai/gsm8k",
        dataset_name="main",
        train_split="train",
        test_split="test",
        question_field="question",
        preamble=MARKER_PREAMBLE,
        answer_style="marker",
        fewshot_pool=GSM8K_FEWSHOT,
        gold_fn=_gsm8k_gold,
        q_label="Question",
        a_label="Answer",
    ),
    # The task with headroom for a 4B student (GSM8K saturates at that scale).
    # MATH-500 has no train split, so distillation prompts come from the full MATH train
    # set (`--train-dataset`), keeping the 500 test rows genuinely held out.
    "math500": TaskSpec(
        key="math500",
        dataset="HuggingFaceH4/MATH-500",
        dataset_name=None,
        train_split="test",  # unused for training; see TRAIN_DATASET in env_vars
        test_split="test",
        question_field="problem",
        preamble=BOXED_PREAMBLE,
        answer_style="boxed",
        fewshot_pool=MATH_FEWSHOT,
        gold_fn=_math_gold,
        q_label="Problem",
        a_label="Solution",
    ),
    # GSM8K in the format Qwen3.5 was post-trained for: zero-shot, bare question, the
    # format instruction as a suffix, answer in \boxed{}. MEASURED on 100 GSM8K rows
    # (greedy): 0.60 vs 0.51 for the 4-shot '####' spec above, with a ~6x shorter prompt
    # (76 vs 492 tokens). Same dataset and gold answers -- only the prompt differs.
    # Costs length: rollouts run ~530 vs ~360 tokens, so raise MAX_NEW_TOKENS with it.
    "gsm8k_native": TaskSpec(
        key="gsm8k_native",
        dataset="openai/gsm8k",
        dataset_name="main",
        train_split="train",
        test_split="test",
        question_field="question",
        preamble="",
        answer_style="boxed",
        fewshot_pool=[],      # zero-shot: the suffix carries the format instruction
        gold_fn=_gsm8k_gold,
        stop_seqs=(),         # no few-shot prefix to run on into
        q_label="",
        a_label="",
        suffix="\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
}


def get_task_spec(key: str) -> TaskSpec:
    if key not in TASK_SPECS:
        raise KeyError(f"unknown task {key!r}; known: {sorted(TASK_SPECS)}")
    return TASK_SPECS[key]


# ---------------------------------------------------------------------------
# Prompt construction -- used by BOTH train_distill.py AND eval_student.py
# ---------------------------------------------------------------------------


def build_fewshot_prefix(spec: TaskSpec, n: int) -> str:
    """`n` worked examples in the exact format we ask the model to produce."""
    shots = spec.fewshot_pool[: max(0, min(n, len(spec.fewshot_pool)))]
    return "".join(
        f"{spec.q_label}: {q}\n{spec.a_label}: {a}\n\n" for q, a in shots
    )


def build_completion_prompt(spec: TaskSpec, question: str, fewshot_prefix: str) -> str:
    """The raw completion-style prompt (no chat template)."""
    head = f"{spec.q_label}: " if spec.q_label else ""
    tail = f"\n{spec.a_label}:" if spec.a_label else ""
    return f"{spec.preamble}{fewshot_prefix}{head}{question.strip()}{spec.suffix}{tail}"


def build_user_message(spec: TaskSpec, question: str, fewshot_prefix: str) -> str:
    """The chat *user message content* -- what goes inside the chat template.

    Exposed separately because `eval_student.py` hands this to Inspect AI, which applies
    the tokenizer's chat template itself. Both paths therefore build the same text.
    """
    head = f"{spec.q_label}: " if spec.q_label else ""
    return f"{spec.preamble}{fewshot_prefix}{head}{question.strip()}{spec.suffix}"


def build_prompt(
    spec: TaskSpec,
    question: str,
    fewshot: int = 0,
    tokenizer: Any | None = None,
    chat: bool = True,
    enable_thinking: bool = False,
) -> str:
    """The one prompt builder. Identical text for training and eval.

    `chat=True` renders through the tokenizer's chat template (instruct models);
    `chat=False` renders the raw few-shot completion prompt (base models). We *never*
    silently fall back between the two: a missing chat template raises, because the
    silent fallback is exactly the bug that made the first run's gains vanish.
    """
    prefix = build_fewshot_prefix(spec, fewshot)
    if not chat:
        return build_completion_prompt(spec, question, prefix)

    if tokenizer is None:
        raise ValueError("chat=True requires a tokenizer")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"chat=True but {tokenizer.name_or_path!r} has no chat_template. "
            "Pass chat=False (raw completion prompt) for a -Base model -- do not let "
            "this fall back silently, or training and eval formats will diverge."
        )
    user = build_user_message(spec, question, prefix)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    # Qwen-style reasoning templates accept enable_thinking; older ones do not.
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)


# ---------------------------------------------------------------------------
# Number / LaTeX parsing
# ---------------------------------------------------------------------------

# A bare decimal, a fraction ("3/4", "1 1/2") or a percentage -- fraction first so the
# alternation does not stop at the numerator. Currency and thousands separators are
# tolerated here and stripped during parsing.
_DECIMAL = r"-?\$?\d[\d,]*(?:\.\d+)?"
ANSWER_TOKEN_RE = re.compile(rf"{_DECIMAL}\s*/\s*{_DECIMAL}|{_DECIMAL}\s*%?")
_FRACTION_RE = re.compile(
    r"^(?P<sign>-?)(?:(?P<whole>\d+(?:\.\d+)?)\s+)?"
    r"(?P<num>\d+(?:\.\d+)?)\s*/\s*(?P<den>\d+(?:\.\d+)?)$"
)

_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "half": 0.5, "quarter": 0.25,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_WORD_NUMBER_RE = re.compile(
    r"\b(?:(?:negative|minus)\s+)?"
    r"(?:" + "|".join(sorted(_WORD_UNITS, key=len, reverse=True)) + r"|"
    + "|".join(_WORD_SCALES) + r"|and)"
    r"(?:[\s-]+(?:" + "|".join(sorted(_WORD_UNITS, key=len, reverse=True)) + r"|"
    + "|".join(_WORD_SCALES) + r"|and))*\b",
    re.IGNORECASE,
)

DEFAULT_ANSWER_REL_TOL = 1e-3
_ANSWER_ABS_TOL = 1e-9
DEFAULT_TEXT_SIMILARITY = 0.9

# LaTeX decoration that carries no mathematical content for answer comparison.
_LATEX_STRIP = (
    r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", "$", "\\$",
    r"^\circ", r"^{\circ}", r"\%", "%", r"\cdot", r"\times",
)
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|mbox|textbf|mathbf)\s*\{([^{}]*)\}")
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")


def extract_boxed(text: str) -> str | None:
    """Contents of the LAST ``\\boxed{...}`` in `text`, brace-balanced.

    A regex cannot do this correctly -- ``\\boxed{\\frac{1}{2}}`` has nested braces -- so
    we scan for the marker and walk the braces.
    """
    if not text:
        return None
    best: str | None = None
    for marker in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            brace = text.find("{", idx)
            if brace == -1:
                start = idx + len(marker)
                continue
            depth, i = 0, brace
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        best = text[brace + 1 : i]
                        break
                i += 1
            start = idx + len(marker)
    return best.strip() if best is not None else None


def _normalise_latex(raw: str) -> str:
    """Reduce a LaTeX answer to a comparable core: strip decoration, flatten \\frac."""
    text = raw.strip()
    text = _LATEX_TEXT_RE.sub(r"\1", text)
    for token in _LATEX_STRIP:
        text = text.replace(token, "")
    # \frac{a}{b} -> a/b so the numeric parser can take it from here.
    for _ in range(3):  # a couple of passes handles \frac inside \frac
        new = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", text)
        if new == text:
            break
        text = new
    text = _LATEX_SQRT_RE.sub(r"sqrt(\1)", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(text.split())


def _parse_word_number(text: str) -> float | None:
    """'seventy-two' -> 72.0. Returns None on any token that is not a number word."""
    tokens = [t for t in re.split(r"[\s\-]+", text.strip().lower()) if t and t != "and"]
    if not tokens:
        return None
    sign = 1.0
    if tokens[0] in {"negative", "minus"}:
        sign, tokens = -1.0, tokens[1:]
    total = current = 0.0
    seen = False
    for token in tokens:
        if token in _WORD_UNITS:
            current += _WORD_UNITS[token]
            seen = True
        elif token in _WORD_SCALES:
            scale = _WORD_SCALES[token]
            if scale == 100:
                current = (current or 1.0) * 100
            else:
                total += (current or 1.0) * scale
                current = 0.0
            seen = True
        elif token in {"a", "an"}:
            continue
        else:
            return None
    return sign * (total + current) if seen else None


def _parse_number(raw: str | None) -> float | None:
    """Any answer spelling -> a float: '$1,000.00', '3/4', '50%', 'seventy-two', '\\frac12'."""
    if raw is None:
        return None
    text = _normalise_latex(raw) if "\\" in raw or "{" in raw else raw
    text = text.strip().lower().replace("$", "").replace(",", "").replace("_", "")
    text = text.rstrip("%").strip().rstrip(".").strip()
    if not text or text in {"-", "."}:
        return None
    # "(1)/(2)" from the \frac flattening, and plain "1/2".
    bare = text.replace("(", "").replace(")", "").strip()
    fraction = _FRACTION_RE.match(bare)
    if fraction:
        denominator = float(fraction["den"])
        if denominator == 0:
            return None
        value = float(fraction["whole"] or 0) + float(fraction["num"]) / denominator
        return -value if fraction["sign"] else value
    try:
        value = float(bare)
    except ValueError:
        return _parse_word_number(bare)
    return value if math.isfinite(value) else None


def clean_number(raw: str) -> str | None:
    """Normalise a matched answer to a bare canonical numeric string."""
    value = _parse_number(raw)
    if value is None:
        return None
    return str(int(value)) if value.is_integer() else repr(value)


def _normalise_text(raw: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- for the non-numeric fallback."""
    base = _normalise_latex(raw) if ("\\" in raw or "{" in raw) else raw
    return " ".join(re.sub(r"[^0-9a-z]+", " ", base.strip().lower()).split())


def answers_equivalent(
    answer: str | None,
    gold: str | None,
    rel_tol: float = DEFAULT_ANSWER_REL_TOL,
    text_similarity: float = DEFAULT_TEXT_SIMILARITY,
) -> tuple[bool, str]:
    """Is `answer` the same answer as `gold`? Returns (equivalent, how-it-matched).

    Semantic rather than literal: both sides are parsed to a value first, so differences
    of spelling ('1,000' / '$1000'), notation ('3/4' / '0.75' / '\\frac{3}{4}'),
    decoration ('50%', '$', '\\left') and wording ('seventy-two' / '72') do not count as
    wrong answers. Only when neither side parses as a number does it fall back to
    normalised-string comparison, which is what MATH's symbolic answers need.
    """
    if answer is None or gold is None:
        return False, "missing"
    a_value, g_value = _parse_number(answer), _parse_number(gold)
    if a_value is not None and g_value is not None:
        if a_value == g_value:
            return True, "exact"
        if math.isclose(a_value, g_value, rel_tol=rel_tol, abs_tol=_ANSWER_ABS_TOL):
            return True, "tolerance"
        return False, "numeric_mismatch"
    # Symbolic answers ("x^2+1", "(3,\\pi)"): compare the normalised LaTeX core.
    a_latex, g_latex = _normalise_latex(answer), _normalise_latex(gold)
    if a_latex and g_latex and a_latex.replace(" ", "") == g_latex.replace(" ", ""):
        return True, "exact"
    a_text, g_text = _normalise_text(answer), _normalise_text(gold)
    if not a_text or not g_text:
        return False, "missing"
    if a_text == g_text:
        return True, "exact"
    ratio = SequenceMatcher(None, a_text, g_text).ratio()
    return (True, "similarity") if ratio >= text_similarity else (False, "text_mismatch")


def extract_answer(text: str, style: AnswerStyle = "marker") -> str | None:
    """Final answer from a completion, honouring the task's answer style.

    `style="boxed"` prefers ``\\boxed{...}`` (MATH convention) and keeps symbolic answers
    as text; `style="marker"` prefers ``#### n`` (GSM8K convention). Both fall back to the
    last number in the first answer block, because a model that stops early still deserves
    credit for a correct final line.
    """
    if not text:
        return None

    if style == "boxed":
        boxed = extract_boxed(text)
        if boxed:
            return clean_number(boxed) or _normalise_latex(boxed) or None

    marked = re.findall(rf"####\s*({ANSWER_TOKEN_RE.pattern})", text)
    for candidate in reversed(marked):
        cleaned = clean_number(candidate)
        if cleaned is not None:
            return cleaned

    if style == "marker":
        boxed = extract_boxed(text)
        if boxed:
            cleaned = clean_number(boxed)
            if cleaned is not None:
                return cleaned

    # Models often stop before emitting the marker, or run on into the next few-shot
    # question -- fall back to the last number of the first answer block.
    head = re.split(r"\n\s*(?:Question|Problem|Q)\s*:", text)[0]
    for candidate in reversed(ANSWER_TOKEN_RE.findall(head)):
        cleaned = clean_number(candidate)
        if cleaned is not None:
            return cleaned
    # No digits anywhere: the model may have spelled the answer out in words.
    for candidate in reversed(_WORD_NUMBER_RE.findall(head)):
        cleaned = clean_number(candidate)
        if cleaned is not None:
            return cleaned
    return None


def has_answer_marker(text: str, style: AnswerStyle = "marker") -> bool:
    """Did the completion follow the answer-format instruction? (format-compliance metric)"""
    if not text:
        return False
    return ("\\boxed" in text) if style == "boxed" else ("####" in text)
