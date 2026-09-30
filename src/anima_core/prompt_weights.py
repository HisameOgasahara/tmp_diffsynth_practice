"""ComfyUI Anima 방식의 괄호 문법과 토큰별 가중치 대응."""

import torch


def parse_prompt_weights(text, default_multiplier=1.1):
    """(text)는 곱셈, (text:number)는 현재 구간의 절대 가중치입니다.

    이스케이프한 괄호와 닫히지 않은 괄호는 문자로 유지합니다.
    대괄호는 일반 문자입니다.
    """
    units = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in "()":
            units.append((text[index + 1], True))
            index += 2
        else:
            units.append((text[index], False))
            index += 1
    pairs, stack = {}, []
    for index, (char, escaped) in enumerate(units):
        if not escaped and char == "(":
            stack.append(index)
        elif not escaped and char == ")" and stack:
            pairs[stack.pop()] = index

    def collect(start, end, weight):
        result, pending = [], []
        index = start
        while index < end:
            if index in pairs and pairs[index] < end:
                if pending:
                    result.append(("".join(pending), weight))
                    pending = []
                closing = pairs[index]
                inner_end = closing
                inner_weight = weight * default_multiplier
                colons = [i for i in range(index + 1, closing) if units[i][0] == ":"]
                if colons and colons[-1] > index + 1:
                    colon = colons[-1]
                    try:
                        inner_weight = float("".join(c for c, _ in units[colon + 1:closing]))
                        inner_end = colon
                    except ValueError:
                        pass
                result.extend(collect(index + 1, inner_end, inner_weight))
                index = closing + 1
            else:
                pending.append(units[index][0])
                index += 1
        if pending:
            result.append(("".join(pending), weight))
        return result

    return collect(0, len(units), 1.0)


def _tokenize_segments(tokenizer, segments, max_length):
    ids, weights = [], []
    for text, weight in segments:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(tokens)
        weights.extend([weight] * len(tokens))
    # Anima의 Qwen은 추가 토큰이 없고 T5는 EOS를 뒤에 붙입니다.
    # 공개 호출 API로 suffix를 얻어 tokenizer 버전별 내부 메서드 차이를 피합니다.
    suffix = tokenizer("", add_special_tokens=True)["input_ids"]
    capacity = max_length - len(suffix)
    ids, weights = ids[:capacity], weights[:capacity]
    ids.extend(suffix)
    weights.extend([1.0] * len(suffix))
    return ids, weights


def tokenize_weighted_prompt(qwen_tokenizer, t5_tokenizer, prompt, max_length=512):
    segments = parse_prompt_weights(prompt)
    qwen_ids, _ = _tokenize_segments(qwen_tokenizer, segments, max_length)
    t5_ids, t5_weights = _tokenize_segments(t5_tokenizer, segments, max_length)
    qwen_inputs = qwen_tokenizer.pad(
        {"input_ids": [qwen_ids], "attention_mask": [[1] * len(qwen_ids)]},
        padding="max_length", max_length=max_length, return_tensors="pt",
    )
    return qwen_inputs, torch.tensor([t5_ids], dtype=torch.long), torch.tensor(
        t5_weights, dtype=torch.float32
    ).reshape(1, -1, 1)
