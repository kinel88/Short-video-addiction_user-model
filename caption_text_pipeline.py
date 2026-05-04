from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


CAPTION_ALL_COLUMNS = [
    "video_id",
    "manual_cover_text",
    "caption",
    "topic_tag",
    "first_level_category_id",
    "first_level_category_name",
    "second_level_category_id",
    "second_level_category_name",
    "third_level_category_id",
    "third_level_category_name",
]

CAPTION_TEXT_COLUMNS = [
    "manual_cover_text",
    "caption",
    "topic_tag",
    "first_level_category_name",
    "second_level_category_name",
    "third_level_category_name",
]

CATEGORY_NAME_COLUMNS = [
    "first_level_category_name",
    "second_level_category_name",
    "third_level_category_name",
]

INVALID_TEXT_VALUES = {
    "",
    "UNKNOWN",
    "UNK",
    "NULL",
    "NONE",
    "NAN",
    "NA",
    "N/A",
    "[]",
}

BUILTIN_SHORT_VIDEO_PHRASES = {
    "\u76f4\u64ad",
    "\u76f4\u64ad\u95f4",
    "\u4e0a\u70ed\u95e8",
    "\u53cc\u51fb",
    "\u5173\u6ce8\u6211",
    "\u641e\u7b11",
    "\u989c\u503c",
    "\u5973\u88c5",
    "\u7f8e\u98df",
    "\u666f\u7269\u6444\u5f71",
    "\u611f\u8c22\u5feb\u624b\u6211\u8981\u4e0a\u70ed\u95e8",
    "\u611f\u8c22\u63a8\u5e7f\u5c0f\u52a9\u624b",
    "\u5546\u5bb6\u53f7\u6218\u75ab\u884c\u52a8",
    "\u4f5c\u54c1\u63a8\u5e7f",
}

SEMANTIC_SINGLE_CHARS = {
    "\u7f8e",
    "\u7231",
    "\u7b11",
    "\u9177",
    "\u840c",
    "\u751c",
    "\u71c3",
    "\u6f6e",
    "\u8d5e",
    "\u5e05",
    "\u725b",
    "\u55e8",
    "\u6696",
    "\u9999",
    "\u8fa3",
}

HASHTAG_PATTERN = re.compile(r"#([^\s#]+)")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:+-][A-Za-z0-9]+)*")
TOKEN_BODY_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
MIXED_SPAN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._:+-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+"
)


def _strip_wrapping_quotes(text: str) -> str:
    pairs = (
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
    )
    out = text.strip()
    changed = True
    while changed and len(out) >= 2:
        changed = False
        for left, right in pairs:
            if out.startswith(left) and out.endswith(right):
                out = out[len(left) : len(out) - len(right)].strip()
                changed = True
                break
    return out


def _normalize_text_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    text = str(value).replace("\ufeff", "").strip()
    if not text:
        return ""

    text = _strip_wrapping_quotes(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    if text.upper() in INVALID_TEXT_VALUES:
        return ""
    return text


def _token_body(text: Any) -> str:
    normalized = _normalize_text_value(text)
    if not normalized:
        return ""
    return "".join(TOKEN_BODY_PATTERN.findall(normalized))


def _parse_fixed_width_row(line: str, column_slices: list[tuple[int, int | None]]) -> dict[str, str]:
    record: dict[str, str] = {}
    for column, (start, end) in zip(CAPTION_ALL_COLUMNS, column_slices):
        value = line[start:end] if end is not None else line[start:]
        record[column] = value.strip()
    return record


def _parse_fallback_row(line: str) -> dict[str, str] | None:
    parts = line.rstrip("\r\n").split(",")
    if len(parts) < len(CAPTION_ALL_COLUMNS):
        return None

    suffix = parts[-6:]
    prefix = parts[:-6]
    if len(prefix) < 4:
        return None

    values = [
        prefix[0],
        prefix[1],
        prefix[2],
        ",".join(prefix[3:]),
        *suffix,
    ]
    return {
        column: str(value).strip()
        for column, value in zip(CAPTION_ALL_COLUMNS, values)
    }


def load_kuairec_caption_category_rows(csv_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_data_lines = 0
    invalid_video_id_rows = 0
    fallback_rows = 0
    duplicate_video_rows = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
        header_line = handle.readline()
        if not header_line:
            empty = pd.DataFrame(columns=CAPTION_ALL_COLUMNS)
            return empty, {
                "caption_csv": csv_path,
                "physical_data_lines": 0,
                "parsed_rows_before_dedupe": 0,
                "invalid_video_id_rows": 0,
                "duplicate_video_rows": 0,
                "fallback_rows": 0,
                "valid_video_rows": 0,
            }

        comma_positions = [idx for idx, ch in enumerate(header_line) if ch == ","]
        if len(comma_positions) != len(CAPTION_ALL_COLUMNS) - 1:
            raise ValueError(
                f"Unexpected caption csv header layout: found {len(comma_positions)} commas."
            )

        starts = [0] + [pos + 1 for pos in comma_positions]
        ends: list[int | None] = comma_positions + [None]
        column_slices = list(zip(starts, ends))

        seen_video_ids: set[int] = set()
        for line in handle:
            if not line.strip():
                continue
            total_data_lines += 1

            record = _parse_fixed_width_row(line, column_slices)
            video_id_value = pd.to_numeric(record["video_id"], errors="coerce")
            if pd.isna(video_id_value):
                fallback_record = _parse_fallback_row(line)
                if fallback_record is not None:
                    fallback_rows += 1
                    record = fallback_record
                    video_id_value = pd.to_numeric(record["video_id"], errors="coerce")

            if pd.isna(video_id_value):
                invalid_video_id_rows += 1
                continue

            video_id = int(video_id_value)
            if video_id in seen_video_ids:
                duplicate_video_rows += 1
                continue

            seen_video_ids.add(video_id)
            parsed = {column: record.get(column, "").strip() for column in CAPTION_ALL_COLUMNS}
            parsed["video_id"] = video_id
            for level_column in (
                "first_level_category_id",
                "second_level_category_id",
                "third_level_category_id",
            ):
                parsed[level_column] = pd.to_numeric(
                    parsed.get(level_column, ""), errors="coerce"
                )
            records.append(parsed)

    caption_df = pd.DataFrame(records, columns=CAPTION_ALL_COLUMNS)
    if not caption_df.empty:
        caption_df["video_id"] = caption_df["video_id"].astype(np.int64)
        for field in CAPTION_TEXT_COLUMNS:
            caption_df[field] = caption_df[field].fillna("").astype(str).str.strip()

    parse_meta = {
        "caption_csv": csv_path,
        "physical_data_lines": int(total_data_lines),
        "parsed_rows_before_dedupe": int(total_data_lines - invalid_video_id_rows),
        "invalid_video_id_rows": int(invalid_video_id_rows),
        "duplicate_video_rows": int(duplicate_video_rows),
        "fallback_rows": int(fallback_rows),
        "valid_video_rows": int(len(caption_df)),
    }
    return caption_df, parse_meta


def _parse_topic_tags(topic_tag_text: Any) -> list[str]:
    normalized = _normalize_text_value(topic_tag_text)
    if not normalized:
        return []

    inner = normalized
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()

    tags: list[str] = []
    seen: set[str] = set()
    for piece in inner.split(","):
        body = _token_body(piece)
        if body and body not in seen:
            tags.append(body)
            seen.add(body)
    return tags


def _extract_hashtag_tokens(text: str) -> tuple[list[str], str]:
    tokens: list[str] = []
    for tag_text in HASHTAG_PATTERN.findall(text):
        body = _token_body(tag_text)
        if body:
            tokens.append(body)
            tokens.append(f"hash_{body}")
    cleaned = HASHTAG_PATTERN.sub(" ", text)
    return tokens, cleaned


def _build_phrase_lexicon(caption_df: pd.DataFrame) -> dict[str, list[str]]:
    phrases: set[str] = set()
    for field in CATEGORY_NAME_COLUMNS:
        for value in caption_df[field].tolist():
            body = _token_body(value)
            if len(body) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", body):
                phrases.add(body)

    for value in caption_df["topic_tag"].tolist():
        for tag in _parse_topic_tags(value):
            if len(tag) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", tag):
                phrases.add(tag)

    for field in ("manual_cover_text", "caption"):
        for value in caption_df[field].tolist():
            normalized = _normalize_text_value(value)
            if not normalized:
                continue
            for tag_text in HASHTAG_PATTERN.findall(normalized):
                body = _token_body(tag_text)
                if len(body) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", body):
                    phrases.add(body)

    for phrase in BUILTIN_SHORT_VIDEO_PHRASES:
        body = _token_body(phrase)
        if len(body) >= 2 and re.fullmatch(r"[\u4e00-\u9fff]+", body):
            phrases.add(body)

    lexicon: dict[str, list[str]] = defaultdict(list)
    for phrase in phrases:
        lexicon[phrase[0]].append(phrase)

    for first_char in lexicon:
        lexicon[first_char].sort(key=lambda phrase: (-len(phrase), phrase))
    return dict(lexicon)


def _segment_chinese_span(span: str, phrase_lexicon: dict[str, list[str]]) -> list[str]:
    tokens: list[str] = []
    idx = 0
    while idx < len(span):
        matched = None
        for phrase in phrase_lexicon.get(span[idx], []):
            if span.startswith(phrase, idx):
                matched = phrase
                break

        if matched is not None:
            tokens.append(matched)
            idx += len(matched)
            continue

        if idx + 1 < len(span):
            tokens.append(span[idx : idx + 2])
            idx += 1
            continue

        if span[idx] in SEMANTIC_SINGLE_CHARS:
            tokens.append(span[idx])
        idx += 1

    return tokens


def _segment_body_text(text: Any, phrase_lexicon: dict[str, list[str]]) -> list[str]:
    normalized = _normalize_text_value(text)
    if not normalized:
        return []

    hashtag_tokens, cleaned_text = _extract_hashtag_tokens(normalized)
    tokens = list(hashtag_tokens)
    for match in MIXED_SPAN_PATTERN.finditer(cleaned_text):
        span = match.group(0)
        if ALNUM_PATTERN.fullmatch(span):
            body = _token_body(span)
            if body:
                tokens.append(body)
            continue
        tokens.extend(_segment_chinese_span(span, phrase_lexicon))
    return tokens


def _category_tokens(row: pd.Series) -> list[str]:
    tokens: list[str] = []
    for level_idx, field in enumerate(CATEGORY_NAME_COLUMNS, start=1):
        body = _token_body(row.get(field, ""))
        if body:
            tokens.append(body)
            tokens.append(f"cat{level_idx}_{body}")
    return tokens


def _topic_tag_tokens(topic_tag_text: Any) -> list[str]:
    tokens: list[str] = []
    for tag in _parse_topic_tags(topic_tag_text):
        tokens.append(tag)
        tokens.append(f"tag_{tag}")
    return tokens


def segment_caption_rows(caption_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if caption_df.empty:
        empty = pd.DataFrame(
            columns=[
                "video_id",
                *CAPTION_TEXT_COLUMNS,
                "segmented_manual_cover_text",
                "segmented_caption",
                "segmented_topic_tag",
                "segmented_category_tokens",
                "segmented_text_all",
            ]
        )
        return empty, {
            "phrase_lexicon_size": 0,
            "field_weights": {"body": 1, "topic_tag": 3, "category": 2},
        }

    phrase_lexicon = _build_phrase_lexicon(caption_df)
    rows: list[dict[str, Any]] = []
    for _, row in caption_df.iterrows():
        manual_tokens = _segment_body_text(row["manual_cover_text"], phrase_lexicon)
        caption_tokens = _segment_body_text(row["caption"], phrase_lexicon)
        topic_tokens = _topic_tag_tokens(row["topic_tag"])
        category_tokens = _category_tokens(row)

        weighted_tokens = (
            manual_tokens
            + caption_tokens
            + (topic_tokens * 3)
            + (category_tokens * 2)
        )

        rows.append(
            {
                "video_id": int(row["video_id"]),
                "manual_cover_text": row["manual_cover_text"],
                "caption": row["caption"],
                "topic_tag": row["topic_tag"],
                "first_level_category_name": row["first_level_category_name"],
                "second_level_category_name": row["second_level_category_name"],
                "third_level_category_name": row["third_level_category_name"],
                "segmented_manual_cover_text": " ".join(manual_tokens),
                "segmented_caption": " ".join(caption_tokens),
                "segmented_topic_tag": " ".join(topic_tokens),
                "segmented_category_tokens": " ".join(category_tokens),
                "segmented_text_all": " ".join(weighted_tokens),
            }
        )

    segmented_df = pd.DataFrame(rows)
    segment_meta = {
        "phrase_lexicon_size": int(sum(len(v) for v in phrase_lexicon.values())),
        "field_weights": {"body": 1, "topic_tag": 3, "category": 2},
        "segmentation": {
            "hashtags": "preserve raw hashtag body and add hash_ prefixed token",
            "topic_tags": "split by ASCII comma, preserve whole tag and add tag_ prefixed token",
            "category_names": "preserve whole phrase and add cat{level}_ prefixed token",
            "body_text": "longest phrase match over Chinese spans, fallback to 2-char sliding tokens",
        },
    }
    return segmented_df, segment_meta


def build_caption_text_feature_bundle(
    caption_path: str,
    text_dim: int,
    max_features: int = 2000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    caption_df, parse_meta = load_kuairec_caption_category_rows(caption_path)
    segmented_df, segment_meta = segment_caption_rows(caption_df)

    if segmented_df.empty:
        text_meta = {
            **parse_meta,
            **segment_meta,
            "text_feature_dim": 0,
            "tfidf": {
                "tokenizer": "str.split",
                "preprocessor": None,
                "token_pattern": None,
                "lowercase": False,
                "max_features": int(max_features),
                "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
            },
        }
        return None, segmented_df, text_meta

    text_all = segmented_df["segmented_text_all"].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
    )
    x_text = tfidf.fit_transform(text_all)
    if x_text.shape[1] > 1:
        n_components = min(int(text_dim), x_text.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        text_emb = svd.fit_transform(x_text).astype(np.float32)
    else:
        text_emb = x_text.toarray().astype(np.float32)

    text_feature = pd.concat(
        [
            segmented_df[["video_id"]].reset_index(drop=True),
            pd.DataFrame(
                text_emb,
                columns=[f"text_svd_{idx}" for idx in range(text_emb.shape[1])],
            ),
        ],
        axis=1,
    )
    text_meta = {
        **parse_meta,
        **segment_meta,
        "text_feature_dim": int(text_emb.shape[1]),
        "tfidf_vocabulary_size": int(x_text.shape[1]),
        "tfidf": {
            "tokenizer": "str.split",
            "preprocessor": None,
            "token_pattern": None,
            "lowercase": False,
            "max_features": int(max_features),
            "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
        },
    }
    return text_feature, segmented_df, text_meta
