import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd


INVALID_TOKENS = {
    "", "unknown", "unk", "null", "none", "nan", "na", "n/a", "missing"
}


@dataclass(frozen=True)
class SemanticNode:
    video_id: int
    level1: str
    level2: str
    level3: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "level1": self.level1,
            "level2": self.level2,
            "level3": self.level3,
        }



def _clean_text(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in INVALID_TOKENS:
        return None
    return s



def load_kuairec_caption_category(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=",",
        encoding="utf-8",
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
    )

    # 去掉列名首尾空格
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "video_id",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in semantic csv: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # 先把 video_id 强制转成数字，转不了的变成 NaN
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce")

    # 丢掉 video_id 无法解析的坏行
    df = df.dropna(subset=["video_id"]).copy()

    # 再转成 int
    df["video_id"] = df["video_id"].astype(int)

    return df



def build_semantic_library(
    caption_category_csv: str,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
) -> pd.DataFrame:
    """
    为每个视频构建 3 层语义库：
    video_id -> (level1, level2, level3)

    规则：
    1. 优先使用三级类目字段。
    2. 若某层缺失，则保留上一层，缺失层记为 UNKNOWN。
    3. 若三级类目全缺失，但 topic_tag 有值，则把 topic_tag 放到 level3。
       - level1 / level2 仍保持 UNKNOWN，因为 topic_tag 常常更像细粒度补充标签。
    4. 若什么都没有，则整条记为 UNKNOWN。
    """
    df = load_kuairec_caption_category(caption_category_csv).copy()

    for c in [
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]:
        df[c] = df[c].map(_clean_text)

    rows = []
    for r in df.itertuples(index=False):
        video_id = int(r.video_id)
        l1 = r.first_level_category_name
        l2 = r.second_level_category_name
        l3 = r.third_level_category_name
        topic_tag = r.topic_tag

        if l1 is None and l2 is None and l3 is None:
            if topic_tag is not None:
                node = SemanticNode(video_id, "UNKNOWN", "UNKNOWN", topic_tag)
            else:
                node = SemanticNode(video_id, "UNKNOWN", "UNKNOWN", "UNKNOWN")
        else:
            node = SemanticNode(
                video_id,
                l1 if l1 is not None else "UNKNOWN",
                l2 if l2 is not None else "UNKNOWN",
                l3 if l3 is not None else (topic_tag if topic_tag is not None else "UNKNOWN"),
            )

        rows.append(node.as_dict())

    lib = pd.DataFrame(rows).drop_duplicates(subset=["video_id"]).sort_values("video_id")
    lib["semantic_path"] = lib[["level1", "level2", "level3"]].agg(" > ".join, axis=1)

    if output_csv:
        lib.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if output_json:
        records = lib.to_dict(orient="records")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return lib


class HierarchicalSemanticMatcher:
    """
    基于“第一次出现不同的层级”来计算 novelty。

    默认权重：
    - level1 首次不同: 1.0
    - level2 首次不同: 0.6
    - level3 首次不同: 0.3
    - 全相同: 0.0

    关键处理：
    如果某一层有 UNKNOWN，则这一层不强行判成“不同”，
    避免脏数据把 novelty 虚高。
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.6, l3_weight: float = 0.3):
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.l3_weight = float(l3_weight)

    @staticmethod
    def _unknown(x: Optional[str]) -> bool:
        return x is None or str(x).strip().upper() == "UNKNOWN"

    def compare(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        a1, a2, a3 = a.get("level1", "UNKNOWN"), a.get("level2", "UNKNOWN"), a.get("level3", "UNKNOWN")
        b1, b2, b3 = b.get("level1", "UNKNOWN"), b.get("level2", "UNKNOWN"), b.get("level3", "UNKNOWN")

        if (not self._unknown(a1)) and (not self._unknown(b1)) and a1 != b1:
            return self.l1_weight
        if (not self._unknown(a2)) and (not self._unknown(b2)) and a2 != b2:
            return self.l2_weight
        if (not self._unknown(a3)) and (not self._unknown(b3)) and a3 != b3:
            return self.l3_weight
        return 0.0


class SemanticLibrary:
    def __init__(self, semantic_df: pd.DataFrame):
        required = {"video_id", "level1", "level2", "level3"}
        if not required.issubset(set(semantic_df.columns)):
            raise ValueError(f"Semantic library must contain columns: {sorted(required)}")
        self.df = semantic_df.copy()
        self.lookup = {
            int(r.video_id): {
                "level1": str(r.level1),
                "level2": str(r.level2),
                "level3": str(r.level3),
                "semantic_path": str(getattr(r, "semantic_path", f"{r.level1} > {r.level2} > {r.level3}")),
            }
            for r in self.df.itertuples(index=False)
        }

    @classmethod
    def from_csv(cls, csv_path: str) -> "SemanticLibrary":
        df = pd.read_csv(csv_path)
        return cls(df)

    def get(self, video_id: int) -> Dict[str, str]:
        return self.lookup.get(int(video_id), {
            "level1": "UNKNOWN",
            "level2": "UNKNOWN",
            "level3": "UNKNOWN",
            "semantic_path": "UNKNOWN > UNKNOWN > UNKNOWN",
        })


class SemanticAwareFeatureProvider:
    """
    给你现有 NpyFeatureProvider 的一层包装：
    继续返回 quality / position / topic，
    但把真正的层级语义也一起返回，供 novelty 比较使用。
    """

    def __init__(self, base_feature_provider: Any, semantic_library: SemanticLibrary):
        self.base = base_feature_provider
        self.semantic_library = semantic_library

    def get_user_features(self, user_id: int):
        return self.base.get_user_features(user_id)

    def get_video_features(self, video_id: int):
        base_feat = self.base.get_video_features(video_id)
        semantic = self.semantic_library.get(int(video_id))
        merged = dict(base_feat)
        merged.update(semantic)
        return merged


# ------------------------
# 你现有 RecSim 环境里可直接复用的 novelty 函数
# ------------------------

def compute_semantic_novelty_from_features(
    prev_semantic: Dict[str, Any],
    current_video_feat: Dict[str, Any],
    l1_weight: float = 1.0,
    l2_weight: float = 0.6,
    l3_weight: float = 0.3,
) -> float:
    matcher = HierarchicalSemanticMatcher(l1_weight=l1_weight, l2_weight=l2_weight, l3_weight=l3_weight)
    return matcher.compare(prev_semantic, current_video_feat)


# ------------------------
# 使用示例
# ------------------------
if __name__ == "__main__":
    # 1. 构建语义库
    lib = build_semantic_library(
        caption_category_csv="kuairec_caption_category.csv",
        output_csv="video_semantic_library.csv",
        output_json="video_semantic_library.json",
    )
    print("Semantic library built:", lib.head())

    # 2. 读取并比较 novelty
    semantic_library = SemanticLibrary(lib)
    matcher = HierarchicalSemanticMatcher()

    v1 = semantic_library.get(1)
    v2 = semantic_library.get(2)
    novelty = matcher.compare(v1, v2)
    print("video 1 semantic:", v1)
    print("video 2 semantic:", v2)
    print("novelty:", novelty)
