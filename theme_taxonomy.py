from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ThemeDefinition:
    key: str
    label: str
    aliases: tuple[str, ...]
    description: str


THEME_DEFINITIONS: tuple[ThemeDefinition, ...] = (
    ThemeDefinition(
        key="comedy",
        label="搞笑",
        aliases=("搞笑", "喜剧", "欢乐", "沙雕"),
        description="Comedy-focused works.",
    ),
    ThemeDefinition(
        key="romance",
        label="恋爱",
        aliases=("恋爱", "爱情", "纯爱", "恋爱喜剧"),
        description="Romance-oriented works.",
    ),
    ThemeDefinition(
        key="school",
        label="校园",
        aliases=("校园", "学园", "学校", "校园生活"),
        description="School and campus life themes.",
    ),
    ThemeDefinition(
        key="slice_of_life",
        label="日常",
        aliases=("日常", "日常系", "生活系"),
        description="Slice-of-life stories.",
    ),
    ThemeDefinition(
        key="healing",
        label="治愈",
        aliases=("治愈", "温馨", "疗愈"),
        description="Healing and warm narratives.",
    ),
    ThemeDefinition(
        key="science_fiction",
        label="科幻",
        aliases=("科幻", "sci-fi", "sf", "机甲", "未来"),
        description="Science fiction and mecha.",
    ),
    ThemeDefinition(
        key="fantasy_isekai",
        label="奇幻异世界",
        aliases=("奇幻", "异世界", "魔法", "转生", "勇者"),
        description="Fantasy and isekai themes.",
    ),
    ThemeDefinition(
        key="action",
        label="动作战斗",
        aliases=("动作", "战斗", "热血", "武斗"),
        description="Action and battle-heavy works.",
    ),
    ThemeDefinition(
        key="mystery",
        label="悬疑推理",
        aliases=("悬疑", "推理", "惊悚", "犯罪", "心理"),
        description="Mystery, suspense, and investigation.",
    ),
    ThemeDefinition(
        key="sports",
        label="运动竞技",
        aliases=("运动", "竞技", "体育"),
        description="Sports and competition.",
    ),
    ThemeDefinition(
        key="music",
        label="音乐偶像",
        aliases=("音乐", "偶像", "乐队", "歌舞"),
        description="Music, idol, and band-oriented works.",
    ),
    ThemeDefinition(
        key="original",
        label="原创",
        aliases=("原创",),
        description="Original works rather than adaptation tags.",
    ),
    ThemeDefinition(
        key="game_adaptation",
        label="游戏改",
        aliases=("游戏改", "gal改", "galgame", "手游改", "页游改", "游戏原作"),
        description="Game adaptations.",
    ),
    ThemeDefinition(
        key="harem",
        label="后宫",
        aliases=("后宫", "多女主"),
        description="Harem-style narratives.",
    ),
    ThemeDefinition(
        key="yuri",
        label="百合",
        aliases=("百合", "girlslove", "girls'love", "gl"),
        description="Yuri / girls' love works.",
    ),
    ThemeDefinition(
        key="bl",
        label="BL",
        aliases=("bl", "耽美", "boyslove", "boys'love"),
        description="BL / boys' love works.",
    ),
    ThemeDefinition(
        key="children",
        label="子供向",
        aliases=("子供向", "儿童", "亲子", "低龄"),
        description="Children-oriented works.",
    ),
)


def normalize_tag(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return "".join(normalized.split())


_NORMALIZED_ALIASES: dict[str, tuple[str, ...]] = {
    theme.label: tuple(normalize_tag(alias) for alias in theme.aliases)
    for theme in THEME_DEFINITIONS
}


def _matches_alias(tag: str, alias: str) -> bool:
    if not alias:
        return False
    if len(alias) <= 3:
        return tag == alias
    return tag == alias or alias in tag


def assign_themes(tags: Iterable[str]) -> list[str]:
    normalized_tags = []
    for tag in tags:
        if not tag:
            continue
        normalized = normalize_tag(tag)
        if normalized:
            normalized_tags.append(normalized)

    matched: list[str] = []
    for theme in THEME_DEFINITIONS:
        aliases = _NORMALIZED_ALIASES[theme.label]
        if any(_matches_alias(tag, alias) for tag in normalized_tags for alias in aliases):
            matched.append(theme.label)
    return matched


def taxonomy_records() -> list[dict[str, str]]:
    records = []
    for theme in THEME_DEFINITIONS:
        records.append(
            {
                "theme_key": theme.key,
                "theme_label": theme.label,
                "aliases": ", ".join(theme.aliases),
                "description": theme.description,
            }
        )
    return records
