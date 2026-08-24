"""从 misaki 生成中文 G2P 的黄金对照数据。

Rust 侧要复刻的是**这个**输出。有了它,移植的正确性是可以断言的,而不是靠耳朵。

用法(需要 `pip install "misaki[zh] @ git+https://github.com/hexgrad/misaki.git"`):

    python make_golden.py zh_corpus.txt zh_golden.json

产出的每条记录带三层,粒度从粗到细,方便定位差异出在哪一层:

    text      原文
    phonemes  最终音素串(注音 + 数字声调)  <- 模型真正吃的东西
    pinyin    分词后每个词的 TONE3 拼音    <- 差异多半出在这一层
    words     jieba 的分词与词性
"""
import json
import sys

from misaki import zh
import jieba.posseg as psg
from pypinyin import lazy_pinyin, Style


def main(corpus_path: str, out_path: str) -> None:
    g2p = zh.ZHG2P(version="1.1")
    # 预热:jieba 第一次要建词典,不预热的话第一条记录会把它算进去。
    g2p("预热。")

    records = []
    for raw in open(corpus_path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out = g2p(line)
        phonemes = out[0] if isinstance(out, tuple) else out
        words = [(w.word, w.flag) for w in psg.cut(line)]
        pinyin = [
            lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
            for w, _ in words
        ]
        records.append(
            {
                "text": line,
                "phonemes": phonemes,
                "words": words,
                "pinyin": pinyin,
            }
        )

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"{len(records)} 条 -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "zh_corpus.txt",
         sys.argv[2] if len(sys.argv) > 2 else "zh_golden.json")
