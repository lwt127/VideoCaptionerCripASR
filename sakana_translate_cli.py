"""独立命令行工具：使用 Sakana Translate（https://chat.sakana.ai/translate）
翻译日文到中文（或 en/ja/zh/zh-Hant 之间的任意组合）。

这是一个基于分析网页请求得到的非官方接口封装，详见
``app/core/subtitle_processor/sakana_translate_client.py`` 中的说明。

用法
----
    # 直接翻译一段文本
    python sakana_translate_cli.py "こんにちは、世界。"

    # 指定源语言/目标语言（默认 ja -> zh）
    python sakana_translate_cli.py --from en --to ja "I like cats."

    # 从文件逐行翻译，输出到另一个文件
    python sakana_translate_cli.py --from ja --to zh --input in.txt --output out.txt

    # 输出礼貌体而非口语体
    python sakana_translate_cli.py --variant polite "猫が好きです。"

也可以在 Python 代码中直接使用：

    from app.core.subtitle_processor.sakana_translate_client import SakanaTranslateClient

    client = SakanaTranslateClient()
    zh_text = client.translate("猫が好きです", source_lang="ja", target_lang="zh")
"""

import argparse
import sys
from pathlib import Path

# 确保可以定位到 app 包（脚本放在仓库根目录，与 main.py 同级）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.subtitle_processor.sakana_translate_client import (  # noqa: E402
    SUPPORTED_LANGS,
    SakanaTranslateClient,
    SakanaTranslateError,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 Sakana Translate 翻译文本（日文<->中文<->英文）"
    )
    parser.add_argument("text", nargs="?", help="要翻译的文本（未提供 --input 时必填）")
    parser.add_argument(
        "--from",
        dest="source_lang",
        default="ja",
        choices=sorted(SUPPORTED_LANGS),
        help="源语言代码，默认 ja",
    )
    parser.add_argument(
        "--to",
        dest="target_lang",
        default="zh",
        choices=sorted(SUPPORTED_LANGS),
        help="目标语言代码，默认 zh",
    )
    parser.add_argument(
        "--variant",
        default="casual",
        choices=["casual", "polite"],
        help="译文风格：casual(口语) 或 polite(礼貌/正式)，默认 casual",
    )
    parser.add_argument("--input", help="输入文件路径（按行翻译，UTF-8编码）")
    parser.add_argument("--output", help="输出文件路径（配合 --input 使用）")
    args = parser.parse_args()

    if not args.text and not args.input:
        parser.error("请提供要翻译的文本，或使用 --input 指定输入文件")

    client = SakanaTranslateClient()

    if args.input:
        input_path = Path(args.input)
        lines = input_path.read_text(encoding="utf-8").splitlines()
        results = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                results.append("")
                continue
            try:
                translated = client.translate(
                    text=line,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    variant=args.variant,
                )
            except SakanaTranslateError as e:
                print(f"[第{i}行翻译失败] {e}", file=sys.stderr)
                translated = f"[ERROR] {line}"
            results.append(translated)
            print(f"{i}: {line} -> {translated}")

        output_text = "\n".join(results)
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"\n已保存到 {args.output}")
        return 0

    try:
        translated = client.translate(
            text=args.text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            variant=args.variant,
        )
    except SakanaTranslateError as e:
        print(f"翻译失败：{e}", file=sys.stderr)
        return 1

    print(translated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
