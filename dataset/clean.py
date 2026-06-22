import re
import argparse


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input strings file")
    parser.add_argument("output_path", help="Output cleaned strings file")
    args = parser.parse_args()

    count = 0

    with open(args.input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            cleaned = clean_text(line)
            fout.write(cleaned + "\n")
            count += 1

    print(f"[INFO] cleaned {count} lines")
    print(f"[INFO] input  = {args.input_path}")
    print(f"[INFO] output = {args.output_path}")


if __name__ == "__main__":
    main()