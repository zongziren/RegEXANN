import re
from lxml import etree

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_title(text):
    if not text:
        return False
    text = text.strip()
    if len(text) < 5:
        return False
    if " " not in text:  # 至少两个词
        return False
    blacklist = {"foreword", "preface", "vorwort", "editorial", "cover", "programme", "index", "author index"}
    lowered = text.lower().strip(".:")
    if lowered in blacklist:
        return False
    return True

def extract_top_k_titles(xml_path, k=100, output_path="sift100_titles.txt"):
    count = 0
    with open(output_path, "w", encoding="utf-8") as out_file:
        context = etree.iterparse(xml_path, events=("end",), tag=("article", "inproceedings"))
        for event, elem in context:
            title_elem = elem.find("title")
            if title_elem is not None and is_valid_title(title_elem.text):
                cleaned = clean_text(title_elem.text)
                if len(cleaned) < 5:
                    continue
                out_file.write(cleaned + "\n")
                count += 1
            elem.clear()
            if count >= k:
                break
    print(f"✅ 成功写入 {count} 条清洗后的标题到 {output_path}")


extract_top_k_titles("dblp.xml", k=100, output_path="sift100_titles.txt")