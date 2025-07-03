import re

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)        
    return text.strip()                     

input_path = "./arxiv100K/arxiv_titles_100k.txt"
output_path = "./arxiv100K/arxiv_titles_100k_clean.txt"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    count = 0
    for line in fin:
        cleaned = clean_text(line)
        fout.write(cleaned + "\n")
        count += 1

print(f"✅ 成功写入 {count} 条清洗后的标题到 {output_path}")
