import re

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)        
    return text.strip()                     

input_path = "./sift/sift_titles.txt"
output_path = "./sift/sift_titles_clean.txt"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    count = 0
    for line in fin:
        cleaned = clean_text(line)
        fout.write(cleaned + "\n")
        count += 1
