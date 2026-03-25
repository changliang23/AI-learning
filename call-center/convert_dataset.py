import json

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
for item in data:
    new_data.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["output"]
    })

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ 已生成 train.json")
