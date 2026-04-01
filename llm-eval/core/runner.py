import json
from models.openai_model import call_model
from evaluators.rule_eval import exact_match
from evaluators.llm_judge import judge
from evaluators.stability_eval import stability


def run_eval(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    pass_count = 0

    for item in data:
        prompt = item['prompt']
        expected = item['expected']

        pred = call_model(prompt)
        is_pass = exact_match(pred, expected)

        if is_pass:
            pass_count += 1

        stab_score, outputs = stability(prompt)
        judge_score = judge(prompt, pred, expected)

        results.append({
            "id": item['id'],
            "prompt": prompt,
            "pred": pred,
            "expected": expected,
            "pass": is_pass,
            "stability": stab_score,
            "outputs": outputs,
            "judge": judge_score
        })

    report = {
        "total": len(data),
        "pass_rate": pass_count / len(data),
        "details": results
    }

    with open("reports/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
