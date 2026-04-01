from core.runner import run_eval

if __name__ == "__main__":
    report = run_eval("datasets/sample.json")
    print("Evaluation Done")
    print(report)