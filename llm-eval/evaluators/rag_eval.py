"""
RAG系统评测示例
评测维度：答案正确性、检索相关性、回答完整性
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import json


# ==================== 1. 定义评测数据结构 ====================

@dataclass
class RAGTestCase:
    """单个测试用例"""
    question: str  # 用户问题
    ground_truth: str  # 标准答案（黄金答案）
    relevant_docs: List[str]  # 相关的文档ID列表（用于评测检索）
    expected_answer_keywords: List[str]  # 答案应包含的关键词


@dataclass
class RAGResult:
    """RAG系统的输出结果"""
    question: str
    generated_answer: str  # RAG生成的答案
    retrieved_docs: List[str]  # 检索到的文档内容
    retrieved_doc_ids: List[str]  # 检索到的文档ID


# ==================== 2. 评测指标计算函数 ====================

class RAGEvaluator:
    """RAG系统评测器"""

    def __init__(self):
        self.results = []

    # ---------- 检索指标 ----------
    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Hit Rate：检索到的相关文档比例
        公式：|检索到的相关文档| / |所有相关文档|
        """
        if not relevant_ids:
            return 1.0
        hits = set(retrieved_ids) & set(relevant_ids)
        return len(hits) / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank)：第一个相关文档的倒数排名
        公式：1 / rank_of_first_relevant
        """
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K：前K个检索结果中相关文档的比例
        """
        if k == 0:
            return 0.0
        retrieved_top_k = retrieved_ids[:k]
        relevant_retrieved = [doc for doc in retrieved_top_k if doc in relevant_ids]
        return len(relevant_retrieved) / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K：前K个检索结果召回的相关文档比例
        """
        if not relevant_ids:
            return 1.0
        retrieved_top_k = set(retrieved_ids[:k])
        relevant_retrieved = retrieved_top_k & set(relevant_ids)
        return len(relevant_retrieved) / len(relevant_ids)

    # ---------- 生成指标 ----------
    @staticmethod
    def exact_match(generated: str, ground_truth: str) -> float:
        """完全匹配（归一化后比较）"""
        return 1.0 if generated.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def keyword_coverage(generated: str, keywords: List[str]) -> float:
        """关键词覆盖率：生成答案中包含的关键词比例"""
        if not keywords:
            return 1.0
        generated_lower = generated.lower()
        covered = sum(1 for kw in keywords if kw.lower() in generated_lower)
        return covered / len(keywords)

    @staticmethod
    def answer_relevance(generated: str, question: str) -> float:
        """
        答案相关性（简化版：基于共同词汇）
        实际应用中可使用Embedding相似度或NLI模型
        """
        # 简单实现：计算问题与答案的词重叠率
        q_words = set(question.lower().split())
        a_words = set(generated.lower().split())
        if not q_words:
            return 0.0
        overlap = len(q_words & a_words)
        return overlap / len(q_words)

    @staticmethod
    def bleu_score(generated: str, reference: str) -> float:
        """
        简化版BLEU：基于n-gram精确度
        实际应用建议使用sacrebleu库
        """
        # 分词（简化）
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        # 1-gram精确度
        gen_1gram = set(gen_tokens)
        ref_1gram = set(ref_tokens)
        if not gen_1gram:
            return 0.0

        overlap_1gram = len(gen_1gram & ref_1gram)
        precision_1gram = overlap_1gram / len(gen_1gram)

        return precision_1gram  # 简化版只使用1-gram

    # ---------- 综合评测 ----------
    def evaluate_retrieval(self, results: List[RAGResult], test_cases: List[RAGTestCase]) -> Dict:
        """综合评测检索质量"""
        metrics = {
            'hit_rate': [],
            'mrr': [],
            'precision@1': [],
            'precision@3': [],
            'recall@5': []
        }

        for result, test_case in zip(results, test_cases):
            retrieved_ids = result.retrieved_doc_ids
            relevant_ids = test_case.relevant_docs

            metrics['hit_rate'].append(self.hit_rate(retrieved_ids, relevant_ids))
            metrics['mrr'].append(self.mrr(retrieved_ids, relevant_ids))
            metrics['precision@1'].append(self.precision_at_k(retrieved_ids, relevant_ids, 1))
            metrics['precision@3'].append(self.precision_at_k(retrieved_ids, relevant_ids, 3))
            metrics['recall@5'].append(self.recall_at_k(retrieved_ids, relevant_ids, 5))

        # 计算平均值
        return {k: np.mean(v) for k, v in metrics.items()}

    def evaluate_generation(self, results: List[RAGResult], test_cases: List[RAGTestCase]) -> Dict:
        """综合评测生成质量"""
        metrics = {
            'exact_match': [],
            'keyword_coverage': [],
            'answer_relevance': [],
            'bleu': []
        }

        for result, test_case in zip(results, test_cases):
            metrics['exact_match'].append(
                self.exact_match(result.generated_answer, test_case.ground_truth)
            )
            metrics['keyword_coverage'].append(
                self.keyword_coverage(result.generated_answer, test_case.expected_answer_keywords)
            )
            metrics['answer_relevance'].append(
                self.answer_relevance(result.generated_answer, test_case.question)
            )
            metrics['bleu'].append(
                self.bleu_score(result.generated_answer, test_case.ground_truth)
            )

        return {k: np.mean(v) for k, v in metrics.items()}

    def full_evaluation(self, results: List[RAGResult], test_cases: List[RAGTestCase]) -> Dict:
        """完整评测报告"""
        retrieval_metrics = self.evaluate_retrieval(results, test_cases)
        generation_metrics = self.evaluate_generation(results, test_cases)

        # 综合得分（加权平均）
        overall_score = (
                0.3 * retrieval_metrics['hit_rate'] +
                0.3 * retrieval_metrics['mrr'] +
                0.2 * generation_metrics['keyword_coverage'] +
                0.2 * generation_metrics['answer_relevance']
        )

        return {
            'retrieval_metrics': retrieval_metrics,
            'generation_metrics': generation_metrics,
            'overall_score': overall_score
        }


# ==================== 3. 模拟RAG系统 ====================

class MockRAGSystem:
    """模拟RAG系统（用于演示）"""

    def __init__(self, knowledge_base: Dict[str, str]):
        self.knowledge_base = knowledge_base

    def retrieve(self, question: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """简化的检索：基于关键词匹配"""
        question_words = set(question.lower().split())
        doc_scores = []

        for doc_id, doc_content in self.knowledge_base.items():
            # 计算共同词数量
            doc_words = set(doc_content.lower().split())
            score = len(question_words & doc_words)
            doc_scores.append((doc_id, doc_content, score))

        # 排序并取top_k
        doc_scores.sort(key=lambda x: x[2], reverse=True)
        top_docs = doc_scores[:top_k]

        doc_ids = [doc[0] for doc in top_docs]
        doc_contents = [doc[1] for doc in top_docs]

        return doc_ids, doc_contents

    def generate(self, question: str, retrieved_docs: List[str]) -> str:
        """简化的生成：从检索文档中提取信息"""
        # 合并检索内容
        context = " ".join(retrieved_docs)

        # 简单规则生成答案（真实RAG会用LLM）
        if "年假" in question:
            if "5天" in context:
                return "根据公司规定，年假为每年5天。"
            elif "10天" in context:
                return "根据公司规定，年假为每年10天。"
        elif "请假" in question:
            if "3天" in context:
                return "请假需提前3天申请。"
            elif "主管" in context:
                return "请假需经主管批准。"
        elif "报销" in question:
            if "发票" in context:
                return "报销需提供发票，流程为：提交申请→财务审核→打款。"

        return "根据现有资料，无法确认相关信息。"

    def query(self, question: str) -> str:
        """完整RAG流程"""
        doc_ids, doc_contents = self.retrieve(question)
        answer = self.generate(question, doc_contents)
        return answer, doc_ids, doc_contents


# ==================== 4. 运行评测 ====================

def main():
    # 4.1 准备知识库
    knowledge_base = {
        "doc_1": "公司年假政策：员工每年享有5天带薪年假。",
        "doc_2": "请假流程：员工请假需提前3天提交申请，经主管批准。",
        "doc_3": "报销流程：差旅费报销需提供发票，填写报销单，财务审核后打款。",
        "doc_4": "加班政策：加班可申请调休或加班费。",
        "doc_5": "社保缴纳：公司为员工缴纳五险一金。"
    }

    # 4.2 准备测试用例
    test_cases = [
        RAGTestCase(
            question="年假有多少天？",
            ground_truth="公司年假为每年5天。",
            relevant_docs=["doc_1"],
            expected_answer_keywords=["5天", "年假"]
        ),
        RAGTestCase(
            question="请假需要提前几天申请？",
            ground_truth="请假需提前3天申请。",
            relevant_docs=["doc_2"],
            expected_answer_keywords=["提前", "3天"]
        ),
        RAGTestCase(
            question="如何报销差旅费？",
            ground_truth="差旅费报销需要提供发票，填写报销单，经财务审核后打款。",
            relevant_docs=["doc_3"],
            expected_answer_keywords=["发票", "报销单", "财务"]
        ),
        RAGTestCase(
            question="公司有哪些福利？",
            ground_truth="公司提供年假、五险一金等福利。",
            relevant_docs=["doc_1", "doc_5"],
            expected_answer_keywords=["年假", "五险一金"]
        )
    ]

    # 4.3 初始化RAG系统和评测器
    rag = MockRAGSystem(knowledge_base)
    evaluator = RAGEvaluator()
    results = []

    # 4.4 执行RAG查询并收集结果
    print("=" * 60)
    print("RAG系统执行结果")
    print("=" * 60)

    for test_case in test_cases:
        answer, doc_ids, doc_contents = rag.query(test_case.question)

        result = RAGResult(
            question=test_case.question,
            generated_answer=answer,
            retrieved_docs=doc_contents,
            retrieved_doc_ids=doc_ids
        )
        results.append(result)

        # 打印每个测试用例的结果
        print(f"\n问题: {test_case.question}")
        print(f"生成答案: {answer}")
        print(f"检索到的文档: {doc_ids}")
        print(f"相关文档(标注): {test_case.relevant_docs}")
        print(f"标准答案: {test_case.ground_truth}")

    # 4.5 执行评测
    print("\n" + "=" * 60)
    print("RAG系统评测报告")
    print("=" * 60)

    evaluation = evaluator.full_evaluation(results, test_cases)

    print("\n【检索质量指标】")
    for metric, value in evaluation['retrieval_metrics'].items():
        print(f"  {metric:15s}: {value:.4f}")

    print("\n【生成质量指标】")
    for metric, value in evaluation['generation_metrics'].items():
        print(f"  {metric:15s}: {value:.4f}")

    print(f"\n【综合得分】: {evaluation['overall_score']:.4f}")

    # 4.6 按测试用例详细分析
    print("\n" + "=" * 60)
    print("逐用例详细分析")
    print("=" * 60)

    for i, (result, test_case) in enumerate(zip(results, test_cases)):
        print(f"\n用例 {i + 1}: {test_case.question}")
        print(f"  Hit Rate: {evaluator.hit_rate(result.retrieved_doc_ids, test_case.relevant_docs):.2f}")
        print(f"  MRR: {evaluator.mrr(result.retrieved_doc_ids, test_case.relevant_docs):.2f}")
        print(
            f"  关键词覆盖率: {evaluator.keyword_coverage(result.generated_answer, test_case.expected_answer_keywords):.2f}")
        print(f"  答案相关性: {evaluator.answer_relevance(result.generated_answer, test_case.question):.2f}")

    # 4.7 保存评测结果到文件
    with open('rag_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': evaluation,
            'detailed_results': [
                {
                    'question': r.question,
                    'generated_answer': r.generated_answer,
                    'ground_truth': tc.ground_truth,
                    'retrieved_docs': r.retrieved_doc_ids,
                    'relevant_docs': tc.relevant_docs
                }
                for r, tc in zip(results, test_cases)
            ]
        }, f, ensure_ascii=False, indent=2)

    print("\n评测结果已保存到 rag_evaluation_results.json")


if __name__ == "__main__":
    main()