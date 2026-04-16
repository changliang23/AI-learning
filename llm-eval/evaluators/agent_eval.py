"""
Agent评测示例 - 模拟一个旅行预订Agent
评测维度：任务完成率、工具调用正确率、轨迹效率、成本
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==================== 1. 定义评测数据结构 ====================

class ToolCallStatus(Enum):
    """工具调用状态"""
    SUCCESS = "success"
    FAILED = "failed"
    WRONG_TOOL = "wrong_tool"
    WRONG_PARAMS = "wrong_params"


@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    params: Dict[str, Any]
    result: Any
    status: ToolCallStatus
    timestamp: float
    duration_ms: float


@dataclass
class AgentTrajectory:
    """Agent执行轨迹"""
    step_id: int
    thought: str  # Agent的思考过程
    tool_calls: List[ToolCall]  # 该步骤的工具调用
    observation: str  # 观察结果
    is_final: bool = False
    final_answer: str = ""


@dataclass
class AgentTask:
    """评测任务"""
    task_id: str
    user_query: str
    expected_tools: List[str]  # 应该使用的工具
    expected_tool_sequence: List[str]  # 预期的调用顺序
    expected_answer_keywords: List[str]
    ground_truth_answer: str
    success_criteria: Dict[str, Any]  # 成功判定条件


@dataclass
class AgentResult:
    """Agent执行结果"""
    task_id: str
    user_query: str
    final_answer: str
    trajectories: List[AgentTrajectory]
    total_time_ms: float
    total_cost: float
    tool_calls_summary: List[Dict]
    success: bool
    error_message: str = ""


# ==================== 2. 评测指标计算器 ====================

class AgentEvaluator:
    """Agent评测器"""

    def __init__(self):
        self.results = []

    # ---------- 任务完成率 ----------
    @staticmethod
    def task_completion_rate(results: List[AgentResult]) -> float:
        """任务完成率 = 成功任务数 / 总任务数"""
        if not results:
            return 0.0
        success_count = sum(1 for r in results if r.success)
        return success_count / len(results)

    # ---------- 工具调用正确率 ----------
    @staticmethod
    def tool_selection_accuracy(results: List[AgentResult],
                                tasks: List[AgentTask]) -> float:
        """工具选择准确率"""
        total_calls = 0
        correct_calls = 0

        for result, task in zip(results, tasks):
            for call_summary in result.tool_calls_summary:
                total_calls += 1
                if call_summary['status'] == 'success':
                    correct_calls += 1

        return correct_calls / total_calls if total_calls > 0 else 0.0

    @staticmethod
    def tool_sequence_accuracy(results: List[AgentResult],
                               tasks: List[AgentTask]) -> float:
        """工具调用序列准确率（检查顺序是否正确）"""
        correct_sequences = 0

        for result, task in zip(results, tasks):
            actual_sequence = [call['tool_name']
                               for call in result.tool_calls_summary]

            # 检查是否按预期顺序调用
            is_correct = True
            expected_idx = 0

            for actual_tool in actual_sequence:
                if expected_idx < len(task.expected_tool_sequence):
                    if actual_tool == task.expected_tool_sequence[expected_idx]:
                        expected_idx += 1
                    # 允许中间有额外调用（如信息确认），但不跳过必要步骤
                # 如果所有预期工具都已调用，结束

            if expected_idx == len(task.expected_tool_sequence):
                correct_sequences += 1

        return correct_sequences / len(results) if results else 0.0

    # ---------- 答案质量 ----------
    @staticmethod
    def keyword_coverage(result: AgentResult, task: AgentTask) -> float:
        """关键词覆盖率"""
        if not task.expected_answer_keywords:
            return 1.0

        answer_lower = result.final_answer.lower()
        covered = sum(1 for kw in task.expected_answer_keywords
                      if kw.lower() in answer_lower)
        return covered / len(task.expected_answer_keywords)

    @staticmethod
    def exact_match_rate(results: List[AgentResult],
                         tasks: List[AgentTask]) -> float:
        """完全匹配率（归一化后）"""
        matches = 0
        for result, task in zip(results, tasks):
            if result.final_answer.strip().lower() == task.ground_truth_answer.strip().lower():
                matches += 1
        return matches / len(results) if results else 0.0

    # ---------- 效率指标 ----------
    @staticmethod
    def avg_response_time(results: List[AgentResult]) -> float:
        """平均响应时间（毫秒）"""
        if not results:
            return 0.0
        return sum(r.total_time_ms for r in results) / len(results)

    @staticmethod
    def avg_cost(results: List[AgentResult]) -> float:
        """平均成本"""
        if not results:
            return 0.0
        return sum(r.total_cost for r in results) / len(results)

    @staticmethod
    def cost_per_success(results: List[AgentResult]) -> float:
        """单次成功成本 = 总成本 / 成功任务数"""
        total_cost = sum(r.total_cost for r in results)
        success_count = sum(1 for r in results if r.success)
        return total_cost / success_count if success_count > 0 else float('inf')

    @staticmethod
    def avg_steps(results: List[AgentResult]) -> float:
        """平均执行步数"""
        if not results:
            return 0.0
        return sum(len(r.trajectories) for r in results) / len(results)

    # ---------- 可靠性指标（Pass^k）----------
    @staticmethod
    def pass_at_k(results_by_task: Dict[str, List[AgentResult]], k: int = 1) -> Dict[str, float]:
        """
        Pass@k：k次尝试中至少成功1次的概率
        results_by_task: {task_id: [result1, result2, ...]}
        """
        pass_at_k_scores = {}

        for task_id, task_results in results_by_task.items():
            if len(task_results) < k:
                continue

            # 至少一次成功
            at_least_one_success = any(r.success for r in task_results[:k])
            pass_at_k_scores[task_id] = 1.0 if at_least_one_success else 0.0

        return pass_at_k_scores

    @staticmethod
    def pass_k(results_by_task: Dict[str, List[AgentResult]], k: int = 1) -> Dict[str, float]:
        """
        Pass^k：k次尝试全部成功的概率（稳定性指标）
        """
        pass_k_scores = {}

        for task_id, task_results in results_by_task.items():
            if len(task_results) < k:
                continue

            # 全部成功
            all_success = all(r.success for r in task_results[:k])
            pass_k_scores[task_id] = 1.0 if all_success else 0.0

        return pass_k_scores

    # ---------- 轨迹评估 ----------
    @staticmethod
    def trajectory_efficiency(results: List[AgentResult]) -> Dict:
        """轨迹效率评估"""
        total_steps = 0
        redundant_calls = 0
        successful_paths = []

        for result in results:
            total_steps += len(result.trajectories)

            # 检测冗余调用（同一工具重复调用）
            tool_names = [call['tool_name']
                          for call in result.tool_calls_summary]
            redundant_calls += len(tool_names) - len(set(tool_names))

            if result.success:
                successful_paths.append(len(result.trajectories))

        return {
            'avg_steps': total_steps / len(results) if results else 0,
            'redundant_calls': redundant_calls,
            'avg_successful_steps': sum(successful_paths) / len(successful_paths) if successful_paths else 0
        }

    # ---------- 综合评测 ----------
    def full_evaluation(self, results: List[AgentResult],
                        tasks: List[AgentTask]) -> Dict:
        """完整评测报告"""

        # 按任务分组（用于可靠性指标）
        results_by_task = {}
        for result, task in zip(results, tasks):
            if task.task_id not in results_by_task:
                results_by_task[task.task_id] = []
            results_by_task[task.task_id].append(result)

        # 基础指标
        completion_rate = self.task_completion_rate(results)
        tool_accuracy = self.tool_selection_accuracy(results, tasks)
        sequence_accuracy = self.tool_sequence_accuracy(results, tasks)
        exact_match = self.exact_match_rate(results, tasks)

        # 平均关键词覆盖率
        avg_keyword_coverage = sum(
            self.keyword_coverage(r, t)
            for r, t in zip(results, tasks)
        ) / len(results) if results else 0

        # 效率指标
        avg_time = self.avg_response_time(results)
        avg_cost_val = self.avg_cost(results)
        cost_per_success_val = self.cost_per_success(results)

        # 可靠性指标（需要多次运行）
        pass_1 = self.pass_at_k(results_by_task, k=1)
        pass_3 = self.pass_at_k(results_by_task, k=3)
        pass_2_2 = self.pass_k(results_by_task, k=2)

        # 轨迹指标
        trajectory_metrics = self.trajectory_efficiency(results)

        # 综合得分（加权）
        overall_score = (
                0.3 * completion_rate +
                0.2 * tool_accuracy +
                0.15 * avg_keyword_coverage +
                0.15 * (1 - min(1, avg_time / 30000)) +  # 30秒基准
                0.2 * (1 / (1 + avg_cost_val))  # 成本越低越好
        )

        return {
            'task_completion_rate': completion_rate,
            'tool_selection_accuracy': tool_accuracy,
            'tool_sequence_accuracy': sequence_accuracy,
            'exact_match_rate': exact_match,
            'avg_keyword_coverage': avg_keyword_coverage,
            'avg_response_time_ms': avg_time,
            'avg_cost_per_task': avg_cost_val,
            'cost_per_success': cost_per_success_val,
            'pass@1_avg': sum(pass_1.values()) / len(pass_1) if pass_1 else 0,
            'pass@3_avg': sum(pass_3.values()) / len(pass_3) if pass_3 else 0,
            'pass^2_avg': sum(pass_2_2.values()) / len(pass_2_2) if pass_2_2 else 0,
            'trajectory_metrics': trajectory_metrics,
            'overall_score': overall_score
        }


# ==================== 3. 模拟Agent ====================

class MockTravelAgent:
    """
    模拟旅行预订Agent
    可以预订航班、酒店、租车
    """

    def __init__(self, mode: str = "normal"):
        """
        mode: normal, slow, error_prone, inefficient
        """
        self.mode = mode
        self.tools = {
            'search_flight': self.search_flight,
            'book_flight': self.book_flight,
            'search_hotel': self.search_hotel,
            'book_hotel': self.book_hotel,
            'search_car': self.search_car,
            'book_car': self.book_car
        }

    def search_flight(self, origin: str, destination: str, date: str) -> Dict:
        """搜索航班"""
        if self.mode == "slow":
            time.sleep(0.5)
        return {
            'flights': [
                {'id': 'CA1234', 'price': 800, 'departure': '08:00', 'arrival': '10:30'},
                {'id': 'MU5678', 'price': 750, 'departure': '14:00', 'arrival': '16:30'}
            ]
        }

    def book_flight(self, flight_id: str, passenger_name: str) -> Dict:
        """预订航班"""
        if self.mode == "error_prone" and flight_id == 'CA1234':
            return {'success': False, 'error': '航班已满'}
        return {'success': True, 'booking_id': f'FL_{flight_id}'}

    def search_hotel(self, city: str, check_in: str, check_out: str) -> Dict:
        """搜索酒店"""
        return {
            'hotels': [
                {'id': 'H001', 'name': '希尔顿', 'price': 600},
                {'id': 'H002', 'name': '喜来登', 'price': 550}
            ]
        }

    def book_hotel(self, hotel_id: str, guest_name: str) -> Dict:
        """预订酒店"""
        return {'success': True, 'booking_id': f'HT_{hotel_id}'}

    def search_car(self, location: str, pick_up: str, drop_off: str) -> Dict:
        """搜索租车"""
        return {
            'cars': [
                {'id': 'C001', 'type': 'SUV', 'price': 300},
                {'id': 'C002', 'type': 'Sedan', 'price': 200}
            ]
        }

    def book_car(self, car_id: str, driver_name: str) -> Dict:
        """预订租车"""
        return {'success': True, 'booking_id': f'CR_{car_id}'}

    def _call_tool(self, tool_name: str, params: Dict) -> tuple:
        """执行工具调用并记录状态"""
        start_time = time.time()

        if tool_name not in self.tools:
            return None, ToolCallStatus.WRONG_TOOL, "Tool not found"

        try:
            result = self.tools[tool_name](**params)
            duration = (time.time() - start_time) * 1000

            # 检查参数是否正确
            if self.mode == "error_prone" and tool_name == 'book_flight':
                if 'flight_id' not in params or 'passenger_name' not in params:
                    return None, ToolCallStatus.WRONG_PARAMS, "Missing required params"

            # 检查结果是否成功
            if isinstance(result, dict) and result.get('success') is False:
                return result, ToolCallStatus.FAILED, result.get('error', '')

            return result, ToolCallStatus.SUCCESS, ""

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return None, ToolCallStatus.FAILED, str(e)

    def run(self, user_query: str, max_steps: int = 10) -> AgentResult:
        """
        执行Agent任务
        简化版：基于规则的Agent，解析用户意图并执行相应操作
        """
        start_time = time.time()
        trajectories = []
        tool_calls_summary = []
        total_cost = 0.0

        # 模拟成本（每次LLM调用$0.001，每次工具调用$0.0001）
        llm_calls = 0
        total_cost += 0.001  # 初始思考

        # 解析用户意图
        query_lower = user_query.lower()

        # 构建执行计划
        steps = []

        if 'flight' in query_lower or '航班' in query_lower:
            # 提取参数（简化）
            import re
            city_match = re.search(r'(?:从|from)\s*(\w+)', query_lower)
            if '北京' in query_lower or 'beijing' in query_lower:
                origin = '北京'
                destination = '上海'
            else:
                origin = '北京'
                destination = '上海'

            steps.append(('search_flight', {'origin': origin, 'destination': destination, 'date': '2025-05-01'}))
            steps.append(('book_flight', {'flight_id': 'CA1234', 'passenger_name': '张三'}))

        if 'hotel' in query_lower or '酒店' in query_lower:
            steps.append(('search_hotel', {'city': '上海', 'check_in': '2025-05-01', 'check_out': '2025-05-03'}))
            steps.append(('book_hotel', {'hotel_id': 'H001', 'guest_name': '张三'}))

        if 'car' in query_lower or '租车' in query_lower:
            steps.append(('search_car', {'location': '上海', 'pick_up': '2025-05-01', 'drop_off': '2025-05-03'}))
            steps.append(('book_car', {'car_id': 'C001', 'driver_name': '张三'}))

        # 模拟慢速模式
        if self.mode == "slow":
            time.sleep(1)

        # 执行步骤
        for step_id, (tool_name, params) in enumerate(steps):
            llm_calls += 1
            total_cost += 0.001

            result, status, error = self._call_tool(tool_name, params)

            call_summary = {
                'tool_name': tool_name,
                'params': params,
                'status': status.value,
                'error': error
            }
            tool_calls_summary.append(call_summary)

            trajectory = AgentTrajectory(
                step_id=step_id,
                thought=f"我需要调用{tool_name}来完成当前步骤",
                tool_calls=[ToolCall(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    status=status,
                    timestamp=time.time(),
                    duration_ms=100
                )],
                observation=str(result) if result else error
            )
            trajectories.append(trajectory)

            # 模拟成本
            total_cost += 0.0001

            # 如果调用失败，停止执行
            if status != ToolCallStatus.SUCCESS:
                break

        # 生成最终答案
        success = all(call['status'] == 'success' for call in tool_calls_summary)

        if success:
            bookings = [call['tool_name'].replace('book_', '')
                        for call in tool_calls_summary if call['tool_name'].startswith('book_')]
            final_answer = f"已成功预订：{', '.join(bookings)}。"
        else:
            failed_tools = [call['tool_name'] for call in tool_calls_summary
                            if call['status'] != 'success']
            final_answer = f"预订失败：{', '.join(failed_tools)} 出现问题。"

        # 模拟错误模式
        if self.mode == "error_prone" and not success:
            final_answer = "抱歉，系统暂时无法完成预订，请稍后重试。"

        total_time = (time.time() - start_time) * 1000

        return AgentResult(
            task_id="",
            user_query=user_query,
            final_answer=final_answer,
            trajectories=trajectories,
            total_time_ms=total_time,
            total_cost=total_cost,
            tool_calls_summary=tool_calls_summary,
            success=success
        )


# ==================== 4. 运行评测 ====================

def run_evaluation(agent_mode: str = "normal", num_runs: int = 1):
    """运行Agent评测"""

    # 定义评测任务
    tasks = [
        AgentTask(
            task_id="T001",
            user_query="帮我预订从北京到上海的航班",
            expected_tools=["search_flight", "book_flight"],
            expected_tool_sequence=["search_flight", "book_flight"],
            expected_answer_keywords=["航班", "预订", "成功"],
            ground_truth_answer="已成功预订：flight。",
            success_criteria={"must_have_booking": True}
        ),
        AgentTask(
            task_id="T002",
            user_query="预订上海的希尔顿酒店，住两晚",
            expected_tools=["search_hotel", "book_hotel"],
            expected_tool_sequence=["search_hotel", "book_hotel"],
            expected_answer_keywords=["酒店", "希尔顿", "预订"],
            ground_truth_answer="已成功预订：hotel。",
            success_criteria={}
        ),
        AgentTask(
            task_id="T003",
            user_query="帮我预订航班和酒店，从北京到上海，住希尔顿",
            expected_tools=["search_flight", "book_flight", "search_hotel", "book_hotel"],
            expected_tool_sequence=["search_flight", "book_flight", "search_hotel", "book_hotel"],
            expected_answer_keywords=["航班", "酒店", "希尔顿", "预订"],
            ground_truth_answer="已成功预订：flight, hotel。",
            success_criteria={}
        )
    ]

    # 初始化Agent和评测器
    agent = MockTravelAgent(mode=agent_mode)
    evaluator = AgentEvaluator()

    all_results = []

    # 多次运行（用于计算Pass@k）
    for run_id in range(num_runs):
        print(f"\n--- 第 {run_id + 1} 次运行 ---")
        run_results = []

        for task in tasks:
            result = agent.run(task.user_query)
            result.task_id = task.task_id
            run_results.append(result)

            print(f"任务 {task.task_id}: {result.user_query}")
            print(f"  答案: {result.final_answer}")
            print(f"  成功: {result.success}")
            print(f"  耗时: {result.total_time_ms:.0f}ms")
            print(f"  成本: ${result.total_cost:.4f}")
            print(f"  工具调用: {[c['tool_name'] for c in result.tool_calls_summary]}")
            print(f"  状态: {[c['status'] for c in result.tool_calls_summary]}")

        all_results.extend(run_results)

    # 分组结果（按任务ID）
    results_by_task = {}
    for result in all_results:
        if result.task_id not in results_by_task:
            results_by_task[result.task_id] = []
        results_by_task[result.task_id].append(result)

    # 执行评测（每个任务取最后一次运行结果用于基础指标）
    # 注意：实际评估中，应该有对应关系
    last_results = all_results[-len(tasks):] if all_results else []

    evaluation = evaluator.full_evaluation(last_results, tasks)

    # 打印报告
    print("\n" + "=" * 60)
    print(f"Agent评测报告 - 模式: {agent_mode}")
    print("=" * 60)

    print("\n【任务完成情况】")
    print(f"  任务完成率: {evaluation['task_completion_rate']:.2%}")

    print("\n【工具调用质量】")
    print(f"  工具选择准确率: {evaluation['tool_selection_accuracy']:.2%}")
    print(f"  工具序列准确率: {evaluation['tool_sequence_accuracy']:.2%}")

    print("\n【答案质量】")
    print(f"  完全匹配率: {evaluation['exact_match_rate']:.2%}")
    print(f"  平均关键词覆盖率: {evaluation['avg_keyword_coverage']:.2%}")

    print("\n【效率指标】")
    print(f"  平均响应时间: {evaluation['avg_response_time_ms']:.0f}ms")
    print(f"  平均每任务成本: ${evaluation['avg_cost_per_task']:.4f}")
    print(f"  单次成功成本: ${evaluation['cost_per_success']:.4f}")
    print(f"  平均执行步数: {evaluation['trajectory_metrics']['avg_steps']:.1f}")
    print(f"  冗余调用次数: {evaluation['trajectory_metrics']['redundant_calls']}")

    print("\n【可靠性指标】")
    print(f"  Pass@1 (平均): {evaluation['pass@1_avg']:.2%}")
    if num_runs >= 3:
        print(f"  Pass@3 (平均): {evaluation['pass@3_avg']:.2%}")
    if num_runs >= 2:
        print(f"  Pass^2 (平均): {evaluation['pass^2_avg']:.2%}")

    print(f"\n【综合得分】: {evaluation['overall_score']:.4f}")

    return evaluation, all_results


# ==================== 5. 对比不同配置的Agent ====================

def compare_agents():
    """对比不同配置的Agent性能"""
    print("\n" + "=" * 70)
    print("Agent配置对比实验")
    print("=" * 70)

    configs = ["normal", "slow", "error_prone"]
    results_summary = {}

    for config in configs:
        print(f"\n>>> 测试配置: {config}")
        eval_result, _ = run_evaluation(agent_mode=config, num_runs=3)
        results_summary[config] = eval_result

    # 对比表格
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)
    print(f"{'配置':<12} {'完成率':<10} {'工具准确率':<12} {'响应时间(ms)':<14} {'成本':<10} {'综合得分':<10}")
    print("-" * 70)

    for config, metrics in results_summary.items():
        print(f"{config:<12} {metrics['task_completion_rate']:<10.2%} "
              f"{metrics['tool_selection_accuracy']:<12.2%} "
              f"{metrics['avg_response_time_ms']:<14.0f} "
              f"${metrics['avg_cost_per_task']:<9.4f} "
              f"{metrics['overall_score']:<10.4f}")

    return results_summary


# ==================== 6. 主函数 ====================

if __name__ == "__main__":
    # 单次评测
    print("=== 单次Agent评测 ===\n")
    run_evaluation(agent_mode="normal", num_runs=1)

    # 对比评测（取消注释以运行）
    # print("\n\n=== 多配置对比评测 ===\n")
    # compare_agents()