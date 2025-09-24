import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path

class PerformanceAnalyzer:
    """Comprehensive performance analysis for hardware accelerator benchmarks.

    This class ingests benchmark data in the format produced by VerilatorGEMM.benchmark_matrix_sizes():
    {
      "64x64": [ {"hw_cycles":..., "hw_gops":..., "speedup":..., "relative_error":...}, ... ],
      ...
    }
    """

    def __init__(self, results_dir: str = "results/performance"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def analyze_benchmark_results(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate a report and visualizations.

        Returns a dictionary with analysis results.
        """
        analysis = {
            "size_analysis": self._analyze_by_size(benchmark_data),
            "performance_trends": self._analyze_performance_trends(benchmark_data),
            "efficiency_analysis": self._analyze_efficiency(benchmark_data),
            "scaling_analysis": self._analyze_scaling(benchmark_data),
        }

        # Generate visualizations
        try:
            self._plot_performance_analysis(analysis)
        except Exception as e:
            print(f"Warning: plotting failed: {e}")

        # Save analysis results (make JSON-serializable)
        def make_serializable(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(analysis, default=make_serializable))
        with open(self.results_dir / "performance_analysis.json", "w") as f:
            json.dump(serializable, f, indent=2)

        return analysis

    def _analyze_by_size(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        size_analysis: Dict[str, Any] = {}

        for size_key, results in benchmark_data.items():
            # Expect key like "64x64"
            size = int(size_key.split("x")[0])

            hw_cycles = np.array([r.get("hw_cycles", 0) for r in results], dtype=float)
            hw_gops = np.array([r.get("hw_gops", 0) for r in results], dtype=float)
            speedups = np.array([r.get("speedup", 0) for r in results], dtype=float)
            errors = np.array([r.get("relative_error", 0) for r in results], dtype=float)

            avg_cycles = float(np.mean(hw_cycles)) if hw_cycles.size else 0.0
            avg_gops = float(np.mean(hw_gops)) if hw_gops.size else 0.0
            avg_speedup = float(np.mean(speedups)) if speedups.size else 0.0
            avg_error = float(np.mean(errors)) if errors.size else 0.0
            std_speedup = float(np.std(speedups)) if speedups.size else 0.0

            operations = 2 * (size ** 3)

            # Avoid division by zero when computing theoretical GOPS
            mean_cycles_for_theory = np.mean([r.get("hw_cycles", 1) for r in results]) if results else 1
            frequency = 100e6  # assume 100 MHz unless provided in metrics
            theoretical_gops = (operations / mean_cycles_for_theory) * (frequency / 1e9) if mean_cycles_for_theory > 0 else 0.0

            size_analysis[size_key] = {
                "matrix_size": size,
                "avg_hw_cycles": avg_cycles,
                "avg_hw_gops": avg_gops,
                "avg_speedup": avg_speedup,
                "avg_error": avg_error,
                "std_speedup": std_speedup,
                "hw_utilization": self._calculate_utilization(results),
                "operations": int(operations),
                "theoretical_gops": float(theoretical_gops),
            }

        return size_analysis

    def _analyze_performance_trends(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        sizes = []
        speedups = []
        gops = []
        utilizations = []
        errors = []

        for size_key, results in benchmark_data.items():
            size = int(size_key.split("x")[0])
            sizes.append(size)
            speedups.append(np.mean([r.get("speedup", 0) for r in results]) if results else 0.0)
            gops.append(np.mean([r.get("hw_gops", 0) for r in results]) if results else 0.0)
            utilizations.append(self._calculate_utilization(results))
            errors.append(np.mean([r.get("relative_error", 0) for r in results]) if results else 0.0)

        if len(sizes) >= 2:
            # Fit quadratic trends where possible
            speedup_trend = np.polyfit(sizes, speedups, 2).tolist()
            gops_trend = np.polyfit(sizes, gops, 2).tolist()
        else:
            speedup_trend = [0, 0, float(speedups[0]) if speedups else 0.0]
            gops_trend = [0, 0, float(gops[0]) if gops else 0.0]

        peak_speedup = float(np.max(speedups)) if speedups else 0.0
        peak_gops = float(np.max(gops)) if gops else 0.0
        best_size = int(sizes[np.argmax(speedups)]) if sizes else 0

        return {
            "sizes": sizes,
            "speedup_trend": speedup_trend,
            "gops_trend": gops_trend,
            "peak_speedup": peak_speedup,
            "peak_gops": peak_gops,
            "best_size": best_size,
            "avg_utilization": float(np.mean(utilizations)) if utilizations else 0.0,
            "avg_error": float(np.mean(errors)) if errors else 0.0,
        }

    def _analyze_efficiency(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        total_operations = 0
        total_cycles = 0
        total_energy_estimate = 0.0

        for size_key, results in benchmark_data.items():
            size = int(size_key.split("x")[0])
            ops = 2 * (size ** 3)
            for result in results:
                cycles = result.get("hw_cycles", 0)
                total_operations += ops
                total_cycles += cycles
                total_energy_estimate += float(cycles)  # simplified

        avg_ops_per_cycle = (total_operations / total_cycles) if total_cycles > 0 else 0.0
        theoretical_peak = 8 * 8 * 2  # 8x8 PE array, 2 ops per MAC

        efficiency_percentage = (avg_ops_per_cycle / theoretical_peak * 100.0) if theoretical_peak > 0 else 0.0
        energy_efficiency_estimate = (total_operations / total_energy_estimate) if total_energy_estimate > 0 else 0.0

        return {
            "total_operations": int(total_operations),
            "total_cycles": int(total_cycles),
            "avg_ops_per_cycle": float(avg_ops_per_cycle),
            "theoretical_peak_ops_per_cycle": int(theoretical_peak),
            "efficiency_percentage": float(efficiency_percentage),
            "energy_efficiency_estimate": float(energy_efficiency_estimate),
        }

    def _analyze_scaling(self, benchmark_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        sizes = []
        normalized_throughput = []

        for size_key, results in benchmark_data.items():
            size = int(size_key.split("x")[0])
            ops = 2 * (size ** 3)
            avg_cycles = np.mean([r.get("hw_cycles", 1) for r in results]) if results else 1
            throughput = (ops / avg_cycles) if avg_cycles > 0 else 0.0
            sizes.append(size)
            normalized_throughput.append(throughput)

        if len(sizes) >= 2 and all(t > 0 for t in normalized_throughput):
            log_sizes = np.log2(sizes)
            log_throughput = np.log2(normalized_throughput)
            scaling_coeffs = np.polyfit(log_sizes, log_throughput, 1)
            scaling_exponent = float(scaling_coeffs[0])
        else:
            scaling_exponent = 0.0

        return {
            "sizes": sizes,
            "normalized_throughput": normalized_throughput,
            "scaling_exponent": scaling_exponent,
            "scaling_interpretation": self._interpret_scaling(scaling_exponent),
        }

    def _calculate_utilization(self, results: List[Dict[str, Any]]) -> float:
        theoretical_peak = 100.0  # GOPS
        actual = [r.get("hw_gops", 0.0) for r in results]
        return float(np.mean(actual) / theoretical_peak * 100.0) if actual else 0.0

    def _interpret_scaling(self, exponent: float) -> str:
        if exponent > 2.5:
            return "Super-linear scaling (excellent cache behavior)"
        elif exponent > 1.5:
            return "Good scaling (near-linear)"
        elif exponent > 0.5:
            return "Sublinear scaling (bandwidth limited)"
        else:
            return "Poor scaling (overhead dominated)"

    def _plot_performance_analysis(self, analysis: Dict[str, Any]):
        size_analysis = analysis["size_analysis"]
        sizes = [int(k.split("x")[0]) for k in size_analysis.keys()]
        sizes_sorted_idx = np.argsort(sizes)
        sizes_sorted = [sizes[i] for i in sizes_sorted_idx]

        speedups = [size_analysis[f"{s}x{s}"]["avg_speedup"] for s in sizes_sorted]
        gops = [size_analysis[f"{s}x{s}"]["avg_hw_gops"] for s in sizes_sorted]
        errors = [size_analysis[f"{s}x{s}"]["avg_error"] for s in sizes_sorted]
        utilizations = [size_analysis[f"{s}x{s}"]["hw_utilization"] for s in sizes_sorted]
        cycles = [size_analysis[f"{s}x{s}"]["avg_hw_cycles"] for s in sizes_sorted]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Hardware Accelerator Performance Analysis", fontsize=16)

        # Speedup
        axes[0, 0].plot(sizes_sorted, speedups, marker='o')
        axes[0, 0].set_xlabel('Matrix Size (N×N)')
        axes[0, 0].set_ylabel('Speedup vs CPU')
        axes[0, 0].grid(True, alpha=0.3)

        # GOPS
        axes[0, 1].plot(sizes_sorted, gops, marker='o')
        axes[0, 1].set_xlabel('Matrix Size (N×N)')
        axes[0, 1].set_ylabel('GOPS')
        axes[0, 1].grid(True, alpha=0.3)

        # Error
        axes[0, 2].semilogy(sizes_sorted, errors, marker='o')
        axes[0, 2].set_xlabel('Matrix Size (N×N)')
        axes[0, 2].set_ylabel('Relative Error')
        axes[0, 2].grid(True, alpha=0.3)

        # Utilization bar
        axes[1, 0].bar([str(s) for s in sizes_sorted], utilizations)
        axes[1, 0].set_ylabel('Utilization (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Cycles vs theoretical
        theoretical_cycles = [(s ** 3) / (8 * 8 * 100) for s in sizes_sorted]
        axes[1, 1].loglog(sizes_sorted, cycles, marker='o', label='Actual')
        axes[1, 1].loglog(sizes_sorted, theoretical_cycles, linestyle='--', label='Theoretical')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Summary bar
        metrics = ['Avg Speedup', 'Peak GOPS', 'HW Utilization', 'Accuracy']
        values = [np.mean(speedups) if speedups else 0.0,
                  max(gops) if gops else 0.0,
                  np.mean(utilizations) if utilizations else 0.0,
                  100.0 - (np.mean(errors) * 100.0) if errors else 0.0]

        axes[1, 2].bar(metrics, values)
        axes[1, 2].set_ylabel('Normalized Score')

        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Detailed comparison
        self._plot_detailed_comparison(analysis)

    def _plot_detailed_comparison(self, analysis: Dict[str, Any]):
        size_analysis = analysis['size_analysis']
        sizes = [int(k.split('x')[0]) for k in size_analysis.keys()]
        sizes_sorted_idx = np.argsort(sizes)
        sizes_sorted = [sizes[i] for i in sizes_sorted_idx]

        hw_times = []
        sw_times = []
        for s in sizes_sorted:
            key = f"{s}x{s}"
            metrics = size_analysis[key]
            hw_cycles = metrics['avg_hw_cycles']
            hw_time_ms = hw_cycles / 100e6 * 1000.0
            sw_time_ms = hw_time_ms * max(metrics['avg_speedup'], 1.0)
            hw_times.append(hw_time_ms)
            sw_times.append(sw_time_ms)

        x = np.arange(len(sizes_sorted))
        width = 0.35

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].bar(x - width/2, sw_times, width, label='CPU Software', color='red')
        axes[0, 0].bar(x + width/2, hw_times, width, label='FPGA Hardware', color='blue')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{s}×{s}" for s in sizes_sorted])
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # GOPS scaling
        gops = [size_analysis[f"{s}x{s}"]['avg_hw_gops'] for s in sizes_sorted]
        theoretical_peak = 100.0
        axes[0, 1].plot(sizes_sorted, gops, marker='o')
        axes[0, 1].axhline(y=theoretical_peak, color='r', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)

        # Utilization
        efficiency = [size_analysis[f"{s}x{s}"]['hw_utilization'] for s in sizes_sorted]
        axes[1, 0].plot(sizes_sorted, efficiency, marker='o')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)

        # Error analysis
        errors = [size_analysis[f"{s}x{s}"]['avg_error'] * 100.0 for s in sizes_sorted]
        axes[1, 1].semilogy(sizes_sorted, errors, marker='o')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_performance_report(self, analysis: Dict[str, Any]) -> str:
        report = []
        report.append("# Hardware Accelerator Performance Report\n\n")
        report.append("## Executive Summary\n\n")

        performance_trends = analysis['performance_trends']
        efficiency_analysis = analysis['efficiency_analysis']

        report.append(f"- **Peak Speedup**: {performance_trends.get('peak_speedup', 0.0):.2f}x over CPU\n")
        report.append(f"- **Peak Throughput**: {performance_trends.get('peak_gops', 0.0):.2f} GOPS\n")
        report.append(f"- **Best Matrix Size**: {performance_trends.get('best_size', 0)}×{performance_trends.get('best_size', 0)}\n")
        report.append(f"- **Average Hardware Utilization**: {performance_trends.get('avg_utilization', 0.0):.1f}%\n")
        report.append(f"- **Hardware Efficiency**: {efficiency_analysis.get('efficiency_percentage', 0.0):.1f}%\n")
        report.append(f"- **Average Numerical Error**: {performance_trends.get('avg_error', 0.0)*100:.4f}%\n\n")

        report.append("## Detailed Analysis\n\n")
        for size_key, metrics in analysis['size_analysis'].items():
            report.append(f"**{size_key} Matrix**:\n")
            report.append(f"  - Speedup: {metrics.get('avg_speedup', 0.0):.2f}x\n")
            report.append(f"  - Throughput: {metrics.get('avg_hw_gops', 0.0):.2f} GOPS\n")
            report.append(f"  - Cycles: {metrics.get('avg_hw_cycles', 0.0):.0f}\n")
            report.append(f"  - Utilization: {metrics.get('hw_utilization', 0.0):.1f}%\n")
            report.append(f"  - Numerical Error: {metrics.get('avg_error', 0.0)*100:.4f}%\n\n")

        scaling = analysis['scaling_analysis']
        report.append("### Scaling Behavior\n\n")
        report.append(f"- **Scaling Exponent**: {scaling.get('scaling_exponent', 0.0):.2f}\n")
        report.append(f"- **Interpretation**: {scaling.get('scaling_interpretation', '')}\n\n")

        report.append("## Recommendations\n\n")
        best_size = performance_trends.get('best_size', 0)
        avg_utilization = performance_trends.get('avg_utilization', 0.0)

        if avg_utilization < 50:
            report.append("- **Low utilization detected**: Consider optimizing data flow or increasing parallelism\n")
        if performance_trends.get('peak_speedup', 0.0) < 5:
            report.append("- **Limited speedup**: Investigate memory bandwidth bottlenecks\n")

        report.append(f"- **Optimal operating point**: Use {best_size}×{best_size} matrices for best performance\n")

        if performance_trends.get('avg_error', 0.0) > 0.01:
            report.append("- **Numerical accuracy**: Consider higher precision arithmetic if needed\n")

        report_text = ''.join(report)
        with open(self.results_dir / "performance_report.md", "w") as f:
            f.write(report_text)

        return report_text


# Usage Example
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()

    example_data = {
        "64x64": [
            {"hw_cycles": 8000, "hw_gops": 12.8, "speedup": 3.2, "relative_error": 0.001},
            {"hw_cycles": 8100, "hw_gops": 12.6, "speedup": 3.1, "relative_error": 0.0012},
        ],
        "128x128": [
            {"hw_cycles": 32000, "hw_gops": 25.6, "speedup": 5.8, "relative_error": 0.0008},
            {"hw_cycles": 31800, "hw_gops": 25.8, "speedup": 5.9, "relative_error": 0.0009},
        ],
    }

    analysis = analyzer.analyze_benchmark_results(example_data)
    report = analyzer.generate_performance_report(analysis)
    print(report)
