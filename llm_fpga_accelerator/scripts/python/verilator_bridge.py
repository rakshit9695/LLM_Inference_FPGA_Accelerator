import subprocess
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import time
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VerilatorGEMM:
    """Python interface to a Verilator GEMM accelerator simulation.

    This class wraps invocation of the compiled Verilator binary and provides
    convenience functions to prepare inputs, run the simulation, and parse outputs.
    """

    def __init__(
        self,
        sim_executable: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        # Auto-detect executable path based on common locations
        if sim_executable is None:
            # Try different possible locations relative to script location
            script_dir = Path(__file__).parent
            possible_paths = [
                # From scripts/python/ directory
                script_dir / "../../sim/obj_dir/Vgemm_accelerator",
                script_dir / "../../sim/build/obj_dir/Vgemm_accelerator", 
                # From project root
                Path("sim/obj_dir/Vgemm_accelerator"),
                Path("sim/build/obj_dir/Vgemm_accelerator"),
                # Current directory
                Path("./obj_dir/Vgemm_accelerator"),
                Path("obj_dir/Vgemm_accelerator"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    sim_executable = str(path.resolve())
                    logger.info(f"Found executable at: {sim_executable}")
                    break
            else:
                # If not found, provide helpful error message
                logger.error("Verilator executable not found in expected locations:")
                for path in possible_paths:
                    logger.error(f"  Checked: {path} - {'EXISTS' if path.exists() else 'NOT FOUND'}")
                raise FileNotFoundError(
                    "Simulator executable not found. Please ensure you have:\n"
                    "1. Built the simulation with 'cd sim && make'\n"
                    "2. Or provide the correct path to the executable\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Script location: {script_dir}"
                )
        
        self.sim_executable = Path(sim_executable)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Final verification
        if not self.sim_executable.exists():
            raise FileNotFoundError(f"Simulator executable not found: {self.sim_executable}")

        logger.info(f"VerilatorGEMM initialized with executable: {self.sim_executable}")
        logger.info(f"Using temp directory: {self.temp_dir}")

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []

    def matrix_multiply(
        self,
        A: np.ndarray,
        B: np.ndarray,
        data_format: str = "int16",
        timeout: float = 30.0,
        enable_trace: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform matrix multiplication using the hardware accelerator.

        Args:
            A: Input matrix A with shape (M, K)
            B: Input matrix B with shape (K, N) 
            data_format: One of "int16", "fp16", "bf16"
            timeout: Maximum time to wait for the simulation (seconds)
            enable_trace: Enable VCD waveform tracing

        Returns:
            (result_matrix (M x N), metrics dict)
        """

        # Basic validation
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D matrices")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: A.shape={A.shape}, B.shape={B.shape}")

        M, K = A.shape
        _, N = B.shape

        logger.info(f"Computing {M}x{N} @ {K} matrix multiplication (format={data_format})")

        # Convert data format for hardware
        A_hw, B_hw = self._convert_matrices(A, B, data_format)

        # Create input files
        input_files = self._create_input_files(A_hw, B_hw, M, N, K, data_format)

        # Run simulation
        start_time = time.perf_counter()
        result_matrix, metrics = self._run_simulation(input_files, M, N, K, timeout, enable_trace)
        end_time = time.perf_counter()

        metrics["python_time"] = end_time - start_time
        metrics["data_format"] = data_format
        metrics["matrix_dimensions"] = (M, N, K)

        # Store history
        self.performance_history.append(metrics)

        return result_matrix, metrics

    def _convert_matrices(self, A: np.ndarray, B: np.ndarray, data_format: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert matrices to hardware-compatible binary representations."""
        if data_format == "int16":
            # Convert floats to int16 range [-32768, 32767]
            # Scale to use most of the dynamic range
            A_scaled = np.clip(A * 1000, -32768, 32767)  # Scale by 1000 for precision
            B_scaled = np.clip(B * 1000, -32768, 32767)
            A_hw = A_scaled.astype(np.int16)
            B_hw = B_scaled.astype(np.int16)
        elif data_format == "fp16":
            A_hw = A.astype(np.float16).view(np.uint16)
            B_hw = B.astype(np.float16).view(np.uint16)
        elif data_format == "bf16":
            A_hw = self._float32_to_bfloat16(A.astype(np.float32))
            B_hw = self._float32_to_bfloat16(B.astype(np.float32))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        return A_hw, B_hw

    def _float32_to_bfloat16(self, x: np.ndarray) -> np.ndarray:
        """Convert float32 numpy array to bfloat16 represented as uint16 bit patterns."""
        x_uint32 = x.view(np.uint32)
        bf16 = (x_uint32 >> 16).astype(np.uint16)
        return bf16

    def _create_input_files(self, A: np.ndarray, B: np.ndarray, M: int, N: int, K: int, data_format: str) -> Dict[str, Path]:
        """Create binary and JSON files consumed by the simulation executable."""
        files: Dict[str, Path] = {}

        # Create configuration JSON
        config = {
            "matrix_m": M,
            "matrix_n": N,
            "matrix_k": K,
            "data_format": {"int16": 2, "fp16": 0, "bf16": 1}.get(data_format, 2),
            "addr_a_base": 4096,    # 0x1000
            "addr_b_base": 20480,   # 0x5000  
            "addr_c_base": 36864,   # 0x9000
        }

        config_file = self.temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        files["config"] = config_file

        # Write matrix A as binary (row-major order)
        a_file = self.temp_dir / "matrix_a.bin"
        A.tobytes('C')  # Ensure C-contiguous (row-major)
        A.tofile(a_file)
        files["matrix_a"] = a_file

        # Write matrix B as binary (row-major order)
        b_file = self.temp_dir / "matrix_b.bin"
        B.tobytes('C')  # Ensure C-contiguous (row-major)
        B.tofile(b_file)
        files["matrix_b"] = b_file

        # Output file paths
        files["output"] = self.temp_dir / "result.bin"
        files["metrics"] = self.temp_dir / "metrics.json"

        return files

    def _run_simulation(self, input_files: Dict[str, Path], M: int, N: int, K: int, timeout: float, enable_trace: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the Verilator simulation binary and collect its outputs."""
        cmd = [
            str(self.sim_executable),
            "--config", str(input_files["config"]),
            "--matrix-a", str(input_files["matrix_a"]),
            "--matrix-b", str(input_files["matrix_b"]),
            "--output", str(input_files["output"]),
            "--metrics", str(input_files["metrics"]),
        ]
        
        if enable_trace:
            cmd.append("--trace")
        else:
            cmd.append("--no-trace")

        logger.debug(f"Running simulation: {' '.join(cmd)}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Executable path: {self.sim_executable}")

        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.temp_dir,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            
            logger.debug(f"Simulation stdout:\\n{stdout}")
            if stderr:
                logger.warning(f"Simulation stderr:\\n{stderr}")
                
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Simulation timed out after {timeout} seconds") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"Simulation failed (rc={e.returncode}): {e.stderr}")
            raise RuntimeError(f"Simulation failed with return code {e.returncode}: {e.stderr}") from e

        # Read result binary (expecting int32 per element)
        out_path = input_files["output"]
        if not out_path.exists():
            raise RuntimeError("Simulation did not produce output file")

        result_data = np.fromfile(out_path, dtype=np.int32)
        try:
            result_matrix = result_data.reshape((M, N))
        except ValueError:
            raise RuntimeError(f"Result file shape mismatch: expected {M}x{N}, got {result_data.size} elements")

        # Load metrics if produced
        metrics: Dict[str, Any] = {}
        metrics_path = input_files["metrics"]
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Metrics file exists but is not valid JSON")
        else:
            logger.debug("No metrics file produced by simulation")

        # Parse stdout for additional info
        self._parse_simulation_output(stdout, metrics)

        return result_matrix, metrics

    def _parse_simulation_output(self, stdout: str, metrics: Dict[str, Any]):
        """Parse simulation stdout for known metric lines."""
        if not stdout:
            return
            
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if "Simulation cycles:" in line:
                try:
                    metrics["hw_cycles"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif "GOPS:" in line:
                try:
                    metrics["hw_gops"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif "Verification" in line and "PASSED" in line:
                metrics["verification_passed"] = True
            elif "Verification" in line and "FAILED" in line:
                metrics["verification_passed"] = False

    def benchmark_matrix_sizes(
        self, 
        sizes: List[int] = [32, 64, 96, 128], 
        data_format: str = "int16", 
        iterations: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run benchmarks across different matrix sizes."""
        logger.info(f"Running benchmark for matrix sizes: {sizes}")
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        for size in sizes:
            logger.info(f"Benchmarking {size}x{size} matrices...")
            size_results: List[Dict[str, Any]] = []
            
            for i in range(iterations):
                # Generate test matrices with reasonable values for int16 conversion
                A = np.random.randn(size, size).astype(np.float32) * 0.1
                B = np.random.randn(size, size).astype(np.float32) * 0.1
                
                # Compute reference
                start_ref = time.perf_counter()
                C_ref = np.matmul(A, B)
                end_ref = time.perf_counter()
                ref_time = end_ref - start_ref
                
                try:
                    # Hardware computation
                    C_hw, metrics = self.matrix_multiply(A, B, data_format, enable_trace=(i==0 and size<=64))
                    
                    # Convert hardware result back to float for comparison
                    if data_format == "int16":
                        # Account for scaling applied during conversion (1000x)
                        C_hw_float = C_hw.astype(np.float32) / (1000.0 * 1000.0)  # Double scaling for multiplication
                        error = np.mean(np.abs(C_hw_float - C_ref) / (np.abs(C_ref) + 1e-8))
                    else:
                        error = np.mean(np.abs(C_hw.astype(np.float32) - C_ref) / (np.abs(C_ref) + 1e-8))

                    metrics.update({
                        "matrix_size": size,
                        "iteration": i,
                        "reference_time": ref_time,
                        "relative_error": float(error),
                        "speedup": float(ref_time / max(metrics.get("python_time", 1e-9), 1e-9)),
                    })

                    size_results.append(metrics)
                    logger.info(f"  Iteration {i+1}: {metrics.get('hw_gops', 0):.2f} GOPS, "
                              f"error={error:.2e}, speedup={metrics.get('speedup', 0):.2f}x")
                              
                except Exception as e:
                    logger.error(f"Benchmark failed for size {size}, iteration {i}: {e}")
                    continue
                    
            results[f"{size}x{size}"] = size_results
        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all performed operations."""
        if not self.performance_history:
            return {}
            
        hw_cycles = [p.get("hw_cycles", 0) for p in self.performance_history if "hw_cycles" in p]
        hw_gops = [p.get("hw_gops", 0) for p in self.performance_history if "hw_gops" in p]
        python_times = [p.get("python_time", 0) for p in self.performance_history]
        speedups = [p.get("speedup", 0) for p in self.performance_history if "speedup" in p]
        
        summary = {
            "total_operations": len(self.performance_history),
            "avg_hw_cycles": float(np.mean(hw_cycles)) if hw_cycles else 0.0,
            "avg_hw_gops": float(np.mean(hw_gops)) if hw_gops else 0.0,
            "avg_python_time": float(np.mean(python_times)) if python_times else 0.0,
            "avg_speedup": float(np.mean(speedups)) if speedups else 0.0,
            "max_speedup": float(np.max(speedups)) if speedups else 0.0,
        }
        return summary


# Integration with RAG Pipeline
class AcceleratedRAG:
    """RAG pipeline with hardware-accelerated matrix operations."""

    def __init__(self, rag_pipeline, verilator_gemm: VerilatorGEMM):
        self.rag = rag_pipeline
        self.hw_gemm = verilator_gemm
        self.acceleration_stats = {
            "hw_operations": 0,
            "sw_operations": 0,
            "total_hw_time": 0.0,
            "total_sw_time": 0.0,
        }

    def accelerated_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Compute attention using hardware acceleration when beneficial."""
        start_time = time.perf_counter()
        
        # Check if we should try hardware acceleration
        if self._should_accelerate(query.shape, key.shape):
            try:
                # QK^T computation
                qk_scores, _ = self.hw_gemm.matrix_multiply(query, key.T, data_format="int16")
                
                # Convert back to float and apply softmax
                qk_float = qk_scores.astype(np.float32) / (1000.0 * 1000.0)  # Undo scaling
                attention_weights = self._softmax(qk_float)
                
                # Attention * Value computation
                if self._should_accelerate(attention_weights.shape, value.shape):
                    # Convert attention weights back to int16 for hardware
                    attn_int16 = np.clip(attention_weights * 1000, -32768, 32767).astype(np.int16)
                    output, _ = self.hw_gemm.matrix_multiply(attn_int16, value, data_format="int16")
                    result = output.astype(np.float32) / (1000.0 * 1000.0)  # Undo scaling
                else:
                    # Use software for second multiplication
                    result = np.matmul(attention_weights, value)
                
                self.acceleration_stats["hw_operations"] += 1
                self.acceleration_stats["total_hw_time"] += time.perf_counter() - start_time
                return result
                
            except Exception as e:
                logger.warning(f"Hardware acceleration failed, falling back to software: {e}")

        # Fallback to software implementation
        qk_scores = np.matmul(query, key.T)
        attention_weights = self._softmax(qk_scores)
        output = np.matmul(attention_weights, value)
        
        self.acceleration_stats["sw_operations"] += 1
        self.acceleration_stats["total_sw_time"] += time.perf_counter() - start_time
        return output

    def _should_accelerate(self, shape_a: Tuple[int, int], shape_b: Tuple[int, int]) -> bool:
        """Determine if operation should use hardware acceleration."""
        # Hardware is most efficient for medium-sized matrices
        min_size = 32
        max_size = 256
        
        m, k = shape_a
        k2, n = shape_b
        
        # Check dimension compatibility and size constraints  
        if k != k2:
            return False
            
        # All dimensions should be within reasonable range for hardware
        return all(min_size <= dim <= max_size for dim in [m, n, k])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_acceleration_summary(self) -> Dict[str, Any]:
        """Get summary of hardware vs software usage."""
        total_ops = self.acceleration_stats["hw_operations"] + self.acceleration_stats["sw_operations"]
        if total_ops == 0:
            return {"acceleration_ratio": 0, "speedup_estimate": 1.0}
            
        hw_ratio = self.acceleration_stats["hw_operations"] / total_ops
        avg_hw_time = self.acceleration_stats["total_hw_time"] / max(self.acceleration_stats["hw_operations"], 1)
        avg_sw_time = self.acceleration_stats["total_sw_time"] / max(self.acceleration_stats["sw_operations"], 1)
        speedup_estimate = avg_sw_time / avg_hw_time if avg_hw_time > 0 else 1.0
        
        return {
            "acceleration_ratio": hw_ratio,
            "hw_operations": self.acceleration_stats["hw_operations"],
            "sw_operations": self.acceleration_stats["sw_operations"],
            "avg_hw_time": avg_hw_time,
            "avg_sw_time": avg_sw_time,
            "speedup_estimate": speedup_estimate,
        }


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("=== Verilator GEMM Bridge Test ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")

    # Initialize hardware accelerator interface
    try:
        hw_gemm = VerilatorGEMM()
        print(f"✅ Successfully initialized hardware interface")
    except FileNotFoundError as e:
        logger.error(e)
        print("\n❌ Failed to initialize hardware interface")
        print("\nTroubleshooting steps:")
        print("1. Ensure you're in the project root directory")
        print("2. Build the simulation: cd sim && make")
        print("3. Check that the executable exists: ls -la sim/obj_dir/Vgemm_accelerator")
        print("4. Or run from project root: cd ../../ && python scripts/python/verilator_bridge.py")
        sys.exit(1)

    # Test basic matrix multiplication
    print("\nTesting basic matrix multiplication...")
    A = np.random.randn(64, 64).astype(np.float32) * 0.1
    B = np.random.randn(64, 64).astype(np.float32) * 0.1

    try:
        C_hw, metrics = hw_gemm.matrix_multiply(A, B, enable_trace=True)
        print(f"✅ Hardware result shape: {C_hw.shape}")
        print(f"✅ Hardware metrics: {metrics}")
        
        # Verify against reference
        C_ref = np.matmul(A, B)
        C_hw_float = C_hw.astype(np.float32) / (1000.0 * 1000.0)  # Convert back to float
        error = np.mean(np.abs(C_hw_float - C_ref))
        print(f"✅ Mean absolute error: {error:.6f}")
        
    except Exception as e:
        logger.error(f"❌ Matrix multiply failed: {e}")

    # Run a small benchmark
    print("\n=== Running Performance Benchmark ===")
    try:
        benchmark_results = hw_gemm.benchmark_matrix_sizes(sizes=[32, 64, 128], iterations=2)
        for size, results in benchmark_results.items():
            if results:
                avg_speedup = np.mean([r.get("speedup", 1.0) for r in results])
                avg_gops = np.mean([r.get("hw_gops", 0.0) for r in results])
                avg_error = np.mean([r.get("relative_error", 0.0) for r in results])
                print(f"✅ {size}: {avg_speedup:.2f}x speedup, {avg_gops:.2f} GOPS, {avg_error:.2e} error")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")

    print("\n=== Performance Summary ===")
    summary = hw_gemm.get_performance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n✅ Bridge test completed successfully!")