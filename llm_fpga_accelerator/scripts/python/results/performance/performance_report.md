# Hardware Accelerator Performance Report

## Executive Summary

- **Peak Speedup**: 5.85x over CPU
- **Peak Throughput**: 25.70 GOPS
- **Best Matrix Size**: 128×128
- **Average Hardware Utilization**: 19.2%
- **Hardware Efficiency**: 92.3%
- **Average Numerical Error**: 0.0975%

## Detailed Analysis

**64x64 Matrix**:
  - Speedup: 3.15x
  - Throughput: 12.70 GOPS
  - Cycles: 8050
  - Utilization: 12.7%
  - Numerical Error: 0.1100%

**128x128 Matrix**:
  - Speedup: 5.85x
  - Throughput: 25.70 GOPS
  - Cycles: 31900
  - Utilization: 25.7%
  - Numerical Error: 0.0850%

### Scaling Behavior

- **Scaling Exponent**: 1.01
- **Interpretation**: Sublinear scaling (bandwidth limited)

## Recommendations

- **Low utilization detected**: Consider optimizing data flow or increasing parallelism
- **Optimal operating point**: Use 128×128 matrices for best performance
