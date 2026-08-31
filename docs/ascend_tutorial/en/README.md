# Ascend Usage Tutorial
## Introduction

Ascend provides full support for using and developing verl. This document provides a comprehensive introduction to using verl on Huawei Ascend NPU chips.

Last updated: 05/14/2026.

## Directory Structure

zh/
├── get_start/                     # Quick start guide
├── feature_support/               # Feature support description
├── model_support/                 # Model support description
├── dev_guide/                     # Development guide
├── faq/                           # Frequently asked questions
└── contribution_guide/            # Community contribution guide
```
## Latest News
- [verl-ascend-recipe repository created](https://github.com/verl-project/verl-ascend-recipe) - New Ascend recipe added
- [verl on Ascend 2026Q2 roadmap](https://github.com/verl-project/verl/issues/5526) - 2026Q2 roadmap released

## Quick Start
- [Ascend Image Guide](./get_start/dockerfile_build_guidance.rst) - Build and use a Docker image for the Ascend environment
- [Ascend Installation Guide](./get_start/install_guidance.rst) - Customize the verl installation on Ascend NPUs
- [Ascend Quick Start Guide](./get_start/quick_start.rst) - Quickly get started running verl on Ascend NPUs

## Feature Support Notes

- [Training Configuration Parameters and Metrics](./dev_guide/model_dev/parameter_and_metrics.md) - Supported verl framework features/parameter list
- [NPU Advanced Features Guide](./feature_support/npu_advance_features.md) - Common NPU-related features/environment variable descriptions

## Model Support Notes

- [NPU Model and Algorithm Support](./model_support/model_and_algorithm_support.md) - List of supported models and algorithms
- [Best Practice Examples](./model_support/examples) - Best practices and model deployment examples


## Development Guide

- [Model Development](./dev_guide/model_dev)
    - [Model Migration to NPU Guide](./dev_guide/model_dev/transfer_to_npu_guide.md) - Model migration guide
    - [Training Configuration Parameters and Metrics Description](./dev_guide/model_dev/parameter_and_metrics.md) - Training parameters and metrics
    - [Model Evaluation](./dev_guide/model_dev/evaluation.md) - Model evaluation guide
- [Precision Debugging](./dev_guide/precision_analysis)
    - [Precision Alignment Guide](./dev_guide/precision_analysis/precision_alignment.md) - Precision alignment guide
    - [Precision Debugger](../en/dev_guide/precision_analysis/precision_debugger.md) - Tool for troubleshooting precision issues
- [Performance Tuning](./dev_guide/performance)
    - [Ascend Performance Analysis Guide](./dev_guide/performance/ascend_performance_analysis_guide.md) - Performance analysis guide
    - [Ascend Performance Tuning Guide](./dev_guide/performance/perf_tuning_on_ascend.rst) - Performance tuning guide
    - [Profiling Collection Guide](./dev_guide/performance/ascend_profiling.rst) - Guide to using the profiling tool


## Support and Feedback

If you encounter any issues while using this guide, you can get help through the following channels:

1. See [NPU FAQ](./faq/faq.rst)
2. Submit an issue in GitHub Issues
3. Contact Ascend technical support

## Contribution Guide
- [verl Community Contributions](../../contributing) - Guide to contributing to the verl community
- [NPU-CI Addition Guide](./contribution_guide/ascend_ci_guide.rst) - CI configuration and testing for Ascend environments

## Related Resources

- [verl official documentation](https://verl.readthedocs.io/)
- [Ascend Developer Community](https://www.hiascend.com/)
