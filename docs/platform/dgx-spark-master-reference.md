# DGX Spark Master Reference

> **Last Updated**: 2026-03-30 (Skeleton — Phase 0)
> **Status**: Awaiting Phase 1 content

## Summary
<!-- 2-5 sentence overview of DGX Spark platform -->

## 1. Hardware Architecture
### 1.1 GB10 Grace Blackwell Superchip
### 1.2 CPU — 20-Core ARM (Cortex-X925 + Cortex-A725)
### 1.3 GPU — Blackwell Architecture (sm_121)
### 1.4 Unified Memory — 128GB LPDDR5x
### 1.5 NVLink C2C Interconnect

## 2. Storage
### 2.1 4TB NVMe M.2 — Specifications
### 2.2 Self-Encryption
### 2.3 Performance Characteristics
### 2.4 Recommended Storage Layout

## 3. Networking
### 3.1 10 GbE Ethernet (RJ-45)
### 3.2 ConnectX-7 (200Gbps QSFP)
### 3.3 Wi-Fi 7
### 3.4 Bluetooth 5.4
### 3.5 Dual-Spark QSFP Configuration

## 4. I/O Ports
### 4.1 USB-C (4×)
### 4.2 HDMI 2.1a
### 4.3 Known Issues (HDMI Deep Sleep Bug)

## 5. Physical & Thermal
### 5.1 Form Factor (150×150×50.5mm, 1.2kg)
### 5.2 Power (240W PSU, 140W TDP)
### 5.3 Acoustic Profile (35 dB(A))
### 5.4 Thermal Throttling Reality
### 5.5 Cooling Requirements & Recommendations
### 5.6 Firmware Patch Status

## 6. Software Stack
### 6.1 DGX OS 7.4.0 (Ubuntu 24.04, Kernel 6.17)
### 6.2 CUDA 13.0.2
### 6.3 GPU Driver 580.126.09
### 6.4 Docker + NVIDIA Container Toolkit
### 6.5 Ollama (Pre-installed)
### 6.6 JupyterLab (via DGX Dashboard)
### 6.7 DGX Dashboard
### 6.8 NVIDIA Sync
### 6.9 NVIDIA NIM

## 7. Bundled Benefits
### 7.1 90-Day AI Enterprise License
### 7.2 DLI Course
### 7.3 NGC Catalog Access
### 7.4 DGX Cloud Migration Path
### 7.5 Developer Forum & Support

## 8. CUDA / cuDNN / TensorRT Stack
### 8.1 CUDA 13.0.2 Details
### 8.2 cuDNN 9.13
### 8.3 TensorRT Installation
### 8.4 sm_121 Implications
### 8.5 CUDA 13 Wheel Gap Analysis

## 9. Container Runtime
### 9.1 Docker Configuration
### 9.2 NVIDIA Container Toolkit Setup
### 9.3 NGC Catalog Access
### 9.4 Docker Override Configuration
### 9.5 Multi-Arch Image Migration

## 10. Limitations & Gotchas
### 10.1 Thermal Throttling
### 10.2 OOM Zombie Mode
### 10.3 No ECC Memory
### 10.4 HDMI Deep Sleep Bug
### 10.5 CUDA 13 Wheel Gap
### 10.6 nvidia-smi Memory Reporting
### 10.7 Clustering Cap (2 Devices)
### 10.8 Setup Non-Resumable
### 10.9 Memory Bandwidth Bottleneck
### 10.10 FP4 Performance Claims

## 11. Best Practices
### 11.1 Cooling & Airflow Management
### 11.2 Docker OOM Score Adjustment
### 11.3 Container-First Workflow
### 11.4 Environment Isolation Strategy
### 11.5 Storage Layout Conventions
### 11.6 Update Cadence
### 11.7 Model Cache Management

## Detailed Findings
<!-- Filled in Phase 1 -->

## Practical Implications
<!-- Filled in Phase 1 -->

## Recommended Actions
<!-- Filled in Phase 1 -->

## Confidence Level
<!-- HIGH / MEDIUM / LOW with reasoning -->

## Source Notes
<!-- Tagged [NVIDIA-OFFICIAL], [COMMUNITY], [TESTED], [INFERRED] -->

## Unresolved Questions
<!-- What we still don't know -->
