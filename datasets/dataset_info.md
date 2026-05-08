# Dataset Reference

## Overview

All datasets are publicly available on Kaggle. Raw data files are NOT stored in this repository. Download links and preprocessing notes are provided below.

---

## 1. Cloud Task Scheduling Dataset

- **Source:** https://www.kaggle.com/datasets/programmer3/cloud-task-scheduling-dataset
- **File:** `Distributed_Task_Scheduling.csv`
- **Rows:** 6,000
- **Priority column:** `Task_Priority` (1=Low, 2=Medium, 3=High)
- **Priority distribution:** High=1794 (29.9%), Medium=2381 (39.7%), Low=1825 (30.4%)
- **Spearman max|ρ|:** 0.0302 — ALL features decorrelated from priority label
- **Key note:** Priority is simulation-assigned, not derived from observable features.
  This dataset serves as a **negative control** in E1.
- **Features used (19):**
  Task_Length_MIPS, Task_Deadline, Data_Upload_Size_MB, Data_Download_Size_MB,
  VM_MIPS, VM_Memory_GB, VM_Bandwidth_MBps, Execution_Time_S, Waiting_Time_S,
  Completion_Time_S, Energy_Consumption_J, Makespan_S, Response_Time_S,
  Execution_Cost_$, Degree_of_Imbalance, Storage_Utilization, Path_Load,
  resource_type_enc, sched_algo_enc

---

## 2. Google Cluster Traces 2019

- **Source:** https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample
- **File:** `borg_traces_data.csv`
- **Rows:** 405,894
- **Priority column:** `priority` (binned: <100=Low, 100-199=Medium, ≥200=High)
- **Priority distribution:** High=156,263 (38.5%), Medium=165,109 (40.7%), Low=84,522 (20.8%)
- **Spearman max|ρ|:** 0.8450 (`scheduler`) — near-perfect correlation
- **Key note:** `scheduler` and `scheduling_class` are Borg policy fields that
  near-bijectively encode priority. Near-deterministic ceiling dataset.
- **Parsed columns:** resource_request→{cpus,memory}, average_usage→{cpus,memory},
  maximum_usage→{cpus,memory}
- **Features used (18):**
  scheduling_class, collection_type, instance_index, assigned_memory,
  page_cache_memory, cycles_per_instruction, memory_accesses_per_instruction,
  sample_rate, scheduler, vertical_scaling, req_cpus, req_memory,
  avg_cpus, avg_memory, max_cpus, max_memory, failed, event_enc

---

## 3. IT Incident Log Dataset

- **Source:** https://www.kaggle.com/datasets/shamiulislamshifat/it-incident-log-dataset
- **File:** `incident_event_log.csv`
- **Rows:** 141,712 raw → **24,918 after deduplication** (last record per incident number)
- **Priority mapping:**
  - 1-Critical → High, 2-High → High
  - 3-Moderate → Medium
  - 4-Low → Low
- **Priority distribution:** Medium=23,466 (94.2%), Low=774 (3.1%), High=678 (2.7%)
- **Imbalance ratio:** 34.6:1 (Medium:High)
- **Spearman max|ρ|:** 0.2217 (`impact_enc`)
- **Key note:** `impact` + `urgency` together encode priority (ITIL standard).
  Extreme class imbalance — SMOTE critical here.
- **Features used (11):**
  reassignment_count, reopen_count, sys_mod_count, impact_enc, urgency_enc,
  category_enc, location_enc, contact_enc, made_sla_enc, knowledge_enc, reopen_flag

---

## 4. Multi-Cloud Service Composition Dataset

- **Source:** https://www.kaggle.com/datasets/ziya07/multi-cloud-service-composition-dataset
- **File:** `multi_cloud_service_dataset.csv`
- **Rows:** 1,000
- **Priority derivation:** `QoS_Score` tertile-cut → Low/Medium/High (balanced)
- **Priority distribution:** Balanced ~333 each
- **Spearman max|ρ|:** 0.4707 (`QoS_Score`) — non-linear boundary
- **Key note:** Non-linear QoS threshold creates decision boundaries that require
  ensemble methods. LogReg fails (κ=0.300) while KATS succeeds (κ=0.997).
- **Features used (14):**
  CPU_Utilization (%), Memory_Usage (MB), Storage_Usage (GB),
  Network_Bandwidth (Mbps), Service_Latency (ms), Response_Time (ms),
  Throughput (Requests/sec), Load_Balancing (%), QoS_Score,
  Workload_Variability, Optimal_Service_Placement,
  service_type_enc, cloud_provider_enc, edge_node_enc
