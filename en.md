# Code Development Standard (STD)
Standard specifications for code development, logging, execution, and output management.

---

## 1. General Code Principles
+ All source code, docstrings, and log messages **must use English and ASCII characters only**.
+ Use Chinese exclusively when explaining concepts or instructions directly to end users.
+ Do not reinvent the wheel. Reuse mature, production-ready implementations whenever possible.
    - Example: For custom data sampling in model training, build upon PyTorch's `DataLoader` instead of implementing data pipelines from scratch, then extend and customize on top of it.
+ Follow the **minimal design principle**: implement only explicitly required features. Do not add any extra functionality not requested by the user.
+ Do not modify any third-party packages or external source code unless explicitly requested by the user. If modification is required, always inform the user of associated risks first.
+ All code must be ASCII-only. Variable names, function names, and inline comments must not contain Chinese or other non-ASCII special characters.
+ All implicit design decisions must be clearly documented, especially those related to algorithm logic and parameter configurations.

---

## 2. Logging Standards
### 2.1 Logger Configuration
+ Reuse global logger settings across all scripts. Do not create redundant logger instances in individual scripts.
+ **Single logger rule**: one logger per file or per CLI entry point. Do not instantiate multiple loggers in one process or script unless absolutely necessary.

### 2.2 Log Format
+ All log output must follow the unified format:

```plain
<timestamp>-[script-name]-[stage]-[content]
```

### 2.3 Mandatory Logging Behavior
+ **Stage boundary logging**: record the start and end of every execution stage. For example, log status when initializing model weights or transferring tensors from CPU to GPU.
+ **Heartbeat logging**: print the current execution stage to the console at a configurable frequency to confirm the process is alive. For example, output status every 10 or 30 seconds.
+ **Progress bar with ETA**: for computationally heavy tasks (data loading, format conversion, feature computation, model training, evaluation, etc.), implement a progress bar with ETA if implementation overhead is low.
    - TQDM is optional, not mandatory.
    - ETA, file size, and similar metrics must use human-readable units (e.g., `1h` instead of `3600s`, `1 GiB` instead of raw byte counts).

### 2.4 Execution Summary
+ **Pre-execution**: print key configuration parameters to the console.
+ **Post-execution**: print total execution time and output directory path to the console.

---

## 3. Runtime & Execution Rules
### 3.1 Environment
+ All scripts must run inside a Conda environment by default. Confirm the environment name with the user if not specified.
+ Buffered output or output redirection will suppress real-time console logs, which may cause users to doubt whether the script is running properly. Avoid such setups unless absolutely necessary.
+ Enable unbuffered stdout/stderr flush to ensure real-time log output.

### 3.2 Long-Running Tasks
+ For long-running heavy tasks, especially on remote hosts, use `tmux` sessions to avoid interruptions caused by SSH connection drops.

### 3.3 Completion Marker
+ Create a `.complete` lock file upon successful task completion.

---

## 4. Path & Output Management
+ Maintain a global path specification utility script:
    - Use annotations to define standard storage paths for different types of artifacts.
    - Provide lightweight helper modules to enforce output directory constraints for all other file-generating scripts.
+ All scripts that accept output directory arguments must comply with the global path specification **at write time**, rather than validating paths after files are created.

---

## 5. Process Lifecycle Management
+ For scripts that spawn multiple child subprocesses, implement graceful shutdown logic:
    - When the main process receives a termination signal, it must first send an interrupt signal to all subprocesses.
    - The main process must wait for all subprocesses to fully exit before terminating itself.

---

## 6. Data Validation Rules
+ Do not use hash-level integrity verification by default. Only design and validate at the field level.
+ Hash operations are permitted only when explicitly requested by the user, or for mandatory use cases such as data deduplication.

---

## 7. Documentation Requirements
+ All documents must include the **last modified date** by default.

