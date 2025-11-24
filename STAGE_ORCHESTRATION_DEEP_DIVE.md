# Stage Orchestration Loop - Deep Dive

## Overview

The Stage Orchestration Loop is the **heart of the execution phase**. It's a custom control flow implemented in `StageOrchestratorAgent` that manages the progressive, adaptive implementation of a multi-stage analysis plan.

**Location**: `src/agentic_data_scientist/agents/adk/stage_orchestrator.py:111-511`

---

## The Core Loop Structure

```python
# Simplified pseudocode
while iteration < max_iterations:
    # 1. CHECK EXIT CONDITION
    if all(criterion["met"] for criterion in criteria):
        return  # Success! All criteria met

    # 2. GET NEXT STAGE
    next_stage = [s for s in stages if not s["completed"]][0]

    # 3. IMPLEMENT STAGE
    run_implementation_loop(next_stage)
    compress_events()  # Manual compression

    # 4. CHECK CRITERIA
    run_criteria_checker()

    # 5. REFLECT & ADAPT
    run_stage_reflector()

    # 6. MARK COMPLETE
    next_stage["completed"] = True
```

**Key insight**: This is NOT a simple for-loop over stages. It's an **adaptive control flow** that:
- Can modify remaining stages mid-execution
- Can add new stages if needed
- Exits when criteria are met (not when stages are done)
- Stages are a means to an end, criteria are the actual goal

---

## Iteration 0: Initialization and Validation

Before the loop starts, the orchestrator performs extensive validation:

```python
# Line 136-143: Initialize state
if "current_stage" not in state:
    state["current_stage"] = None
if "current_stage_index" not in state:
    state["current_stage_index"] = 0
if "stage_implementations" not in state:
    state["stage_implementations"] = []

# Line 144-227: Validate stages and criteria
stages: List[Dict] = state.get("high_level_stages", [])
criteria: List[Dict] = state.get("high_level_success_criteria", [])

if not stages or len(stages) == 0:
    logger.error("[StageOrchestrator] No stages found in state!")
    yield error_event
    return

# Validate structure (check first 3 stages)
for i in range(min(3, len(stages))):
    stage = stages[i]
    if not isinstance(stage, dict) or "index" not in stage or "title" not in stage:
        logger.error(f"Invalid stage structure at index {i}")
        yield error_event
        return
```

**Why validate only first 3?** Because stages can be added dynamically by the reflector. We only validate the initial structure.

**Initial state logging**:
```python
logger.info(f"[StageOrchestrator] Starting orchestration with {len(stages)} stages")
logger.info("[StageOrchestrator] Success Criteria (End-State Goals):")
logger.info(format_criteria_status(criteria))
logger.info("Note: Early 'NOT MET' status is expected and normal.")
```

Example output:
```
[StageOrchestrator] Starting orchestration with 5 stages
[StageOrchestrator] Success Criteria (End-State Goals):
  [❌ NOT MET] Criterion 0: Dataset loaded and validated with quality checks
  [❌ NOT MET] Criterion 1: Model accuracy > 80%
  [❌ NOT MET] Criterion 2: Feature importance analysis completed
  [❌ NOT MET] Criterion 3: Final report with actionable recommendations
```

---

## Step 1: Check Exit Condition (Lines 248-278)

At the start of **every iteration**, check if we're done:

```python
# Refresh state (may have been modified by callbacks)
stages = state.get("high_level_stages", [])
criteria = state.get("high_level_success_criteria", [])

# Count how many criteria are met
criteria_met_count = sum(1 for c in criteria if c.get("met", False))
logger.info(f"[StageOrchestrator] Criteria status: {criteria_met_count}/{len(criteria)} met")

# Exit condition: ALL criteria met
if all(c.get("met", False) for c in criteria):
    logger.info("[StageOrchestrator] 🎉 All success criteria met! Exiting to summary.")

    completion_event = Event(
        author=self.name,
        content=types.Content(
            role="model",
            parts=[types.Part(text=f"\n\n✅ All {len(criteria)} high-level success criteria have been met...")]
        ),
        turn_complete=True,
    )
    yield completion_event
    return  # Exit orchestrator - workflow proceeds to summary agent
```

**Critical design choice**: Exit is based on **criteria**, not stages. You could have:
- 5 stages, 3 criteria → might exit after stage 3 if all criteria met
- 3 stages, 5 criteria → reflector might add stages 4-6 to meet remaining criteria

**Example scenario where this matters**:

```python
# Initial plan has 3 stages
stages = [
    {"index": 0, "title": "Load data", "completed": False},
    {"index": 1, "title": "Train model", "completed": False},
    {"index": 2, "title": "Generate report", "completed": False},
]

# But 4 criteria
criteria = [
    {"criteria": "Data loaded", "met": False},
    {"criteria": "Model accuracy > 80%", "met": False},
    {"criteria": "Feature importance analyzed", "met": False},  # ← Not in original plan!
    {"criteria": "Report created", "met": False},
]

# After stage 2 completes:
# - criteria[0] = met (data loaded)
# - criteria[1] = met (model trained)
# - criteria[2] = NOT MET (feature importance not in plan!)
# - criteria[3] = NOT MET (report not yet generated)

# Reflector adds new stage:
stages.append({
    "index": 3,
    "title": "Feature Importance Analysis",
    "description": "Analyze feature importance to meet criterion 2",
    "completed": False,
})

# Loop continues with 4 stages now
```

---

## Step 2: Get Next Uncompleted Stage (Lines 281-318)

```python
# Get all stages that haven't been completed yet
remaining_stages = [s for s in stages if not s.get("completed", False)]

if not remaining_stages:
    logger.warning("No remaining stages but criteria not met. Asking reflector to extend stages.")

    # Run reflector WITHOUT implementing a stage first
    async for event in self.stage_reflector.run_async(ctx):
        yield event

    # Refresh stages from state (reflector may have added stages)
    stages = state.get("high_level_stages", [])
    remaining_stages = [s for s in stages if not s.get("completed", False)]

    if not remaining_stages:
        # Reflector couldn't add stages - we're stuck
        logger.error("Still no stages after reflection. Exiting despite incomplete criteria.")
        yield warning_event
        return

# Get the FIRST uncompleted stage (sequential execution)
next_stage = remaining_stages[0]
stage_idx = next_stage["index"]
```

**Important**: Stages are executed **sequentially** (not in parallel). This is because:
1. Later stages often depend on earlier ones (e.g., "Train model" depends on "Load data")
2. Reflector needs to see results before adapting plan
3. Criteria checking is cumulative (earlier stages can meet criteria too)

**Edge case handling**: If we run out of stages but criteria aren't met, the reflector is called **proactively** to extend the plan. If it still can't add stages, the orchestrator exits with a warning.

**Stage start event**:
```python
stage_start_event = Event(
    author=self.name,
    content=types.Content(
        role="model",
        parts=[types.Part(text=f"\n\n### Stage {stage_idx + 1}: {next_stage['title']}\n\n"
                               f"{next_stage['description']}\n\n"
                               "Beginning implementation...\n\n")]
    ),
    partial=False,
)
yield stage_start_event
```

This creates a clear section in the output stream.

---

## Step 3: Implement Stage (Lines 340-395)

### 3.1 Set Current Stage Context

```python
# Set current stage in state (for implementation loop to read)
state["current_stage"] = {
    "index": next_stage["index"],
    "title": next_stage["title"],
    "description": next_stage["description"],
}

# Clear previous implementation outputs
state.pop("implementation_summary", None)
state.pop("review_feedback", None)
```

**Why clear previous outputs?** To prevent state pollution. If the coding agent fails and doesn't write `implementation_summary`, we don't want the review agent reading the summary from a previous stage.

### 3.2 Run Implementation Loop

```python
logger.info(f"[StageOrchestrator] Running implementation_loop for stage {stage_idx}")

try:
    async for event in self.implementation_loop.run_async(ctx):
        yield event  # Stream events to user in real-time

    logger.info(f"[StageOrchestrator] Completed implementation_loop for stage {stage_idx}")
```

**What happens inside the implementation loop?**

```
NonEscalatingLoopAgent (max 5 iterations)
├─> Coding Agent (ClaudeCodeAgent)
│   - Reads: state["current_stage"]
│   - Implements the stage using Claude Code
│   - Writes: state["implementation_summary"]
│
├─> Review Agent (LoopDetectionAgent)
│   - Reads: state["implementation_summary"]
│   - Reviews code quality, correctness
│   - Writes: state["review_feedback"]
│
└─> Implementation Review Confirmation (LoopDetectionAgent)
    - Reads: state["review_feedback"]
    - Decides: {"exit": true/false, "reason": "..."}
    - If exit=true: sets escalate flag → exits loop
    - If exit=false: loops back to Coding Agent
```

**Example iteration**:

Iteration 1:
```
Coding Agent: "I'll load the data using pandas..."
               Creates: workflow/01_load_data.py
               state["implementation_summary"] = "Loaded data from data.csv, 50k rows..."

Review Agent: "Code looks good but missing error handling for missing columns"
              state["review_feedback"] = "Add try/except for missing columns..."

Review Confirmation: {"exit": false, "reason": "Missing error handling"}
                     → Continue loop
```

Iteration 2:
```
Coding Agent: "I'll add error handling..."
               Updates: workflow/01_load_data.py
               state["implementation_summary"] = "Added error handling, validates columns..."

Review Agent: "Implementation meets requirements, code is clean"
              state["review_feedback"] = "Approved - all requirements met"

Review Confirmation: {"exit": true, "reason": "Implementation approved"}
                     → Exit loop
```

### 3.3 Manual Event Compression

```python
# === Manual Event Compression After Implementation Loop ===
logger.info("[StageOrchestrator] Running manual event compression after implementation loop")

try:
    await compress_events_manually(
        ctx=ctx,
        event_threshold=40,
        overlap_size=20,
    )
except Exception as compress_err:
    logger.warning(f"[StageOrchestrator] Manual compression failed: {compress_err}")
```

**Why manual compression here?**

The implementation loop can generate 100+ events:
- Coding agent: text output, tool calls (Read, Write, Bash), tool responses
- Review agent: text output, tool calls (Read files for review)
- Multiple iterations

Without compression, events would accumulate rapidly and hit context limits. By compressing **immediately after implementation**, we:
1. Summarize the detailed implementation work
2. Keep only the essential information (what was implemented, key decisions)
3. Free up context for criteria checking and reflection

**What gets compressed**:
```
Events 0-40:  [COMPRESSED] Implementation loop for stage 0 involved creating
              workflow/01_load_data.py with pandas data loading, error handling
              for missing columns, and validation checks. Review feedback led
              to two iterations with improvements to error messages...

Events 41-60: Recent events (kept uncompressed)
```

### 3.4 Store Implementation Result

```python
# Store implementation result in stage object
next_stage["implementation_result"] = state.get("implementation_summary", "")

# Add to completed stages history
stage_implementations = state.get("stage_implementations", [])
stage_implementations.append({
    "stage_index": next_stage["index"],
    "stage_title": next_stage["title"],
    "implementation_summary": next_stage["implementation_result"],
})
state["stage_implementations"] = stage_implementations
```

**Why store in TWO places?**

1. `next_stage["implementation_result"]` - Attached to the stage object for reference
2. `state["stage_implementations"]` - Chronological history for criteria checker and reflector

The history array is critical for downstream agents:

```python
# Criteria checker uses this to understand what's been accomplished
stage_implementations = [
    {"stage_index": 0, "stage_title": "Load data", "implementation_summary": "..."},
    {"stage_index": 1, "stage_title": "Train model", "implementation_summary": "..."},
]

# Reflector uses this to decide if plan needs adaptation
```

### 3.5 Error Handling

```python
except Exception as e:
    logger.error(f"Implementation loop failed for stage {stage_idx}: {e}", exc_info=True)

    error_event = Event(
        author=self.name,
        content=types.Content(
            role="model",
            parts=[types.Part(text=f"\n\n❌ Implementation loop failed for stage {stage_idx} "
                                   f"({next_stage['title']}): {str(e)}\n\n"
                                   "Skipping to next stage...\n\n")]
        ),
        turn_complete=True,
    )
    yield error_event
    # Skip this stage and continue to next
    continue
```

**Critical design choice**: Errors **skip the stage** rather than failing the entire workflow. This allows partial progress to be made even if one stage fails.

The stage is NOT marked as completed, so:
- It remains in the stage list
- Reflector can see it failed
- Reflector might modify it or suggest a workaround

---

## Step 4: Check Success Criteria (Lines 412-448)

```python
logger.info(f"[StageOrchestrator] Running criteria_checker after stage {stage_idx}")

try:
    async for event in self.criteria_checker.run_async(ctx):
        yield event

    # Criteria checker updates state["high_level_success_criteria"] via callback
    criteria = state.get("high_level_success_criteria", [])

    criteria_met_count = sum(1 for c in criteria if c.get("met", False))
    logger.info(f"[StageOrchestrator] Criteria status after check: {criteria_met_count}/{len(criteria)} met")
```

**What the criteria checker does**:

1. **Reads context from state**:
   ```python
   # From criteria_checker.md prompt
   **Original User Request:** {original_user_input?}
   **Success Criteria to Check:** {high_level_success_criteria?}
   **Completed Stage Implementations:** {stage_implementations?}
   ```

2. **Uses tools to inspect working directory**:
   ```python
   # Criteria checker has access to:
   tools = [
       read_file,           # Read file contents
       list_directory,      # See what files exist
       directory_tree,      # Recursive directory view
       search_files,        # Find files by pattern
       get_file_info,       # File metadata (size, modified time)
   ]
   ```

3. **Produces structured output**:
   ```python
   # Output schema: CriteriaCheckerOutput
   {
       "criteria_updates": [
           {
               "index": 0,
               "met": true,
               "evidence": "Dataset loaded in data/processed/customer_data.csv with 50,000 rows. Validation checks passed in workflow/01_data_loading.py."
           },
           {
               "index": 1,
               "met": false,
               "evidence": "Model accuracy is 76.5% according to results/model_metrics.json, which is below the 80% threshold required."
           }
       ]
   }
   ```

4. **Callback updates state**:
   ```python
   def criteria_checker_callback(callback_context: CallbackContext):
       for update in updates:
           idx = update["index"]
           criteria[idx]["met"] = update["met"]
           criteria[idx]["evidence"] = update["evidence"]

           logger.info(f"[CriteriaChecker] Criterion {idx}: {'✅ MET' if update['met'] else '❌ NOT MET'}")
           logger.info(f"  └─ Evidence: {update['evidence']}")
   ```

**Example execution**:

Before checking:
```python
criteria = [
    {"index": 0, "criteria": "Data loaded", "met": False, "evidence": None},
    {"index": 1, "criteria": "Model accuracy > 80%", "met": False, "evidence": None},
]
```

Criteria checker inspects:
```python
# Uses read_file to check workflow/01_load_data.py
# Uses read_file to check results/model_metrics.json
# Uses list_directory to verify files exist
```

After checking:
```python
criteria = [
    {
        "index": 0,
        "criteria": "Data loaded",
        "met": True,  # ← Changed!
        "evidence": "workflow/01_load_data.py successfully loads data.csv, validated 50k rows"
    },
    {
        "index": 1,
        "criteria": "Model accuracy > 80%",
        "met": False,  # ← Still false
        "evidence": "results/model_metrics.json shows 76.5% accuracy, below threshold"
    },
]
```

**Progressive fulfillment**: Criteria transition from `met=False` to `met=True` as stages complete. Once true, they generally stay true (but can be marked false if evidence shows regression).

---

## Step 5: Reflect and Adapt Plan (Lines 450-483)

```python
logger.info(f"[StageOrchestrator] Running stage_reflector after stage {stage_idx}")

try:
    async for event in self.stage_reflector.run_async(ctx):
        yield event

    # Reflector may modify state["high_level_stages"] via callback
    stages = state.get("high_level_stages", [])
```

**What the stage reflector does**:

1. **Analyzes progress**:
   ```python
   # From stage_reflector.md prompt
   **Original User Request:** {original_user_input?}
   **Current Stages (with completion status):** {high_level_stages?}
   **Success Criteria (with current met status):** {high_level_success_criteria?}
   **What's Been Implemented So Far:** {stage_implementations?}
   ```

2. **Inspects working directory** (has same tools as criteria checker)

3. **Decides adaptations**:
   ```python
   # Output schema: StageReflectorOutput
   {
       "stage_modifications": [
           {
               "index": 3,
               "new_description": "Perform hyperparameter tuning to improve model accuracy above 80% threshold. Use GridSearchCV with cross-validation."
           }
       ],
       "new_stages": [
           {
               "title": "Feature Importance Analysis",
               "description": "Analyze feature importance using SHAP values to meet criterion 2 which was not addressed in original plan."
           }
       ]
   }
   ```

4. **Callback modifies plan**:
   ```python
   def stage_reflector_callback(callback_context: CallbackContext):
       # Apply modifications to existing uncompleted stages
       for mod in modifications:
           idx = mod["index"]
           if not stages[idx].get("completed", False):  # ← Only modify uncompleted!
               stages[idx]["description"] = mod["new_description"]
               logger.info(f"[StageReflector] Modified stage {idx} description")

       # Add new stages to the end
       for new_stage in new_stages:
           new_idx = len(stages)
           stages.append({
               "index": new_idx,
               "title": new_stage["title"],
               "description": new_stage["description"],
               "completed": False,
               "implementation_result": None,
           })
           logger.info(f"[StageReflector] Added new stage {new_idx}: {new_stage['title']}")

       state["high_level_stages"] = stages
   ```

**Example adaptation scenario**:

**Before reflection** (after stage 1 completes):
```python
stages = [
    {"index": 0, "title": "Load data", "completed": True},
    {"index": 1, "title": "Train model", "completed": True},
    {"index": 2, "title": "Generate report", "completed": False},
]

criteria = [
    {"criteria": "Data loaded", "met": True},
    {"criteria": "Model accuracy > 80%", "met": False},  # ← Problem: accuracy only 76.5%
    {"criteria": "Report created", "met": False},
]
```

**Reflector reasoning**:
```
The model accuracy is 76.5% but criterion requires 80%. Stage 2 "Generate report"
won't address this. Need to add hyperparameter tuning before reporting.

Also, I notice criterion "Feature importance analyzed" but no stage addresses it.
Need to add that too.
```

**After reflection**:
```python
stages = [
    {"index": 0, "title": "Load data", "completed": True},
    {"index": 1, "title": "Train model", "completed": True},
    {"index": 2, "title": "Generate report", "completed": False, "description": "Updated to include model performance analysis..."},  # ← Modified
    {"index": 3, "title": "Hyperparameter Tuning", "completed": False},  # ← New!
    {"index": 4, "title": "Feature Importance Analysis", "completed": False},  # ← New!
]
```

**Next iteration**: The orchestrator will implement stage 2, but now with updated description. After that, stages 3 and 4 will be implemented.

---

## Step 6: Mark Stage Complete (Lines 485-494)

```python
# NOW mark stage as completed (after criteria check and reflection)
next_stage["completed"] = True

# Update stages in state
state["high_level_stages"] = stages

logger.info(f"[StageOrchestrator] Stage {stage_idx} cycle complete. Continuing to next iteration.")

# Update current_stage_index for tracking
state["current_stage_index"] = stage_idx
```

**Why mark complete AFTER reflection?**

Because the reflector needs to see:
1. The stage that just finished (`completed=False` initially)
2. What it accomplished
3. Whether it actually moved the needle on criteria

If we marked it complete before reflection, the reflector would see:
```python
stages = [
    {"index": 0, "completed": True},
    {"index": 1, "completed": True},  # ← Just finished
    {"index": 2, "completed": False},
]
```

But wouldn't know that stage 1 JUST completed (vs. completed earlier). The current design lets the reflector see the transition.

---

## Complete Iteration Example

Let's trace through a complete iteration:

### Initial State (Iteration 1)
```python
stages = [
    {"index": 0, "title": "Load data", "completed": True},
    {"index": 1, "title": "Train model", "completed": False},
    {"index": 2, "title": "Generate report", "completed": False},
]

criteria = [
    {"criteria": "Data loaded and validated", "met": True, "evidence": "..."},
    {"criteria": "Model accuracy > 80%", "met": False, "evidence": None},
    {"criteria": "Feature importance analyzed", "met": False, "evidence": None},
    {"criteria": "Report with recommendations", "met": False, "evidence": None},
]
```

### Step 1: Check Exit
```python
all(c["met"] for c in criteria) == False  # Only 1/4 criteria met
# Continue...
```

### Step 2: Get Next Stage
```python
remaining_stages = [stages[1], stages[2]]
next_stage = stages[1]  # "Train model"
```

### Step 3: Implement Stage
```python
# Implementation loop runs (5 iterations max)
# Iteration 1: Code model training
# Iteration 2: Review finds issues, suggests improvements
# Iteration 3: Code improved
# Iteration 4: Review approves
# Result: state["implementation_summary"] = "Trained RandomForest with 100 trees, accuracy 76.5%"

# Manual compression reduces 80 events to 1 summary event

# Store result
stages[1]["implementation_result"] = "Trained RandomForest..."
state["stage_implementations"].append({
    "stage_index": 1,
    "stage_title": "Train model",
    "implementation_summary": "Trained RandomForest...",
})
```

### Step 4: Check Criteria
```python
# Criteria checker inspects:
# - Reads results/model_metrics.json
# - Sees accuracy = 76.5%

# Updates criteria
criteria[1]["met"] = False  # Accuracy < 80%
criteria[1]["evidence"] = "results/model_metrics.json shows 76.5%, below 80% threshold"

# Log: "Criteria status: 1/4 met"
```

### Step 5: Reflect
```python
# Reflector sees:
# - Stage 1 just completed
# - Accuracy criterion not met
# - Stage 2 "Generate report" won't fix accuracy
# - Criterion 2 "Feature importance" not addressed in any remaining stage

# Reflector output:
{
    "stage_modifications": [
        {
            "index": 2,
            "new_description": "Generate comprehensive report including model performance, feature importance, and recommendations"
        }
    ],
    "new_stages": [
        {
            "title": "Hyperparameter Tuning",
            "description": "Optimize model hyperparameters using GridSearchCV to improve accuracy above 80% threshold. Try max_depth in [10, 20, 30] and n_estimators in [100, 200, 300]."
        }
    ]
}

# Callback updates stages:
stages[2]["description"] = "Generate comprehensive report..."
stages.append({
    "index": 3,
    "title": "Hyperparameter Tuning",
    "completed": False,
})
```

### Step 6: Mark Complete
```python
stages[1]["completed"] = True
state["current_stage_index"] = 1
```

### Updated State (End of Iteration 1)
```python
stages = [
    {"index": 0, "title": "Load data", "completed": True},
    {"index": 1, "title": "Train model", "completed": True},  # ← Just completed
    {"index": 2, "title": "Generate report", "completed": False, "description": "Updated..."},  # ← Modified
    {"index": 3, "title": "Hyperparameter Tuning", "completed": False},  # ← New!
]

criteria = [
    {"criteria": "Data loaded and validated", "met": True},
    {"criteria": "Model accuracy > 80%", "met": False},  # ← Still not met
    {"criteria": "Feature importance analyzed", "met": False},
    {"criteria": "Report with recommendations", "met": False},
]
```

**Next iteration** will implement stage 2 (Generate report) with the updated description that includes feature importance.

---

## Safety Mechanisms

### 1. Maximum Iterations
```python
max_iterations = 50  # Safety limit

if iteration >= max_iterations:
    logger.error(f"Reached maximum iterations ({max_iterations}). Exiting.")
    yield timeout_event
    return
```

**Why 50?** With typical analyses having 3-7 stages, and reflector rarely adding more than 2-3 stages, 50 iterations provides ample room while preventing infinite loops.

### 2. Stuck Detection
```python
if not remaining_stages:
    # Try to have reflector add stages
    run_stage_reflector()

    if still_no_stages:
        logger.error("No stages and reflector can't help. Exiting.")
        return
```

### 3. Graceful Failure
```python
try:
    run_implementation_loop()
except Exception as e:
    logger.error(f"Stage failed: {e}")
    # Skip stage, continue to next
    continue
```

Stages can fail without killing the entire workflow.

### 4. State Refresh
```python
# At start of every iteration
stages = state.get("high_level_stages", [])
criteria = state.get("high_level_success_criteria", [])
```

**Why?** Because callbacks modify state. Always re-fetch to get latest values.

---

## Exit Conditions

### Success Exit
```python
if all(c.get("met", False) for c in criteria):
    yield completion_event
    return  # → Workflow continues to Summary Agent
```

### Warning Exit
```python
if not remaining_stages and criteria_not_met:
    yield warning_event
    return  # → Workflow continues to Summary Agent (partial completion)
```

### Error Exit
```python
if stages_invalid or criteria_invalid:
    yield error_event
    return  # → Workflow terminates
```

### Timeout Exit
```python
if iteration >= max_iterations:
    yield timeout_event
    return  # → Workflow continues to Summary Agent (partial completion)
```

---

## Key Design Principles

### 1. Criteria-Driven, Not Stage-Driven
Stages are a **means to an end**. Criteria are the **actual goal**. Exit when criteria met, not when stages done.

### 2. Adaptive Planning
The plan is **not static**. It evolves based on discoveries during implementation.

### 3. Evidence-Based Validation
Don't assume completion. **Inspect files** to verify claims.

### 4. Progressive Refinement
Each stage builds on previous work. Criteria accumulate (once met, stay met).

### 5. Fail Gracefully
Stage failures don't kill the workflow. Skip and continue.

### 6. Observable Progress
Extensive logging provides transparency into the decision-making process.

---

## Comparison to Traditional Workflows

| Traditional Workflow | Stage Orchestration Loop |
|---------------------|--------------------------|
| Fixed plan | Adaptive plan |
| Execute all stages | Exit when criteria met |
| Stage completion = success | Criteria fulfillment = success |
| Linear execution | Reflective execution |
| Fail on error | Skip and continue |
| Static stages | Dynamic stages (can add/modify) |
| No validation | Evidence-based validation |

---

## Performance Characteristics

**Time complexity**: O(n * m) where:
- n = number of stages (3-10 typical, can grow to 15+)
- m = average iterations per stage (2-3 typical in implementation loop)

**Space complexity**: O(e) where e = event count (managed by compression)

**Typical execution**:
- 5 stages × 2.5 iterations × 2 minutes = 25 minutes
- But can exit early if criteria met sooner
- Or extend if reflector adds stages

**Worst case**:
- 50 iterations × 5 minutes = 4+ hours
- In practice, timeout or criteria met well before this

---

## Conclusion

The Stage Orchestration Loop is a **sophisticated adaptive control flow** that implements:

1. **Criteria-based exit conditions** (not stage-based)
2. **Adaptive replanning** (reflector modifies plan mid-execution)
3. **Evidence-based validation** (criteria checker inspects files)
4. **Progressive refinement** (multiple review loops per stage)
5. **Graceful degradation** (skip failed stages)
6. **Context management** (manual compression after each stage)

It's not a simple for-loop - it's a **cognitive control system** that balances exploration (implementing stages) with validation (checking criteria) and adaptation (reflecting on progress).

This design allows the system to handle:
- **Incomplete initial plans** (reflector adds missing stages)
- **Discovery during implementation** (reflector adapts based on findings)
- **Changing requirements** (criteria can reveal gaps in plan)
- **Partial failures** (skip failed stages, continue making progress)

The result is a **resilient, adaptive execution engine** that completes complex multi-stage analyses even when the initial plan is imperfect.
