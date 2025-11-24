# Agentic Data Scientist - Comprehensive Architecture Review

## Executive Summary

This repository implements a sophisticated **hybrid multi-agent cognitive orchestration system** that combines Google's Agent Development Kit (ADK) for planning/orchestration with Anthropic's Claude Agent SDK for code execution. The architecture demonstrates advanced AI engineering patterns including adaptive planning, iterative refinement loops, continuous validation, and aggressive context management.

---

## 1. Google ADK vs Claude Agent SDK: A Tale of Two Frameworks

### 1.1 Google ADK (Agent Development Kit)

**Role in this system**: Planning, orchestration, validation, and reflection

**Key characteristics**:
- **Multi-agent composition**: Native support for hierarchical agent structures (SequentialAgent, LoopAgent)
- **Session management**: Built-in event streaming and state management
- **Tool integration**: Explicit tool declaration with function signatures
- **Event-based architecture**: All agent communication flows through Event objects
- **Context caching**: Automatic prompt caching for repeated contexts
- **Model agnostic**: Works with any LiteLLM-compatible model (uses OpenRouter for Gemini models)

**What it handles here**:
```python
# ADK manages the entire workflow structure
workflow = SequentialAgent(
    sub_agents=[
        high_level_planning_loop,      # Plan creation & review
        high_level_plan_parser,         # Structured plan extraction
        stage_orchestrator,             # Execution orchestration
        summary_agent,                  # Final report generation
    ]
)
```

**Agents built with ADK** (from `src/agentic_data_scientist/agents/adk/agent.py`):
1. **plan_maker_agent** - Creates comprehensive analysis plans
2. **plan_reviewer_agent** - Validates plan completeness
3. **plan_parser** - Converts natural language plans to structured JSON
4. **success_criteria_checker** - Verifies which criteria are met
5. **stage_reflector** - Adapts remaining stages based on progress
6. **summary_agent** - Synthesizes final reports
7. **review_confirmation_agent** - Decides loop exits

### 1.2 Claude Agent SDK

**Role in this system**: Code execution, file operations, and scientific computation

**Key characteristics**:
- **Agentic coding**: Autonomous coding agent with extended reasoning
- **Built-in tools**: File system, bash execution, code editing
- **Skills system**: Loads from `.claude/skills/` directory (120+ scientific skills)
- **MCP integration**: Model Context Protocol for tool servers
- **Preset system prompts**: "claude_code" preset for coding-focused behavior
- **Subprocess architecture**: Runs in separate process with IPC via JSON messages

**What it handles here** (from `src/agentic_data_scientist/agents/claude_code/agent.py`):
```python
# Claude Code handles all implementation work
ClaudeCodeAgent(
    working_dir=working_dir,
    model="claude-sonnet-4-5-20250929",
    system_prompt={"type": "preset", "preset": "claude_code", "append": custom_instructions},
    setting_sources=["project", "user", "local"],  # Loads skills from .claude/
)
```

### 1.3 The Integration Pattern: ADK Wraps Claude SDK

**Critical architectural decision** (from `ClaudeCodeAgent._run_async_impl`):

```python
# Claude SDK produces Message objects (AssistantMessage, UserMessage, etc.)
async for message in query(prompt=prompt, options=options):
    # Convert Claude's message types to Google GenAI types
    if message_type == "AssistantMessage":
        # TextBlock → Part.from_text(text=...)
        # ThinkingBlock → Part(text=..., thought=True)
        # ToolUseBlock → Part.from_function_call(name=..., args=...)
    elif message_type == "UserMessage":
        # ToolResultBlock → Part.from_function_response(name=..., response=...)

    # Wrap in ADK Event and yield upstream
    yield Event(author=self.name, content=Content(role="model", parts=parts))
```

**Why this matters**: The `ClaudeCodeAgent` class is a **bridge agent** that:
1. Takes ADK InvocationContext as input
2. Extracts stage information from ADK session state
3. Calls Claude SDK's `query()` function
4. Translates Claude's message stream to ADK Events
5. Stores results back in ADK session state

---

## 2. Cognitive Orchestration Architecture

### 2.1 The Control Flow Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW ROOT                                 │
│                  (SequentialAgent)                               │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ PLANNING LOOP (NonEscalatingLoopAgent, max 10 iterations)  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Plan Maker (LoopDetectionAgent)                      │  │  │
│  │  │ - Creates comprehensive plan from user request       │  │  │
│  │  │ - Has tools: read_file, list_directory, fetch_url    │  │  │
│  │  │ - Output: Natural language plan → state["high_level_plan"] │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Plan Reviewer (LoopDetectionAgent)                   │  │  │
│  │  │ - Reviews plan for completeness & correctness        │  │  │
│  │  │ - Output: Feedback → state["plan_review_feedback"]   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Review Confirmation (LoopDetectionAgent)             │  │  │
│  │  │ - Decides: exit loop or iterate?                     │  │  │
│  │  │ - Output: {"exit": bool, "reason": str}              │  │  │
│  │  │ - Escalates if exit=true → exits NonEscalatingLoop   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ PLAN PARSER (LoopDetectionAgent)                          │  │
│  │ - Converts plan to structured stages + success criteria    │  │
│  │ - Output schema: PlanParserOutput (Pydantic)               │  │
│  │ - Callback parses JSON → state["high_level_stages"]        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ STAGE ORCHESTRATOR (Custom Agent)                         │  │
│  │                                                             │  │
│  │  For each uncompleted stage:                               │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ IMPLEMENTATION LOOP (NonEscalatingLoopAgent)         │  │  │
│  │  │  ┌────────────────────────────────────────────────┐  │  │  │
│  │  │  │ Coding Agent (ClaudeCodeAgent)                 │  │  │  │
│  │  │  │ - Wraps Claude Agent SDK                       │  │  │  │
│  │  │  │ - Access to 120+ scientific Skills             │  │  │  │
│  │  │  │ - Output → state["implementation_summary"]     │  │  │  │
│  │  │  └────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌────────────────────────────────────────────────┐  │  │  │
│  │  │  │ Review Agent (LoopDetectionAgent)              │  │  │  │
│  │  │  │ - Validates implementation quality             │  │  │  │
│  │  │  │ - Output → state["review_feedback"]            │  │  │  │
│  │  │  └────────────────────────────────────────────────┘  │  │  │
│  │  │  ┌────────────────────────────────────────────────┐  │  │  │
│  │  │  │ Implementation Review Confirmation             │  │  │  │
│  │  │  │ - Decides: approve or iterate?                 │  │  │  │
│  │  │  └────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Criteria Checker (LoopDetectionAgent)                │  │  │
│  │  │ - Inspects files to verify criteria                  │  │  │
│  │  │ - Updates: state["high_level_success_criteria"]      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Stage Reflector (LoopDetectionAgent)                 │  │  │
│  │  │ - Adapts remaining stages based on progress          │  │  │
│  │  │ - Can modify or add stages                            │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  Exit when: all(criterion["met"] for criterion in criteria)│  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ SUMMARY AGENT (LoopDetectionAgent)                        │  │
│  │ - Synthesizes all work into publication-ready report       │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Plan Creation: The Scientific Method Encoded

**Location**: `src/agentic_data_scientist/prompts/base/plan_maker.md`

The plan maker implements a **data science-first cognitive model**:

```markdown
# You are the **plan_maker** – a data science strategist

Transform the user's request into a comprehensive, detailed high-level plan
based on data science intuition and domain expertise.

## Output Format:
1. Analysis Stages (logical milestones, not technical tasks)
2. Success Criteria (end-state requirements)
3. Recommended Approaches (methodologies and techniques)
```

**Key cognitive primitives encoded**:

1. **Intuition-First Design** (line 44):
   ```markdown
   - **Intuition First**: Let analytical and scientific reasoning drive your plan,
     not technical constraints
   ```

2. **Analytical Flow Pattern** (line 46):
   ```markdown
   - **Analytical Flow**: Structure stages to follow natural analysis progression
     (e.g., exploration → analysis → interpretation → validation)
   ```

3. **No Tool Contamination** (line 49):
   ```markdown
   - **No Tool Names in Plan**: You must not include any specific tool name...
     Focus on the methodology, algorithm, and success criteria
   ```

4. **Separation of Concerns** (line 51):
   ```markdown
   - **Success Criteria vs Stages**: Success criteria are end-state requirements
     for the entire analysis. Stages are progressive steps. They need NOT be one-to-one.
   ```

### 2.3 Plan Specification: Structured Output with Pydantic

**Location**: `src/agentic_data_scientist/agents/adk/agent.py:53-106`

Plans are converted to structured data via **output_schema**:

```python
class Stage(BaseModel):
    """A high-level implementation stage."""
    title: str = Field(description="Stage title")
    description: str = Field(description="Detailed stage description")

class SuccessCriterion(BaseModel):
    """A success criterion for completion."""
    criteria: str = Field(description="Success criterion description")

class PlanParserOutput(BaseModel):
    """Parsed high-level plan into stages and success criteria."""
    stages: List[Stage] = Field(...)
    success_criteria: List[SuccessCriterion] = Field(...)
```

**The callback transforms this into runtime state** (`plan_parser_callback`, line 117):

```python
def plan_parser_callback(callback_context: CallbackContext):
    parsed_output = state.get("parsed_plan_output")

    # Initialize stages with tracking fields
    stages = []
    for idx, stage in enumerate(stages_data):
        stages.append({
            "index": idx,
            "title": stage["title"],
            "description": stage["description"],
            "completed": False,              # ← Runtime tracking
            "implementation_result": None,   # ← Stores implementation summary
        })

    # Initialize criteria with tracking fields
    criteria = []
    for idx, crit in enumerate(criteria_data):
        criteria.append({
            "index": idx,
            "criteria": crit["criteria"],
            "met": False,                    # ← Runtime tracking
            "evidence": None,                # ← Populated by criteria checker
        })

    state["high_level_stages"] = stages
    state["high_level_success_criteria"] = criteria
```

### 2.4 User Feedback Loops: Iterative Refinement

**There are THREE feedback loops in the system**:

#### Loop 1: Planning Loop (lines 571-580)
```python
high_level_planning_loop = NonEscalatingLoopAgent(
    sub_agents=[
        plan_maker_agent,
        plan_reviewer_agent,
        create_review_confirmation_agent(auto_exit_on_completion=True),
    ],
    max_iterations=10,
)
```

**Flow**:
1. Plan Maker creates plan → `state["high_level_plan"]`
2. Plan Reviewer reviews → `state["plan_review_feedback"]`
3. Review Confirmation decides:
   - If `{"exit": true}` → escalates → exits loop
   - If `{"exit": false}` → continues to step 1

**The confirmation agent uses structured reasoning**:
```python
# From plan_review_confirmation.md
"""
Review the plan_review_feedback and decide:
- If the plan is complete enough to proceed → exit: true
- If the plan needs significant revision → exit: false

Output JSON: {"exit": true/false, "reason": "..."}
"""
```

#### Loop 2: Implementation Loop (lines 492-497)
```python
implementation_loop = NonEscalatingLoopAgent(
    sub_agents=[coding_agent, review_agent, review_confirmation],
    max_iterations=5,
)
```

**Flow** (per stage):
1. Coding Agent implements → `state["implementation_summary"]`
2. Review Agent reviews → `state["review_feedback"]`
3. Review Confirmation decides exit/continue

#### Loop 3: Stage Orchestration Loop (lines 244-511)
```python
# From stage_orchestrator.py
while iteration < max_iterations:
    # Check exit condition
    if all(c.get("met", False) for c in criteria):
        return  # All criteria met!

    # Get next uncompleted stage
    next_stage = remaining_stages[0]

    # Run implementation loop for this stage
    async for event in self.implementation_loop.run_async(ctx):
        yield event

    # Check criteria
    async for event in self.criteria_checker.run_async(ctx):
        yield event

    # Reflect and adapt remaining stages
    async for event in self.stage_reflector.run_async(ctx):
        yield event

    # Mark stage complete
    next_stage["completed"] = True
```

**Exit condition**: `all(criterion["met"] for criterion in criteria)`

### 2.5 Plan Completion Validation: Objective Evidence-Based

**Location**: `src/agentic_data_scientist/prompts/base/criteria_checker.md`

The criteria checker implements **verification by inspection**:

```markdown
# You are the **success_criteria_checker**

For each criterion:
1. **Actively use your file inspection tools** to examine outputs
2. **Don't assume - verify** by reading relevant files
3. Determine if the criterion is NOW met based on concrete evidence
4. Provide specific evidence (file paths, metrics, observations)

# Important Rules
- **Check ALL criteria every time**
- **Once met, generally stays met**
- **Require CONCRETE EVIDENCE** - only mark as met if you can verify it
```

**Output format** (structured JSON):
```json
{
  "criteria_updates": [
    {
      "index": 0,
      "met": true,
      "evidence": "Dataset loaded in data/processed/customer_data.csv with 50,000 rows"
    },
    {
      "index": 1,
      "met": false,
      "evidence": "Model accuracy is 76.5%, below 80% threshold required"
    }
  ]
}
```

**The callback updates runtime state** (`criteria_checker_callback`, line 209):

```python
def criteria_checker_callback(callback_context: CallbackContext):
    for update in updates:
        idx = update["index"]
        criteria[idx]["met"] = update["met"]
        criteria[idx]["evidence"] = update["evidence"]
```

### 2.6 Plan Steering During Execution: Adaptive Replanning

**Location**: `src/agentic_data_scientist/prompts/base/stage_reflector.md`

The stage reflector implements **online plan adaptation**:

```markdown
# You are the **stage_reflector** – you adapt the implementation plan

After each implementation stage, reflect on:
1. What has been completed so far
2. What still needs to be done based on success criteria
3. Whether remaining stages need adjustment or extension

# You Can:
- **Modify remaining stages**: Update descriptions based on discoveries
- **Add new stages**: Extend the plan if additional work is needed
- **Do nothing**: If remaining stages are still appropriate
```

**Output schema**:
```python
class StageModification(BaseModel):
    index: int = Field(description="Stage index to modify")
    new_description: str = Field(description="Updated stage description")

class NewStage(BaseModel):
    title: str = Field(description="New stage title")
    description: str = Field(description="New stage description")

class StageReflectorOutput(BaseModel):
    stage_modifications: List[StageModification]
    new_stages: List[NewStage]
```

**Example adaptation scenario**:
```json
{
  "stage_modifications": [
    {
      "index": 3,
      "new_description": "Perform additional feature selection based on model performance observed in stage 2. Apply recursive feature elimination to improve interpretability."
    }
  ],
  "new_stages": [
    {
      "title": "Model Ensemble",
      "description": "Create ensemble of top-performing models to improve prediction accuracy beyond 85% threshold"
    }
  ]
}
```

**The callback modifies the plan** (`stage_reflector_callback`, line 291):

```python
def stage_reflector_callback(callback_context: CallbackContext):
    # Apply modifications to existing stages
    for mod in modifications:
        idx = mod["index"]
        if not stages[idx].get("completed", False):  # Only modify uncompleted
            stages[idx]["description"] = mod["new_description"]

    # Add new stages
    for new_stage in new_stages:
        stages.append({
            "index": len(stages),
            "title": new_stage["title"],
            "description": new_stage["description"],
            "completed": False,
            "implementation_result": None,
        })
```

### 2.7 Control Loops: Defense in Depth

The system has **FOUR layers of loop protection**:

#### Layer 1: Loop Detection (Repetition Detection)
**File**: `src/agentic_data_scientist/agents/adk/loop_detection.py`

```python
class LoopDetectionAgent(LlmAgent):
    """
    Monitors partial events during streaming and detects repetitive patterns.

    Configuration:
    - min_pattern_length: 200 chars (avoid false positives on file paths)
    - repetition_threshold: 5 repetitions (conservative)
    - window_size: 5000 chars (sliding window)
    """

    def _detect_pattern_repetition(self, text: str) -> Tuple[bool, Optional[str]]:
        # Optimized: check smallest patterns first, exit on first match
        for pattern_len in range(self.min_pattern_length, max_pattern):
            for start in range(0, text_len - pattern_len * self.repetition_threshold):
                pattern = clean_text[start : start + pattern_len]
                count = count_consecutive_repetitions(pattern, text, start)

                if count >= self.repetition_threshold:
                    logger.warning(f"Loop detected: Pattern of length {pattern_len} repeated {count} times")
                    return True, pattern
```

**When triggered**: Yields warning event and **returns (exits agent only, not workflow)**

#### Layer 2: NonEscalatingLoopAgent (Max Iterations)
**File**: `src/agentic_data_scientist/agents/adk/agent.py:379-400`

```python
class NonEscalatingLoopAgent(LoopAgent):
    """
    A loop agent that does not propagate escalate flags upward.

    This allows iterative refinement without failing the workflow.
    """

    async def _run_async_impl(self, ctx: InvocationContext):
        times_looped = 0
        while not self.max_iterations or times_looped < self.max_iterations:
            for sub_agent in self.sub_agents:
                # Run sub-agent
                async for event in sub_agent.run_async(ctx):
                    if event.actions.escalate:
                        event.actions.escalate = False  # ← Suppress escalation
                        return  # Exit loop gracefully
                    yield event
            times_looped += 1
```

**Max iterations**:
- Planning loop: 10 iterations
- Implementation loop: 5 iterations per stage
- Stage orchestrator: 50 iterations total

#### Layer 3: Event Compression (Context Window Management)
**File**: `src/agentic_data_scientist/agents/adk/event_compression.py`

```python
async def compression_callback(callback_context: CallbackContext):
    """
    Callback that compresses events when threshold is exceeded.

    Strategy:
    1. Trigger when events > threshold (default: 40)
    2. Keep recent events (overlap_size: 20)
    3. Compress old events using LLM summarization
    4. Truncate large texts (>10KB → 1KB)
    5. Direct assignment: session.events = new_events
    """
    if len(events) > event_threshold:
        # Get events to compress
        events_to_compress = events[start_idx:end_idx]

        # Truncate large texts
        _truncate_large_event_texts(events_to_compress)

        # Generate summary with LLM
        summary_text = await _create_event_summary_with_llm(events_to_compress)

        # Create compaction event
        compaction = EventCompaction(
            compacted_content=Content(role='model', parts=[Part(text=summary_text)])
        )
        summary_event = Event(author='event_compression', actions=EventActions(compaction=compaction))

        # Replace old events with summary
        session.events = events[:start_idx] + [summary_event] + events[end_idx:]
```

**Compression is triggered**:
- Automatically after each agent (via `after_agent_callback`)
- Manually in stage orchestrator after implementation loop
- Uses **LLM-based summarization** (400-600 word summaries)

#### Layer 4: Hard Limit (Emergency Fallback)
```python
def create_hard_limit_callback(max_events: int = 50):
    """Simply discard oldest events without summarization."""
    def hard_limit_callback(callback_context):
        if len(events) > max_events:
            session.events = events[-max_events:]  # Keep only recent
```

---

## 3. Agent Harness Dissection

### 3.1 The State Machine

**Session state is a shared dictionary** that all agents read/write:

```python
state = {
    # Planning phase
    'original_user_input': "...",
    'high_level_plan': "...",              # Natural language plan
    'plan_review_feedback': "...",         # Reviewer comments
    'parsed_plan_output': {...},           # Structured JSON

    # Execution phase
    'high_level_stages': [                 # Runtime stage tracking
        {
            "index": 0,
            "title": "...",
            "description": "...",
            "completed": False,
            "implementation_result": None,
        }
    ],
    'high_level_success_criteria': [       # Runtime criteria tracking
        {
            "index": 0,
            "criteria": "...",
            "met": False,
            "evidence": None,
        }
    ],
    'current_stage': {...},                # Stage being implemented
    'current_stage_index': 0,
    'stage_implementations': [             # History of completed work
        {
            "stage_index": 0,
            "stage_title": "...",
            "implementation_summary": "...",
        }
    ],

    # Per-stage outputs
    'implementation_summary': "...",       # From coding agent
    'review_feedback': "...",              # From review agent
    'criteria_checker_output': {...},     # Structured JSON
    'stage_reflector_output': {...},      # Structured JSON
}
```

### 3.2 The Callback System

ADK agents support **before** and **after** callbacks:

```python
agent = LoopDetectionAgent(
    name="plan_maker",
    before_agent_callback=clear_stale_state,      # Runs before agent starts
    after_agent_callback=compression_callback,    # Runs after agent completes
)
```

**Callback types used**:

1. **State transformation callbacks**:
   - `plan_parser_callback` - Transforms JSON to runtime state
   - `criteria_checker_callback` - Updates criteria met status
   - `stage_reflector_callback` - Modifies stage list

2. **Control flow callbacks**:
   - `exit_loop_callback` - Sets escalate flag based on decision
   - `clear_decision_callback` - Prevents state pollution

3. **Context management callbacks**:
   - `compression_callback` - Compresses events
   - `hard_limit_callback` - Emergency trimming

### 3.3 The Event Stream

All agent communication flows through **Event objects**:

```python
@dataclass
class Event:
    author: str                    # Agent name
    content: Content               # Google GenAI Content
    actions: EventActions          # Escalate, compaction, etc.
    partial: bool                  # Streaming vs complete
    turn_complete: bool            # End of agent turn
    timestamp: float
    invocation_id: str
```

**Event types**:
- **MessageEvent**: Text output from agents
- **FunctionCallEvent**: Tool invocations (converted from `Part.from_function_call`)
- **FunctionResponseEvent**: Tool results (converted from `Part.from_function_response`)
- **UsageEvent**: Token counts
- **CompletedEvent**: Workflow finished
- **ErrorEvent**: Failures

**The Claude→ADK translation** (from `claude_code/agent.py:376-468`):

```python
# Claude Agent SDK → Google GenAI → ADK Events
#
# AssistantMessage:
#   TextBlock      → Part.from_text(text=...)                    → MessageEvent
#   ThinkingBlock  → Part(text=..., thought=True)                → MessageEvent (is_thought=True)
#   ToolUseBlock   → Part.from_function_call(name=..., args=...) → FunctionCallEvent
#
# UserMessage:
#   ToolResultBlock → Part.from_function_response(name=..., response=...) → FunctionResponseEvent
```

---

## 4. Scientific Aspects & Cognitive Primitives

### 4.1 Claude Scientific Skills (The "Science")

**Location**: Auto-cloned from https://github.com/K-Dense-AI/claude-scientific-skills

```python
def setup_skills_directory(working_dir: str):
    """
    Clone claude-scientific-skills repository and copy skills to .claude/skills/.

    The repository contains 120+ scientific skills including:
    - Scientific databases: UniProt, PubChem, PDB, KEGG, PubMed
    - Scientific packages: BioPython, RDKit, PyDESeq2, scanpy
    """
    repo_url = "https://github.com/K-Dense-AI/claude-scientific-skills.git"
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(tmp_repo)])

    # Copy each skill directory to .claude/skills/
    for skill_dir in (tmp_repo / "scientific-skills").iterdir():
        shutil.copytree(skill_dir, skills_dir / skill_dir.name)
```

**Skills are loaded automatically** by Claude Agent SDK via `setting_sources=["project", "user", "local"]`

### 4.2 Cognitive Primitives from Science

The prompts encode **scientific reasoning patterns**:

#### Primitive 1: The Scientific Method (Plan Maker)
```markdown
Structure stages to follow natural analysis progression:
exploration → analysis → interpretation → validation
```

**Encoded in the prompt**:
1. **Exploration**: "Data Profiling and Quality Assessment"
2. **Analysis**: "Performance Analysis and Product Ranking"
3. **Interpretation**: "Insight Synthesis and Strategic Recommendations"
4. **Validation**: "Statistical Testing and Validation"

#### Primitive 2: Evidence-Based Reasoning (Criteria Checker)
```markdown
# Important Rules
- **Don't assume - verify** by reading relevant files
- **Require CONCRETE EVIDENCE** - only mark as met if you can verify it
- **Be objective** - base decisions on evidence, not assumptions
```

This mirrors **empiricism** - knowledge comes from observation, not assumption.

#### Primitive 3: Iterative Refinement (Review Loops)
```markdown
# Constructive Feedback Principles
- Start with what's working well
- Be specific about issues (not vague)
- Suggest concrete improvements
```

This mirrors **peer review** in scientific publishing.

#### Primitive 4: Adaptive Hypothesis Testing (Stage Reflector)
```markdown
After each implementation stage, reflect on:
1. What has been completed so far
2. What still needs to be done based on success criteria
3. Whether remaining stages need adjustment or extension
```

This mirrors **adaptive clinical trials** - modify protocol based on interim results.

#### Primitive 5: Hierarchical Decomposition (Plan Maker)
```markdown
Each stage should:
- Represent a meaningful analytical milestone (not technical tasks)
- Be substantial enough to be implemented as a separate unit of work
- Include a clear title and detailed description (3-6 sentences)
```

This mirrors **divide-and-conquer** problem solving in mathematics.

### 4.3 No Novel AI Algorithms - Just Engineering

**Key insight**: This system does NOT implement new ML algorithms. The "science" is:

1. **Domain-specific prompts** that encode data science workflow patterns
2. **Claude Skills** that provide access to scientific tools/databases
3. **Structured reasoning** via output schemas (Pydantic models)
4. **Iterative refinement loops** inspired by scientific processes

The innovation is in **cognitive orchestration** - how agents are composed, how feedback flows, and how context is managed.

---

## 5. Key Architectural Patterns

### 5.1 Separation of Planning and Execution

**Why**: Planning agents (ADK) don't need code execution, coding agents (Claude) don't need to plan.

**Benefit**:
- Planning uses cheaper models (Gemini via OpenRouter)
- Coding uses best model (Claude Sonnet 4.5)
- Clear separation of concerns

### 5.2 NonEscalatingLoopAgent Pattern

**Problem**: LoopAgent propagates escalate flags, causing entire workflow to fail.

**Solution**: Suppress escalation within loops, only exit loop (not workflow).

```python
if event.actions.escalate:
    event.actions.escalate = False  # Suppress
    return  # Exit loop only
```

### 5.3 Callback-Based State Management

**Why**: Agents produce structured JSON, but need to transform it to runtime state.

**Pattern**: Use `after_agent_callback` to parse JSON and update state dictionary.

```python
agent = LoopDetectionAgent(
    output_schema=PlanParserOutput,  # Agent produces JSON
    output_key="parsed_plan_output",  # Saved to state
    after_agent_callback=plan_parser_callback,  # Transforms JSON → runtime state
)
```

### 5.4 Aggressive Context Compression

**Why**: Multi-stage analyses can exceed 1M token context limit.

**Strategy**:
1. Callback-based compression (automatic)
2. Manual compression (at key points)
3. LLM summarization (preserves semantics)
4. Text truncation (prevents individual bloat)
5. Hard limits (emergency fallback)

---

## 6. Comparison to Google ADK Examples

**This system goes beyond basic ADK usage**:

1. **Custom agents**: `StageOrchestratorAgent` is a fully custom agent, not just composition of built-in agents.

2. **Hybrid SDK integration**: Wrapping Claude SDK inside ADK agents is novel.

3. **Structured state management**: Extensive use of callbacks to maintain runtime state.

4. **Advanced loop control**: `NonEscalatingLoopAgent` pattern not in standard ADK.

5. **Aggressive compression**: Comprehensive context management not in basic examples.

6. **Scientific domain modeling**: Prompts encode domain-specific cognitive patterns.

---

## 7. Weaknesses and Tradeoffs

### 7.1 Complexity
- **12 agents** with intricate dependencies
- Difficult to debug (events flow through many layers)
- High cognitive load to understand full system

### 7.2 Cost
- Orchestrated mode: $1-10 per analysis
- Multiple LLM calls for planning, review, reflection
- Worthwhile for production, expensive for exploration

### 7.3 Latency
- 5-30 minutes for complex analyses
- Iterative refinement adds overhead
- Simple mode bypasses this (30s-5min)

### 7.4 Brittle State Management
- Relies on callback execution order
- State dictionary can become inconsistent
- Requires careful key naming (e.g., `plan_review_confirmation_decision` vs `implementation_review_confirmation_decision`)

### 7.5 Token Overflow Risk
- Despite compression, very complex analyses can still overflow
- Hard limits discard information
- No graceful degradation beyond "summarize and continue"

---

## 8. Innovative Aspects

### 8.1 Hybrid SDK Architecture
**First system I've seen** that cleanly wraps one agent SDK (Claude) inside another (ADK) while preserving full event streaming.

### 8.2 Adaptive Replanning
Stage reflector can **modify the plan during execution** - most systems have static plans.

### 8.3 Evidence-Based Validation
Criteria checker doesn't just ask "are we done?" - it **inspects files** to verify claims.

### 8.4 Scientific Method Encoding
The prompt structure mirrors actual scientific reasoning (not just "do this task").

### 8.5 Callback-Based Compression
Using callbacks for context management is elegant - automatic and transparent.

---

## 9. Recommendations for Future Work

1. **Visualize State Machine**: Create a real-time dashboard showing state transitions.

2. **Checkpoint/Resume**: Save session state to disk, allow resuming from any stage.

3. **Parallel Stage Execution**: Independent stages could run concurrently.

4. **Cost Optimization**: Use cheaper models for certain tasks (e.g., criteria checking).

5. **User-in-the-Loop**: Add breakpoints for human approval before continuing.

6. **Better Error Recovery**: Currently errors skip stages - could add retry logic.

7. **Telemetry**: Add OpenTelemetry spans for observability.

8. **Unit Tests**: Test individual agents in isolation (currently integration tests only).

---

## Conclusion

This is a **sophisticated cognitive orchestration system** that successfully combines two different agent frameworks (Google ADK + Claude SDK) into a cohesive multi-agent workflow. The architecture demonstrates advanced patterns in:

- **Iterative refinement** via NonEscalatingLoopAgent
- **Adaptive planning** via stage reflection
- **Evidence-based validation** via criteria checking
- **Context management** via aggressive compression
- **Scientific reasoning** encoded in prompts

The system is production-ready for complex data science workflows, with thoughtful engineering around context limits, loop detection, and state management. While complex, the architecture is justified by the quality improvements from iterative refinement and validation.

**Bottom line**: This is not just "chain agents together" - it's a carefully designed cognitive architecture with multiple feedback loops, adaptive planning, and scientific domain modeling. The hybrid ADK+Claude approach is innovative and well-executed.
