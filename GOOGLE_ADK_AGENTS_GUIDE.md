# Google ADK Pre-Built Agents Guide

Based on this codebase's usage, here are the **pre-built agent types** in Google ADK and their unique characteristics.

---

## Agent Type Hierarchy

```
Agent (base interface)
│
├── BaseAgent (abstract base class for custom agents)
│   └── Used for: StageOrchestratorAgent (custom)
│
├── LlmAgent (LLM-powered agent with tools)
│   └── Used for: LoopDetectionAgent (extended)
│
├── SequentialAgent (runs sub-agents in sequence)
│   └── Used for: root workflow composition
│
└── LoopAgent (repeats sub-agents with exit conditions)
    └── Used for: NonEscalatingLoopAgent (extended)
```

---

## 1. BaseAgent (Abstract Base)

**Purpose**: Foundation for creating **fully custom agents**

**Location**: `google.adk.agents.BaseAgent`

**What it provides**:
- `name` and `description` fields
- `run_async()` interface for agent execution
- Event streaming protocol
- Access to `InvocationContext` (session, state, etc.)

**What YOU must implement**:
```python
class StageOrchestratorAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ← Write your entire control flow here
        while not done:
            # Custom logic
            yield event
```

**Uniqueness**:
- ✅ **Zero opinions** - total freedom
- ✅ Write **any control flow** (loops, conditionals, recursion)
- ✅ Access raw session state
- ❌ No built-in LLM calling, tool handling, or loops

**Used in this repo**:
- `StageOrchestratorAgent` (stage-by-stage orchestration with adaptive replanning)

**When to use**:
- Complex orchestration logic that doesn't fit pre-built patterns
- Custom state machines
- Agents that coordinate other agents

**Example from codebase** (`stage_orchestrator.py:111-511`):
```python
class StageOrchestratorAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        iteration = 0
        while iteration < max_iterations:
            # 1. Check exit condition (all criteria met?)
            if all(c["met"] for c in criteria):
                return

            # 2. Get next stage
            next_stage = get_next_uncompleted_stage()

            # 3. Run implementation loop
            async for event in self.implementation_loop.run_async(ctx):
                yield event

            # 4. Check criteria
            async for event in self.criteria_checker.run_async(ctx):
                yield event

            # 5. Reflect and adapt
            async for event in self.stage_reflector.run_async(ctx):
                yield event

            # 6. Mark complete
            next_stage["completed"] = True
            iteration += 1
```

**vs LangGraph**: Similar to defining a custom node function, but with full lifecycle control.

---

## 2. LlmAgent (LLM-Powered Agent)

**Purpose**: Agent that **calls an LLM** with tools and structured output

**Location**: `google.adk.agents.LlmAgent`

**What it provides**:
- ✅ **Automatic LLM calling** (via `model` parameter)
- ✅ **Tool integration** (pass `tools` list, ADK handles calling)
- ✅ **Structured output** (via `output_schema` + Pydantic)
- ✅ **Output key** (saves LLM response to `state[output_key]`)
- ✅ **Callbacks** (`before_agent_callback`, `after_agent_callback`)
- ✅ **Planners** (BuiltInPlanner with thinking support)
- ✅ **Context caching** (automatic prompt caching)

**Configuration**:
```python
LlmAgent(
    name="my_agent",
    model="google/gemini-2.5-pro",           # LLM to use
    instruction="You are a helpful...",      # System prompt
    tools=[read_file, search_files],         # Available tools
    output_schema=MyPydanticModel,           # Structured output (optional)
    output_key="my_output",                  # Where to save in state
    planner=BuiltInPlanner(...),             # Thinking config
    generate_content_config=GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=4000,
    ),
    before_agent_callback=setup_fn,          # Pre-execution hook
    after_agent_callback=cleanup_fn,         # Post-execution hook
)
```

**Uniqueness**:
- ✅ **Declarative tool use** - just pass functions, ADK handles calling
- ✅ **Structured output** - enforces JSON schema via `output_schema`
- ✅ **State integration** - automatic save to `state[output_key]`
- ✅ **Streaming** - events streamed automatically
- ❌ **No custom control flow** - just calls LLM once

**Used in this repo**:
- Extended as `LoopDetectionAgent` (adds repetition detection)
- Used for: plan_maker, plan_reviewer, criteria_checker, stage_reflector, summary

**Example from codebase** (`agent.py:617-627`):
```python
success_criteria_checker = LoopDetectionAgent(  # extends LlmAgent
    name="success_criteria_checker",
    model=REVIEW_MODEL,
    description="Checks which high-level success criteria have been met.",
    instruction=criteria_checker_instructions,
    tools=tools,  # Needs tools to inspect files
    output_schema=CRITERIA_CHECKER_OUTPUT_SCHEMA,  # Enforces structure
    output_key="criteria_checker_output",          # Saves to state
    after_agent_callback=combined_criteria_callback,
    generate_content_config=get_generate_content_config(temperature=0.0),
)
```

**Output schema enforcement** (`agent.py:81-85`):
```python
class CriteriaCheckerOutput(BaseModel):
    """Updated success criteria status."""
    criteria_updates: List[CriteriaUpdate] = Field(...)

# LLM is forced to produce JSON matching this schema
```

**vs LangGraph**:
- LangGraph: You call `llm.invoke()` manually, parse response, handle tools
- ADK LlmAgent: Pass tools + schema, everything is automatic

---

## 3. SequentialAgent (Linear Composition)

**Purpose**: Run sub-agents **one after another** in order

**Location**: `google.adk.agents.SequentialAgent`

**What it provides**:
- ✅ **Linear execution** - runs sub-agents in array order
- ✅ **Automatic state passing** - each agent sees updated state
- ✅ **Event aggregation** - streams events from all sub-agents
- ✅ **Early exit** - stops if any agent sets `end_invocation`

**Configuration**:
```python
SequentialAgent(
    name="my_workflow",
    description="Runs agents A, B, C in sequence",
    sub_agents=[agent_a, agent_b, agent_c],
)
```

**Uniqueness**:
- ✅ **Simplest composition** - just an array
- ✅ **No conditionals** - always runs all agents
- ✅ **Hierarchical** - sub-agents can be any agent type (even other SequentialAgents)
- ❌ **No loops** - runs each once
- ❌ **No branching** - linear only

**Used in this repo** (`agent.py:675-684`):
```python
workflow = SequentialAgent(
    name="agentic_data_scientist_workflow",
    description="Complete Agentic Data Scientist workflow...",
    sub_agents=[
        high_level_planning_loop,    # LoopAgent (exits when plan approved)
        high_level_plan_parser,       # LlmAgent (parses plan to JSON)
        stage_orchestrator,           # Custom agent (runs stages)
        summary_agent,                # LlmAgent (generates final report)
    ],
)
```

**Execution flow**:
```
1. high_level_planning_loop runs (may take multiple iterations)
   → Updates state["high_level_plan"]
2. high_level_plan_parser runs
   → Updates state["high_level_stages"], state["high_level_success_criteria"]
3. stage_orchestrator runs
   → Implements all stages, updates state["stage_implementations"]
4. summary_agent runs
   → Reads state["stage_implementations"], produces final report
```

**vs LangGraph**:
```python
# LangGraph equivalent
graph = StateGraph(State)
graph.add_edge("planning_loop", "plan_parser")
graph.add_edge("plan_parser", "stage_orchestrator")
graph.add_edge("stage_orchestrator", "summary")
graph.set_entry_point("planning_loop")
```

**Mental model**: Like a **pipeline** or **middleware chain**

---

## 4. LoopAgent (Iterative Execution)

**Purpose**: **Repeat sub-agents** until exit condition met

**Location**: `google.adk.agents.LoopAgent`

**What it provides**:
- ✅ **Iteration logic** - repeats sub-agents automatically
- ✅ **Max iterations** - safety limit to prevent infinite loops
- ✅ **Escalation detection** - exits when sub-agent escalates
- ✅ **Event streaming** - streams all iterations

**Configuration**:
```python
LoopAgent(
    name="my_loop",
    description="Repeats sub-agents until done",
    sub_agents=[agent_a, agent_b, agent_c],
    max_iterations=10,  # Safety limit
)
```

**Execution flow**:
```python
# Pseudocode for LoopAgent
for iteration in range(max_iterations):
    for sub_agent in sub_agents:
        run sub_agent
        if sub_agent.escalate:
            return  # Exit loop
    # Loop back to first sub_agent
```

**Uniqueness**:
- ✅ **Built-in loop** - no manual while loop needed
- ✅ **Escalation protocol** - sub-agents signal "I'm done" via escalate flag
- ✅ **Iteration limit** - prevents runaway loops
- ❌ **No custom exit logic** - only escalate or max iterations

**Used in this repo** (`agent.py:571-580`):
```python
high_level_planning_loop = NonEscalatingLoopAgent(  # extends LoopAgent
    name="high_level_planning_loop",
    description="Carries out high-level planning through multiple iterations.",
    sub_agents=[
        plan_maker_agent,                # Creates/revises plan
        plan_reviewer_agent,             # Reviews plan
        create_review_confirmation_agent(...),  # Decides: approve or revise?
    ],
    max_iterations=10,
)
```

**How it exits**:
1. `plan_maker` creates plan → no escalate
2. `plan_reviewer` reviews plan → no escalate
3. `review_confirmation` decides:
   - If plan approved: sets `escalate=True` → **exits loop**
   - If needs revision: no escalate → **loops back to step 1**
4. After 10 iterations: **force exit** (safety)

**vs LangGraph**:
```python
# LangGraph equivalent
def should_continue(state):
    if state["iteration"] >= 10:
        return "exit"
    if state["plan_approved"]:
        return "exit"
    return "continue"

graph.add_conditional_edges(
    "review_confirmation",
    should_continue,
    {
        "continue": "plan_maker",  # Loop back
        "exit": END,
    }
)
```

**Mental model**: Like a **do-while loop** with early exit

---

## Custom Extensions in This Repo

### 1. LoopDetectionAgent (extends LlmAgent)

**Location**: `agents/adk/loop_detection.py`

**Added functionality**:
- ✅ **Repetition detection** - monitors streaming output for patterns
- ✅ **Early termination** - stops if agent loops (repeats same text)
- ✅ **Configurable thresholds** - min_pattern_length, repetition_threshold

**Why needed**: LLMs can sometimes enter infinite generation loops

**Implementation** (`loop_detection.py:22-323`):
```python
class LoopDetectionAgent(LlmAgent):
    min_pattern_length: int = 200      # Minimum chars to detect as pattern
    repetition_threshold: int = 5      # How many repetitions = loop
    window_size: int = 5000            # Sliding window for analysis

    async def _run_async_impl(self, ctx: InvocationContext):
        async for event in self._llm_flow.run_async(ctx):
            # Extract text from event
            event_text = self._extract_text_from_event(event)

            # Check for repetitive patterns
            if event_text:
                self._content_buffer += event_text
                loop_detected, pattern = self._detect_pattern_repetition(
                    self._content_buffer
                )

                if loop_detected:
                    yield warning_event
                    return  # Stop this agent (not entire workflow)

            yield event
```

**Pattern detection algorithm**:
```python
def _detect_pattern_repetition(self, text):
    # Check smallest patterns first (optimization)
    for pattern_len in range(min_pattern_length, max_pattern_length):
        for start in range(text_length):
            pattern = text[start:start + pattern_len]
            count = count_consecutive_repetitions(pattern, text)

            if count >= repetition_threshold:
                return True, pattern  # Loop detected!

    return False, None
```

---

### 2. NonEscalatingLoopAgent (extends LoopAgent)

**Location**: `agents/adk/agent.py:379-400`

**Added functionality**:
- ✅ **Suppresses escalate flags** - prevents loop exit from failing parent
- ✅ **Graceful iteration** - allows iterative refinement without termination

**Why needed**: Standard LoopAgent propagates escalate flag up, failing entire workflow. This allows loops to exit without killing the workflow.

**Implementation**:
```python
class NonEscalatingLoopAgent(LoopAgent):
    """A loop agent that does not propagate escalate flags upward."""

    async def _run_async_impl(self, ctx: InvocationContext):
        times_looped = 0
        while not self.max_iterations or times_looped < self.max_iterations:
            for sub_agent in self.sub_agents:
                should_exit = False
                async for event in sub_agent.run_async(ctx):
                    if event.actions.escalate:
                        event.actions.escalate = False  # ← Suppress!
                        should_exit = True
                    yield event
                    if should_exit:
                        break

                if should_exit:
                    return  # Exit loop gracefully (no escalation)
            times_looped += 1
```

**Use case**:
```python
# Without NonEscalating:
LoopAgent([plan_maker, plan_reviewer, confirmation])
# If confirmation escalates → entire workflow fails ❌

# With NonEscalating:
NonEscalatingLoopAgent([plan_maker, plan_reviewer, confirmation])
# If confirmation escalates → loop exits, workflow continues ✅
```

---

## Agent Comparison Table

| Agent Type | Purpose | Control Flow | LLM Calls | Tools | Custom Logic |
|------------|---------|--------------|-----------|-------|--------------|
| **BaseAgent** | Custom agents | You write it | Manual | Manual | ✅ Full control |
| **LlmAgent** | LLM with tools | Single call | Automatic | Automatic | ❌ No control |
| **SequentialAgent** | Linear pipeline | A → B → C | Delegates | Delegates | ❌ Fixed order |
| **LoopAgent** | Iterations | Repeat until exit | Delegates | Delegates | ⚠️ Escalate only |

**Delegates** = Passes to sub-agents

---

## Unique ADK Features Across All Agents

### 1. Callback System

**All agents support**:
```python
agent = LlmAgent(
    before_agent_callback=setup_function,   # Runs before agent starts
    after_agent_callback=cleanup_function,  # Runs after agent completes
)
```

**Used for**:
- State transformations (parse JSON → runtime state)
- Event compression (summarize old events)
- Validation (check criteria, update flags)
- Control flow (set escalate flag based on decision)

**Example** (`agent.py:117-207`):
```python
def plan_parser_callback(callback_context: CallbackContext):
    """Transform parsed JSON into runtime state."""
    parsed = state.get("parsed_plan_output")

    # Transform to runtime format
    stages = [
        {
            "index": i,
            "title": s["title"],
            "completed": False,           # ← Add tracking
            "implementation_result": None,
        }
        for i, s in enumerate(parsed["stages"])
    ]

    state["high_level_stages"] = stages  # ← Update state

# Use in agent
plan_parser = LlmAgent(
    output_schema=PlanParserOutput,
    after_agent_callback=plan_parser_callback,  # ← Transform after LLM
)
```

---

### 2. Output Schema Validation

**LlmAgent only**:
```python
class CriteriaCheckerOutput(BaseModel):
    criteria_updates: List[CriteriaUpdate]

agent = LlmAgent(
    output_schema=CriteriaCheckerOutput,  # ← Enforces structure
    output_key="criteria_checker_output",
)
```

**What happens**:
1. LLM is forced to produce JSON matching schema
2. ADK validates and parses to Pydantic model
3. Saves to `state[output_key]`
4. Callback can transform it further

**This is unique to ADK** - LangGraph doesn't have built-in structured output enforcement.

---

### 3. Hierarchical Composition

**Agents can contain agents**:
```python
SequentialAgent(
    sub_agents=[
        LoopAgent(
            sub_agents=[
                LlmAgent(...),
                LlmAgent(...),
            ]
        ),
        StageOrchestratorAgent(
            implementation_loop=LoopAgent(
                sub_agents=[
                    ClaudeCodeAgent(...),
                    LlmAgent(...),
                ]
            ),
        ),
    ]
)
```

**Tree structure** - not flat graph like LangGraph.

---

### 4. Automatic Event Streaming

**All agents yield events automatically**:
- MessageEvent (text output)
- FunctionCallEvent (tool usage)
- FunctionResponseEvent (tool results)
- UsageEvent (token counts)

**No manual `yield`** needed - ADK handles it.

---

### 5. Session State Integration

**All agents access shared state**:
```python
async def _run_async_impl(self, ctx: InvocationContext):
    state = ctx.session.state  # Shared dict
    state["my_key"] = "my_value"
```

**State persists across agents** - no manual state passing.

---

## What's NOT in ADK (vs other frameworks)

❌ **No pre-built chains** (like LangChain's ConversationalRetrievalChain)
❌ **No ReAct agent** (you build it with LoopAgent + LlmAgent)
❌ **No vector store integrations** (use tools)
❌ **No pre-built memory** (you manage state)
❌ **No execution graph visualization** (LangGraph has this)

ADK is **lower-level** than LangChain but **higher-level** than LangGraph.

---

## Mental Model Summary

**Think of ADK agents as**:

1. **BaseAgent** = `class MyComponent extends React.Component` (full control)
2. **LlmAgent** = `<LLMCall prompt="..." tools={[...]} />` (declarative)
3. **SequentialAgent** = `<Pipeline steps={[A, B, C]} />` (composition)
4. **LoopAgent** = `<Repeat until={condition}>{children}</Repeat>` (iteration)

**Compose them hierarchically** to build complex workflows.

**Extend them** when you need custom behavior.

**vs LangGraph**: ADK gives you **components**, LangGraph gives you **primitives**.

---

## When to Use Each Agent Type

### Use BaseAgent when:
- ✅ Need custom orchestration logic
- ✅ Complex state machines
- ✅ Agents that coordinate other agents
- Example: `StageOrchestratorAgent`

### Use LlmAgent when:
- ✅ Need to call LLM with tools
- ✅ Want structured output (Pydantic)
- ✅ Standard "prompt → LLM → response" flow
- Example: plan_maker, criteria_checker

### Use SequentialAgent when:
- ✅ Linear pipeline (A → B → C → D)
- ✅ Each step depends on previous
- ✅ No loops or conditionals needed
- Example: root workflow

### Use LoopAgent when:
- ✅ Need iterative refinement
- ✅ Repeat until approval
- ✅ Generate → Review → Revise pattern
- Example: planning loop, implementation loop

---

## Conclusion

**Google ADK provides 4 core agent types**:
1. **BaseAgent** - Custom agents (full control)
2. **LlmAgent** - LLM with tools (declarative)
3. **SequentialAgent** - Linear composition
4. **LoopAgent** - Iterative execution

**Plus**:
- Callback system (before/after hooks)
- Output schema validation (Pydantic)
- Hierarchical composition (agents in agents)
- Automatic event streaming
- Session state integration

**This repo extends** LlmAgent and LoopAgent to add:
- Loop detection (prevent infinite generation)
- Non-escalating loops (graceful iteration)

The combination creates a **component library** for building complex multi-agent systems with minimal boilerplate.
