# Google ADK vs LangGraph: Side-by-Side Comparison

## Part 1: The Planning Loop

### In Google ADK (High Level)

```python
planning_loop = LoopAgent(
    sub_agents=[plan_maker, plan_reviewer],
    max_iterations=10
)
```

**What this does**:
1. Run `plan_maker` → produces plan
2. Run `plan_reviewer` → reviews plan
3. If max_iterations not reached, go to step 1
4. Exit after 10 iterations

**You write**: 3 lines
**ADK handles**: Loop logic, event streaming, state management

---

### In LangGraph (Low Level)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# 1. DEFINE STATE (ADK does this automatically via session.state)
class PlanningState(TypedDict):
    """State for planning loop."""
    messages: Annotated[list, operator.add]  # Accumulate messages
    high_level_plan: str
    plan_review_feedback: str
    iteration_count: int
    max_iterations: int

# 2. DEFINE NODES (ADK calls these "agents")
def plan_maker_node(state: PlanningState) -> PlanningState:
    """Run the plan maker LLM."""
    # Get current iteration
    iteration = state.get("iteration_count", 0)

    # Call LLM with prompt
    response = llm.invoke([
        SystemMessage(content=plan_maker_prompt),
        HumanMessage(content=state.get("original_user_input", "")),
        # Include previous feedback if exists
        *([AIMessage(content=state.get("plan_review_feedback", ""))]
          if state.get("plan_review_feedback") else [])
    ])

    # Update state
    return {
        **state,
        "high_level_plan": response.content,
        "messages": [response],
        "iteration_count": iteration + 1,
    }

def plan_reviewer_node(state: PlanningState) -> PlanningState:
    """Run the plan reviewer LLM."""
    # Call LLM to review the plan
    response = llm.invoke([
        SystemMessage(content=plan_reviewer_prompt),
        HumanMessage(content=f"Plan to review:\n{state['high_level_plan']}"),
    ])

    # Update state
    return {
        **state,
        "plan_review_feedback": response.content,
        "messages": [response],
    }

# 3. DEFINE CONDITIONAL LOGIC (ADK's max_iterations is built-in)
def should_continue_planning(state: PlanningState) -> str:
    """Decide whether to continue the loop or exit."""
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 10)

    if iteration >= max_iter:
        return "exit"

    # Could also check if plan is approved
    # (would need to parse plan_review_feedback)

    return "continue"

# 4. BUILD THE GRAPH (ADK does this with LoopAgent)
planning_graph = StateGraph(PlanningState)

# Add nodes
planning_graph.add_node("plan_maker", plan_maker_node)
planning_graph.add_node("plan_reviewer", plan_reviewer_node)

# Define edges
planning_graph.set_entry_point("plan_maker")
planning_graph.add_edge("plan_maker", "plan_reviewer")

# Add conditional edge (the "loop" part)
planning_graph.add_conditional_edges(
    "plan_reviewer",
    should_continue_planning,
    {
        "continue": "plan_maker",  # Loop back
        "exit": END,                # Exit loop
    }
)

# Compile
planning_loop = planning_graph.compile()

# 5. RUN IT (ADK handles this in workflow execution)
result = planning_loop.invoke({
    "original_user_input": "Analyze sales data...",
    "iteration_count": 0,
    "max_iterations": 10,
})
```

**You write**: ~80 lines
**LangGraph handles**: Graph execution, state updates

---

## Part 2: The Complete Workflow

### In Google ADK

```python
# From agent.py:674-684
workflow = SequentialAgent(
    name="agentic_data_scientist_workflow",
    sub_agents=[
        high_level_planning_loop,    # LoopAgent
        high_level_plan_parser,       # LlmAgent
        stage_orchestrator,           # Custom Agent
        summary_agent,                # LlmAgent
    ],
)
```

**4 components, linear flow**

---

### In LangGraph (Full Translation)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator

# ============================================================================
# STATE DEFINITION
# ============================================================================

class WorkflowState(TypedDict):
    """Complete state for the workflow."""
    # Input
    original_user_input: str

    # Planning phase
    high_level_plan: str
    plan_review_feedback: str
    planning_iteration: int

    # Parsed plan
    high_level_stages: list[dict]
    high_level_success_criteria: list[dict]

    # Execution phase
    current_stage_index: int
    current_stage: dict
    stage_implementations: list[dict]
    implementation_summary: str
    review_feedback: str

    # Orchestration
    orchestration_iteration: int

    # Messages (for history)
    messages: Annotated[list, operator.add]

# ============================================================================
# PLANNING LOOP NODES
# ============================================================================

def plan_maker_node(state: WorkflowState) -> WorkflowState:
    """Create or revise the plan."""
    iteration = state.get("planning_iteration", 0)

    # Call LLM
    response = llm.invoke([
        SystemMessage(content=plan_maker_prompt),
        HumanMessage(content=state["original_user_input"]),
        *([AIMessage(content=state.get("plan_review_feedback", ""))]
          if state.get("plan_review_feedback") else [])
    ])

    return {
        **state,
        "high_level_plan": response.content,
        "planning_iteration": iteration + 1,
        "messages": [response],
    }

def plan_reviewer_node(state: WorkflowState) -> WorkflowState:
    """Review the plan."""
    response = llm.invoke([
        SystemMessage(content=plan_reviewer_prompt),
        HumanMessage(content=f"Plan:\n{state['high_level_plan']}"),
    ])

    return {
        **state,
        "plan_review_feedback": response.content,
        "messages": [response],
    }

def plan_review_confirmation_node(state: WorkflowState) -> WorkflowState:
    """Decide if plan is good enough."""
    response = llm.invoke([
        SystemMessage(content=review_confirmation_prompt),
        HumanMessage(content=f"Feedback:\n{state['plan_review_feedback']}"),
    ])

    # Parse JSON response: {"exit": true/false, "reason": "..."}
    decision = json.loads(response.content)

    return {
        **state,
        "plan_approved": decision["exit"],
        "messages": [response],
    }

def should_exit_planning(state: WorkflowState) -> Literal["continue", "exit"]:
    """Conditional edge for planning loop."""
    if state.get("planning_iteration", 0) >= 10:
        return "exit"
    if state.get("plan_approved", False):
        return "exit"
    return "continue"

# ============================================================================
# PLAN PARSER NODE
# ============================================================================

def plan_parser_node(state: WorkflowState) -> WorkflowState:
    """Parse plan into structured stages and criteria."""
    response = llm_with_structured_output.invoke([
        SystemMessage(content=plan_parser_prompt),
        HumanMessage(content=state["high_level_plan"]),
    ])

    # Response is PlanParserOutput (Pydantic)
    parsed = response

    # Initialize stages with tracking fields
    stages = [
        {
            "index": i,
            "title": stage.title,
            "description": stage.description,
            "completed": False,
            "implementation_result": None,
        }
        for i, stage in enumerate(parsed.stages)
    ]

    # Initialize criteria with tracking fields
    criteria = [
        {
            "index": i,
            "criteria": criterion.criteria,
            "met": False,
            "evidence": None,
        }
        for i, criterion in enumerate(parsed.success_criteria)
    ]

    return {
        **state,
        "high_level_stages": stages,
        "high_level_success_criteria": criteria,
        "current_stage_index": 0,
        "stage_implementations": [],
    }

# ============================================================================
# STAGE ORCHESTRATOR (This is complex - simplified here)
# ============================================================================

def stage_orchestrator_node(state: WorkflowState) -> WorkflowState:
    """
    Run stage-by-stage implementation with criteria checking.

    This is a MASSIVE simplification. In reality, this would be
    another nested graph with:
    - Implementation loop (coding -> review -> confirmation)
    - Criteria checker
    - Stage reflector
    """
    stages = state["high_level_stages"]
    criteria = state["high_level_success_criteria"]

    # Find next uncompleted stage
    remaining = [s for s in stages if not s.get("completed", False)]

    if not remaining:
        return state

    next_stage = remaining[0]

    # === IMPLEMENTATION LOOP (would be nested graph) ===
    # Simplified: just run coding agent once
    implementation = coding_agent.invoke(next_stage["description"])

    # Store result
    next_stage["implementation_result"] = implementation
    next_stage["completed"] = True

    state["stage_implementations"].append({
        "stage_index": next_stage["index"],
        "stage_title": next_stage["title"],
        "implementation_summary": implementation,
    })

    # === CRITERIA CHECKER ===
    criteria_response = llm_with_structured_output.invoke([
        SystemMessage(content=criteria_checker_prompt),
        HumanMessage(content=f"Check criteria against: {implementation}"),
    ])

    # Update criteria
    for update in criteria_response.criteria_updates:
        criteria[update.index]["met"] = update.met
        criteria[update.index]["evidence"] = update.evidence

    # === STAGE REFLECTOR ===
    reflector_response = llm_with_structured_output.invoke([
        SystemMessage(content=stage_reflector_prompt),
        HumanMessage(content=f"Progress: {state['stage_implementations']}"),
    ])

    # Apply modifications (add new stages, modify existing)
    for mod in reflector_response.stage_modifications:
        if 0 <= mod.index < len(stages):
            stages[mod.index]["description"] = mod.new_description

    for new_stage in reflector_response.new_stages:
        stages.append({
            "index": len(stages),
            "title": new_stage.title,
            "description": new_stage.description,
            "completed": False,
        })

    return {
        **state,
        "high_level_stages": stages,
        "high_level_success_criteria": criteria,
        "current_stage_index": next_stage["index"],
        "orchestration_iteration": state.get("orchestration_iteration", 0) + 1,
    }

def should_continue_orchestration(state: WorkflowState) -> Literal["continue", "exit"]:
    """Check if all criteria met or max iterations reached."""
    criteria = state.get("high_level_success_criteria", [])
    iteration = state.get("orchestration_iteration", 0)

    # Exit if all criteria met
    if all(c.get("met", False) for c in criteria):
        return "exit"

    # Exit if max iterations
    if iteration >= 50:
        return "exit"

    # Check if any stages remain
    stages = state.get("high_level_stages", [])
    remaining = [s for s in stages if not s.get("completed", False)]
    if not remaining:
        return "exit"

    return "continue"

# ============================================================================
# SUMMARY NODE
# ============================================================================

def summary_node(state: WorkflowState) -> WorkflowState:
    """Generate final summary."""
    response = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(content=f"Synthesize: {state['stage_implementations']}"),
    ])

    return {
        **state,
        "final_summary": response.content,
        "messages": [response],
    }

# ============================================================================
# BUILD THE COMPLETE GRAPH
# ============================================================================

workflow = StateGraph(WorkflowState)

# === PLANNING SUBGRAPH ===
workflow.add_node("plan_maker", plan_maker_node)
workflow.add_node("plan_reviewer", plan_reviewer_node)
workflow.add_node("plan_review_confirmation", plan_review_confirmation_node)

# === PLAN PARSER ===
workflow.add_node("plan_parser", plan_parser_node)

# === STAGE ORCHESTRATION ===
workflow.add_node("stage_orchestrator", stage_orchestrator_node)

# === SUMMARY ===
workflow.add_node("summary", summary_node)

# === EDGES ===

# Entry point
workflow.set_entry_point("plan_maker")

# Planning loop
workflow.add_edge("plan_maker", "plan_reviewer")
workflow.add_edge("plan_reviewer", "plan_review_confirmation")
workflow.add_conditional_edges(
    "plan_review_confirmation",
    should_exit_planning,
    {
        "continue": "plan_maker",      # Loop back
        "exit": "plan_parser",          # Move to next phase
    }
)

# After parsing, go to orchestration
workflow.add_edge("plan_parser", "stage_orchestrator")

# Orchestration loop (stage by stage)
workflow.add_conditional_edges(
    "stage_orchestrator",
    should_continue_orchestration,
    {
        "continue": "stage_orchestrator",  # Process next stage
        "exit": "summary",                  # All done
    }
)

# Summary is the end
workflow.add_edge("summary", END)

# Compile
app = workflow.compile()
```

**You write**: ~300+ lines (and this is SIMPLIFIED!)
**LangGraph handles**: Graph execution, state passing

---

## Visual Comparison

### Google ADK Graph Structure

```
┌─────────────────────────────────────────────────────┐
│                 SequentialAgent                      │
│  (Runs sub-agents in order)                         │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ LoopAgent (high_level_planning_loop)           │ │
│  │   max_iterations=10                            │ │
│  │                                                 │ │
│  │   ┌──────────────┐      ┌──────────────┐      │ │
│  │   │ plan_maker   │  →   │ plan_reviewer│      │ │
│  │   └──────────────┘      └──────────────┘      │ │
│  │         ↑                       │              │ │
│  │         └───────────────────────┘ (loop)       │ │
│  └────────────────────────────────────────────────┘ │
│                      ↓                               │
│  ┌────────────────────────────────────────────────┐ │
│  │ LlmAgent (high_level_plan_parser)              │ │
│  └────────────────────────────────────────────────┘ │
│                      ↓                               │
│  ┌────────────────────────────────────────────────┐ │
│  │ StageOrchestratorAgent (custom)                │ │
│  │   (Complex internal graph)                     │ │
│  └────────────────────────────────────────────────┘ │
│                      ↓                               │
│  ┌────────────────────────────────────────────────┐ │
│  │ LlmAgent (summary_agent)                       │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### LangGraph Equivalent

```
START
  │
  ▼
┌──────────────┐
│ plan_maker   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ plan_reviewer    │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────┐
│ plan_review_confirmation │
└──────┬───────────────────┘
       │
       │ (conditional)
       ├─────────────┐
       │ continue    │ exit
       │             │
       ▼             ▼
  [Loop back]   ┌──────────────┐
  to plan_maker │ plan_parser  │
                └──────┬───────┘
                       │
                       ▼
                ┌─────────────────────┐
                │ stage_orchestrator  │◄───┐
                └──────┬──────────────┘    │
                       │                    │
                       │ (conditional)      │
                       ├─────────────┐      │
                       │ continue    │ exit │
                       │             │      │
                       │ [Loop back]─┘      │
                       │                    │
                       ▼                    │
                ┌──────────────┐            │
                │   summary    │            │
                └──────┬───────┘            │
                       │                    │
                       ▼                    │
                      END                   │
```

---

## Key Differences

### Abstraction Level

| Aspect | Google ADK | LangGraph |
|--------|-----------|-----------|
| **Loop syntax** | `LoopAgent(max_iterations=10)` | Manual: `add_conditional_edges() + should_continue()` |
| **State** | `session.state` (dict, automatic) | `TypedDict` (you define) |
| **Sequential flow** | `SequentialAgent(sub_agents=[...])` | Manual: `add_edge("a", "b")` for each |
| **Conditional logic** | Callbacks or custom agents | `add_conditional_edges()` + function |
| **Event streaming** | Automatic | Manual: yield from nodes |
| **Agent composition** | Hierarchical (agents in agents) | Flat graph (nodes + edges) |

### Code Comparison

**Simple planning loop**:

```python
# ADK: 3 lines
planning_loop = LoopAgent(
    sub_agents=[plan_maker, plan_reviewer],
    max_iterations=10
)

# LangGraph: ~50 lines
graph = StateGraph(PlanningState)
graph.add_node("plan_maker", plan_maker_node)
graph.add_node("plan_reviewer", plan_reviewer_node)
graph.add_edge("plan_maker", "plan_reviewer")
graph.add_conditional_edges(
    "plan_reviewer",
    should_continue,
    {"continue": "plan_maker", "exit": END}
)
```

**Complete workflow**:
- **ADK**: ~100 lines (agent definitions + composition)
- **LangGraph**: ~300+ lines (state + nodes + edges + conditionals)

---

## When You'd Use Each

### Use Google ADK when:
- ✅ Building **multi-agent hierarchies** (agents containing agents)
- ✅ You want **declarative composition** (`SequentialAgent`, `LoopAgent`)
- ✅ You need **automatic event streaming** to users
- ✅ You're okay with **component-level abstraction**

### Use LangGraph when:
- ✅ You need **complete control** over graph structure
- ✅ You want to see **every edge and conditional** explicitly
- ✅ Building **complex state machines** with many branches
- ✅ You prefer **Python-level clarity** over abstractions

---

## The NonEscalatingLoopAgent Example

This shows ADK's extensibility:

### In ADK (Extend LoopAgent)

```python
class NonEscalatingLoopAgent(LoopAgent):
    """Custom loop that suppresses escalate flags."""

    async def _run_async_impl(self, ctx: InvocationContext):
        times_looped = 0
        while not self.max_iterations or times_looped < self.max_iterations:
            for sub_agent in self.sub_agents:
                async for event in sub_agent.run_async(ctx):
                    if event.actions.escalate:
                        event.actions.escalate = False  # ← Custom!
                        return
                    yield event
            times_looped += 1
```

**~15 lines to customize loop behavior**

### In LangGraph (Modify Conditional)

```python
def should_continue_with_escalation_check(state: WorkflowState) -> str:
    """Custom conditional that checks escalation AND iteration count."""

    # Check if escalate flag set
    if state.get("escalate", False):
        # Suppress it (don't propagate)
        state["escalate"] = False
        return "exit"

    # Normal iteration check
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "exit"

    return "continue"

# Use in graph
graph.add_conditional_edges(
    "some_node",
    should_continue_with_escalation_check,
    {"continue": "loop_back", "exit": "next_phase"}
)
```

**Similar complexity, but you write the logic yourself**

---

## Mental Model Bridge

Think of Google ADK as **LangGraph with React-style components**:

```
LangGraph:  You write the graph nodes + edges explicitly
            ↓
Google ADK: You compose pre-built components (SequentialAgent, LoopAgent)
            ↓
            BUT: You can still write custom components (extend BaseAgent)
```

**LangGraph**: "Here's a graph executor, you build everything"
**Google ADK**: "Here's a component library, compose or extend"

Both give you full control, but ADK **reduces boilerplate** for common patterns (loops, sequences).

---

## Conclusion

**The planning loop**:
- ADK: 3 lines
- LangGraph: ~80 lines

**The complete workflow**:
- ADK: ~100 lines
- LangGraph: ~300+ lines (and this is simplified!)

**But**:
- LangGraph gives you **explicit visibility** into every edge
- ADK gives you **compositional abstractions** that hide boilerplate

**Neither is better** - they target different points on the control/abstraction spectrum.

For this codebase, ADK was the right choice because:
1. Hierarchical agent composition (agents in agents)
2. Built-in event streaming (user sees progress)
3. Sequential + loop patterns fit naturally
4. Extensibility when needed (custom agents)

If you needed complex branching logic with many conditionals, LangGraph might be clearer.
