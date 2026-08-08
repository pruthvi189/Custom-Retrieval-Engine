"""Agent loop implementation - ReAct style with planner, tools, observation, reflection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import api.providers as providers
import api.tools as tools

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["doc_search", "wiki_search", "web_search", "finish"]},
        "input": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "input", "reason"],
    "additionalProperties": False,
}

PLANNER_PROMPT = """You are a research agent. Your task is to answer the user's question by using tools to gather evidence.

Available tools:
{tool_descriptions}

Current state:
- Original question: {question}
- Iteration: {iteration}/{max_iterations}
- Tools used so far: {tools_used}
- Accumulated sources: {source_count}

Observations from previous steps:
{observations}

Decide your next action. Output ONLY a JSON object matching this schema:
{schema}

Rules:
- Choose ONE tool per step
- "doc_search": Search your local knowledge base
- "wiki_search": Fetch and permanently store Wikipedia articles (for established facts)
- "web_search": Search live web for current info (temporary context only)
- "finish": You have enough evidence to answer. Provide the final answer in "input".
- Be concise in "reason" - explain why this tool is the right choice now.
"""


def build_planner_prompt(
    question: str,
    iteration: int,
    max_iterations: int,
    tools_used: list[str],
    observations: list[dict],
    source_count: int,
) -> str:
    obs_text = "\n".join(
        f"  [{o['tool']}] {o['query']} -> {len(o['results'])} results, {len(o['sources'])} sources"
        for o in observations
    ) or "  (none yet)"

    return PLANNER_PROMPT.format(
        tool_descriptions=tools.get_tool_descriptions(),
        question=question,
        iteration=iteration,
        max_iterations=max_iterations,
        tools_used=", ".join(tools_used) if tools_used else "none",
        source_count=source_count,
        observations=obs_text,
        schema=json.dumps(PLANNER_SCHEMA, indent=2),
    )


# ---- Planner with retry -----------------------------------------------------------------

def call_planner(prompt: str) -> dict[str, Any]:
    """Call LLM planner, parse JSON, retry once on failure."""
    for attempt in range(2):
        response = providers.generate(prompt)
        try:
            # Extract JSON from response (handle markdown code blocks)
            # Try to find complete JSON object - non-greedy match
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
            if not json_match:
                json_match = re.search(r"\{.*?\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response)

            # Validate required fields
            if all(k in parsed for k in ("tool", "input", "reason")):
                if parsed["tool"] in ("doc_search", "wiki_search", "web_search", "finish"):
                    return parsed

        except (json.JSONDecodeError, KeyError):
            pass

        # Retry with clarification
        if attempt == 0:
            prompt += "\n\nERROR: Invalid JSON or missing fields. Output ONLY valid JSON matching the schema."

    # Fallback: default to web_search on planner failure
    return {
        "tool": "web_search",
        "input": "general information about the topic",
        "reason": "Planner failed, defaulting to web search",
    }


# ---- Agent State ------------------------------------------------------------------------

@dataclass
class AgentState:
    question: str
    max_iterations: int = 5
    iteration: int = 0
    tools_used: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)
    answer: str = ""
    finished: bool = False

    def add_observation(self, tool_result: tools.ToolResult) -> None:
        obs = tool_result.to_dict()
        self.observations.append(obs)
        self.tools_used.append(tool_result.tool)
        self.sources.extend(tool_result.sources)

        # Accumulate context for final synthesis
        for r in tool_result.results:
            if isinstance(r, dict) and r.get("content"):
                self.context.append({
                    "source": tool_result.tool,
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                })
            elif isinstance(r, dict) and r.get("title"):
                self.context.append({
                    "source": tool_result.tool,
                    "title": r.get("title"),
                    "content": str(r),
                })

    def has_evidence(self) -> bool:
        return len(self.sources) > 0 or len(self.context) > 0


# ---- Reflection Step --------------------------------------------------------------------

REFLECTION_PROMPT = """You are evaluating whether you have enough evidence to answer the question.

Question: {question}

Evidence gathered:
- Sources: {source_count}
- Context items: {context_count}
- Tools used: {tools_used}

Recent observations:
{observations}

Do you have enough evidence to provide a comprehensive, accurate answer?
Respond with ONLY: "YES" or "NO"

If NO, what specific information is missing? (One sentence)"""


def reflection_step(state: AgentState) -> bool:
    """Ask planner if evidence is sufficient. Returns True if enough, False if need more."""
    if state.iteration >= state.max_iterations:
        return True

    obs_text = "\n".join(
        f"  [{o['tool']}] {o['query']}: {len(o['results'])} results"
        for o in state.observations[-3:]
    )

    prompt = REFLECTION_PROMPT.format(
        question=state.question,
        source_count=len(state.sources),
        context_count=len(state.context),
        tools_used=", ".join(state.tools_used),
        observations=obs_text or "  (none)",
    )

    response = providers.generate(prompt).strip().upper()
    return response.startswith("YES")


# ---- Final Synthesis --------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are a research assistant. Synthesize a comprehensive answer to the user's question using ONLY the evidence provided below.

Question: {question}

Evidence:
{evidence}

Instructions:
- Use ONLY the provided evidence. Do not use external knowledge.
- Cite sources inline like [Source: Wikipedia - Physics] or [Source: Web - example.com].
- If evidence is insufficient, state what's missing.
- Be thorough but concise.
- Structure with clear sections if appropriate.

Answer:"""


def synthesize_answer(state: AgentState) -> str:
    """Generate final answer from accumulated evidence."""
    if not state.context:
        return "I couldn't find sufficient information to answer your question."

    evidence_lines = []
    for i, ctx in enumerate(state.context):
        src = ctx.get("source", "unknown")
        title = ctx.get("title", f"Item {i+1}")
        content = ctx.get("content", "")[:800]
        evidence_lines.append(f"[{i+1}] Source: {src} - {title}\n{content}")

    evidence_text = "\n\n".join(evidence_lines)
    prompt = SYNTHESIS_PROMPT.format(question=state.question, evidence=evidence_text)

    return providers.generate(prompt)


# ---- Main Agent Loop --------------------------------------------------------------------

MAX_ITERATIONS = 5


def run_agent(question: str, max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    """Run the agent loop and return structured result."""
    state = AgentState(question=question, max_iterations=max_iterations)

    # Main loop
    while state.iteration < state.max_iterations and not state.finished:
        state.iteration += 1

        # Planner decides next action
        prompt = build_planner_prompt(
            question=state.question,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
            tools_used=state.tools_used,
            observations=state.observations,
            source_count=len(state.sources),
        )
        plan = call_planner(prompt)

        tool_name = plan["tool"]
        tool_input = plan["input"]
        reason = plan.get("reason", "")

        if tool_name == "finish":
            state.answer = tool_input
            state.finished = True
            break

        # Execute tool
        tool_result = tools.execute_tool(tool_name, query=tool_input)
        state.add_observation(tool_result)

        # Check if tool failed
        if tool_result.error:
            # Continue with other tools instead of crashing
            pass

    # Reflection step (once)
    if not state.finished and state.has_evidence():
        if not reflection_step(state):
            # One additional search attempt
            state.iteration += 1
            tool_result = tools.execute_tool("web_search", query=state.question)
            state.add_observation(tool_result)

    # Final synthesis
    if not state.answer:
        state.answer = synthesize_answer(state)

    # Build response
    return {
        "answer": state.answer,
        "iterations": state.iteration,
        "steps": [
            {
                "iteration": i + 1,
                "tool": obs["tool"],
                "query": obs["query"],
                "results_count": len(obs["results"]),
                "sources_count": len(obs["sources"]),
                "error": obs.get("error"),
            }
            for i, obs in enumerate(state.observations)
        ],
        "sources": state.sources,
        "context": state.context,
    }