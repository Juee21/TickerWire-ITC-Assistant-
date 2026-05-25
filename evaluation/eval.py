
import sys
if sys.platform == "win32":
    import asyncio as _asyncio_pre
    _asyncio_pre.set_event_loop_policy(_asyncio_pre.WindowsSelectorEventLoopPolicy())
# ─────────────────────────────────────────────────────────────────────

import asyncio
import uuid
import time as time_mod          # aliased — never shadowed by asyncio internals
from typing import Dict, Any, List
from agent.router import compiled_agent_application


def _make_input(query: str) -> Dict[str, Any]:
    """
    Builds a fully-populated AgentGraphState input dict.
    Centralised so adding a new required state field means one change.
    """
    return {
        "query":             query,
        "original_query":    query,
        "trace_id":          f"EVAL_{uuid.uuid4().hex[:8]}",
        "loop_count":        0,
        "max_loops":         2,
        "cost_accumulated":  0.0,
        "detected_fy":       None,
        "router_decision":   None,
        "retrieved_context": None,
        "final_output":      None,
        "evaluation_score":  0.0,
    }


# ── Test scenario definitions ─────────────────────────────────────────
SCENARIOS: List[Dict[str, Any]] = [
    {
        "name":        "1. Normal FY23 retrieval",
        "query":       "What happened with Agri Business segment revenue in FY23?",
        "expected":    ["ANSWER_WITH_CITATION"],
        "description": "Happy path — retrieve, synthesize, exit cleanly.",
    },
    {
        "name":        "2. Multi-year comparison",
        "query":       "Compare ITC FMCG segment revenue in FY22 and FY24",
        "expected":    ["ANSWER_WITH_CITATION"],
        "description": "Multi-year — detected_fy should contain both FY22 and FY24.",
    },
    {
        "name":        "3. Missing fiscal year (clarification path)",
        "query":       "What was ITC's revenue from operations?",
        "expected":    ["ASK_CLARIFYING_QUESTION"],
        "description": "No FY in query — should route to ClarificationGenerator.",
    },
    {
        "name":        "4. Banned company (refusal guardrail)",
        "query":       "Compare ITC revenue with Reliance in FY24",
        "expected":    ["REFUSE"],
        "description": "Banned entity — should route to RefusalGenerator.",
    },
]
# ─────────────────────────────────────────────────────────────────────


async def run_system_evaluation_suite() -> None:
    print("=" * 60)
    print("🚀 TickerWire Evaluation Harness")
    print("=" * 60)

    results = []
    suite_start = time_mod.perf_counter()    # FIX: time_mod alias

    for scenario in SCENARIOS:
        print(f"\n▶  Running : {scenario['name']}")
        print(f"   Query   : {scenario['query']}")

        t0 = time_mod.perf_counter()         # FIX: time_mod alias
        try:
            state_output = await compiled_agent_application.ainvoke(
                _make_input(scenario["query"])
            )
            elapsed  = time_mod.perf_counter() - t0   # FIX: time_mod alias
            output   = state_output.get("final_output") or {}
            decision = output.get("decision", "NO_DECISION")
            passed   = decision in scenario["expected"]

            print(f"   Decision : {decision}")
            print(f"   Time     : {elapsed:.2f}s")
            print(f"   Status   : {'✅ PASS' if passed else '❌ FAIL'}")
            if not passed:
                print(f"   Expected : {scenario['expected']}")

            answer_preview = str(output.get("answer", ""))[:200].replace("\n", " ")
            if answer_preview:
                print(f"   Preview  : {answer_preview}...")

            results.append({
                "name":     scenario["name"],
                "passed":   passed,
                "elapsed":  elapsed,
                "decision": decision,
            })

        except Exception as exc:
            elapsed = time_mod.perf_counter() - t0    # FIX: time_mod alias
            print(f"   💥 EXCEPTION after {elapsed:.2f}s: {exc}")
            results.append({
                "name":     scenario["name"],
                "passed":   False,
                "elapsed":  elapsed,
                "decision": f"EXCEPTION: {exc}",
            })

    # ── Summary report ────────────────────────────────────────────────
    total_elapsed = time_mod.perf_counter() - suite_start   # FIX: time_mod alias
    passed_count  = sum(1 for r in results if r["passed"])
    total_count   = len(results)

    print("\n" + "=" * 60)
    print(f"📊 SUMMARY  ({passed_count}/{total_count} passed, {total_elapsed:.2f}s total)")
    print("=" * 60)
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon}  {r['name']:<42} {r['elapsed']:.2f}s  →  {r['decision']}")

    assert passed_count == total_count, (
        f"Evaluation failed: {total_count - passed_count} scenario(s) did not pass."
    )
    print("\n✅ All scenarios passed. System is stable and production-ready.")


if __name__ == "__main__":
    asyncio.run(run_system_evaluation_suite())