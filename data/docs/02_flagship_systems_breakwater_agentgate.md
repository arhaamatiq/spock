# Flagship Engineering — BreakWater & AgentGate

*Two distinct systems. One recurring question: what does it take to trust an AI system in the real world?*

## Why these projects belong together

A modern AI stack has at least two places where things can go wrong. First, a model can initiate an action that was never authorized. Second, an agent or model can be evaluated by a grader—or an execution system—that reports a result it has not genuinely earned. Arhaam built **AgentGate** to address the first boundary and **BreakWater** to investigate the second.

These are not two names for the same project, and neither is merely a wrapper around an LLM. They are independent systems demonstrating different but complementary engineering strengths. AgentGate shows execution-level authorization, software integration, and empirical security testing. BreakWater shows distributed coordination, recovery under failures, evaluation science, and contributions to existing open-source benchmark infrastructure.

---

## BreakWater — Don't just run the benchmark. Test the thing deciding whether it passed.

**Project:** https://github.com/arhaamatiq/Breakwater  
**Portfolio case study:** https://arhaamatiq.framer.website/projects/breakwater  
**Area:** distributed systems, AI evaluation infrastructure, adversarial testing, fault tolerance

### The problem it notices

An AI benchmark returns a simple verdict: PASS or FAIL. Researchers use that result to compare models, choose systems for deployment, and measure progress. But a pass is an assertion by an evaluator; it is not proof that a solution was genuinely correct. Tests can miss edge cases. A grader can use the wrong contract. The environment can leak information or vary across runs. A distributed runner can lose an experiment, run it twice, or accidentally count a stale result.

BreakWater was built around a more fundamental question: **can the system find evidence that a benchmark's verdict is wrong?**

It does not approach a verifier as an unquestionable source of truth. It treats the verifier as another piece of software that needs to be tested, challenged, and understood.

### What he built

BreakWater combines an adversarial verifier-auditing engine with a distributed trial-execution platform. The auditing side tries deliberately wrong or subtly modified solutions, repeats verdicts, varies environmental conditions, and searches for counterexamples. The execution side runs isolated trials through concurrent Docker workers, PostgreSQL job coordination, and S3-compatible artifact storage, retaining a record of what actually happened.

The architecture matters because the test infrastructure itself must not become a source of misleading evidence. Jobs are leased to workers; heartbeats maintain ownership; expired leases can be reclaimed after a crash or stall; idempotent result recording prevents a retried trial from being counted twice. Attempt-level artifacts preserve the story even when an execution is interrupted.

He did not stop at a diagram of these guarantees. He deliberately crashed and paused the workers during a run.

### The failure test that makes the project memorable

In a local fault-injection test, BreakWater ran **96 Docker-isolated logical trials** over four workers with three concurrent trials each. One worker was killed; another was paused long enough for its leases to expire. The system initiated **102 executions** for those 96 logical trials, requeued **six interrupted trials**, and recovered all six on other workers. A stale worker later resumed and attempted to submit results that had already been handled; the ledger suppressed **three duplicate submissions**.

The final database checks showed:

- **96 / 96 logical trials with accepted results**
- **0 lost results**
- **0 duplicate result rows**
- **6 trials successfully recovered after lease expiry**

The compelling point is not simply the number 96. It is that the conditions deliberately violated the happy path, and the final claim was verified against recorded state. The project's core execution principle is **at-least-once execution with exactly-once result effect**: physical work can be retried, but a logical trial contributes one accepted outcome.

### The verifier is on trial too

BreakWater tests graders by constructing suspicious candidate solutions: no-ops, constants, echoes, exceptions, and small mutations of otherwise-correct programs. If an altered program still passes, it tries to find a concrete input where the altered and correct programs disagree. That creates a more persuasive finding than saying a test suite is probably weak: it produces an example that can be rerun.

The project also probes whether a result changes when the same output is verified again, whether changing execution environment variables changes the verdict, and whether test-case order or shared state causes a different answer. A finding becomes a **replayable falsification receipt** with enough information to independently inspect it.

On its planted-flaw evaluation suite, the tester detected **7 of 7** deliberately introduced verifier weaknesses with their intended failure classifications, generated **9 of 9** receipts that replay-confirmed, and produced **no CRITICAL/HIGH false positives on the 10 sound test tasks** in that controlled suite. These are bounded test results, not a claim of flawless detection everywhere.

### Applying it to benchmarks people actually use

BreakWater's adapter layer separates its general auditing methods from benchmark-specific execution. It can evaluate Python function behavior in HumanEval and machine-state outcomes in Terminal-Bench/Harbor tasks without turning each benchmark into an entirely new project.

In HumanEval, the audit converted **146 of 164** problems into its internal representation and independently reproduced **five known shallow-test weaknesses** from observed behavior. Some tests overlooked important boundary cases; others accepted truthy values without strictly enforcing Boolean return types. Those are not claims of new, unfixed vulnerabilities across every modern variant. They are evidence that the falsification methods could discover known classes of weakness without being handed the answer.

The work also reached **Terminal-Bench's live verifier infrastructure**. An audit of `risk-scorer-replay` exposed a transient SQLite journal-file race: a verifier could attempt to inspect a file that had disappeared during database reconstruction, producing an intermittent failure even for the reference solution. Arhaam reproduced the issue and contributed an upstream fix. He also submitted a separate correction concerning the jump-transition margin in `biped-contact-dynamics`, where grading did not match the task's specified margin.

This is an especially strong signal for recruiters: the work did not end at a self-contained demo. Studying a real open-source system produced concrete reliability improvements for that system.

### Why BreakWater is a serious engineering signal

BreakWater combines concerns that are often discussed separately: distributed job correctness, adversarial software testing, provenance, reproducibility, and benchmark validity. It shows that Arhaam can reason about a system as a chain of claims. An experiment needs correct execution; an accepted result needs correct deduplication; a PASS needs a sound grader; a finding needs a replayable witness. A failure at any link can invalidate the headline number.

**Good 20-second explanation:** “I built BreakWater because AI benchmark scores depend on graders we usually treat as ground truth. It attacks the graders with adversarial probes, then runs those tests on infrastructure designed to survive worker failures. In a 96-trial chaos run it recovered interrupted work with no lost or duplicate results, and its audits uncovered real issues in benchmark verification.”

**What this demonstrates:** distributed state machines, concurrency, isolation, fault injection, careful handling of retries, open-source collaboration, and unusual attention to the integrity of evaluation results.

---

## AgentGate — A boundary between an agent's intention and its actions

**Project:** https://github.com/arhaamatiq/agentgate  
**Package:** https://pypi.org/project/agentgate-py/  
**Portfolio case study:** https://arhaamatiq.framer.website/projects/agentgate  
**Area:** AI agent security, tool-call interception, developer infrastructure, adversarial evaluations

### The problem it notices

For an assistant that only answers questions, checking the final response can be useful. For an agent with real tools, it is not enough. An agent could generate an innocuous message while trying to read a credential file, delete a database table, or transmit information to an unauthorized destination. The mistake is to judge the sentence after the side effect rather than the operation before it happens.

AgentGate moves the trust boundary into execution. It intercepts an agent's actual tool call—including the intended operation and arguments—before the underlying tool runs, then decides whether that action is permitted.

### The design decision that defines the product

AgentGate is based on **declared permission rather than an endless blacklist**. A developer specifies the task's allowed operations and resources. Actions outside that scope are rejected even if the system has never seen their exact wording before.

It also uses a deliberate two-tier decision model. Concrete cases—SQL operations, file paths, HTTP targets, resource boundaries, destructive commands—can often be checked deterministically. More ambiguous actions are escalated to a separate language-model reviewer that considers context and risk. In its measured local evaluation, the deterministic path averaged **0.3 milliseconds**; roughly **24%** of actions required semantic escalation in the documented run. The system is designed to fail closed when an ambiguous action cannot be reliably reviewed.

This is a practical tradeoff: not every harmless operation should wait for a second LLM call, but permission checks should not be reduced to brittle string matching either.

### How he tested whether it worked

Arhaam built an adversarial evaluation suite covering data access, filesystem operations, communications, infrastructure actions, subtle scope violations, and multi-step attack sequences. The public evaluation reports **370 adversarial evaluations across 180 scenarios**. Iterating on real misses and false alarms raised measured recall from **67.3% in the initial run** to **99.5% recall and 97.7% precision in Run 007**.

Multi-turn behavior matters because an individually acceptable action can become dangerous in sequence. Reading data might be legitimate. Sending an email might be legitimate. Collecting sensitive information and forwarding it to a new destination tells a different story. AgentGate evaluates action histories rather than judging every operation as if it happened in a vacuum; the published case study reports **96.0% step-level trajectory accuracy (121/126)** on its multi-turn adversarial test.

Those numbers describe its documented test suites, not a guarantee that every attack in the world can be blocked. What makes the engineering notable is that the library has an evaluation harness built to surface its own failure modes.

### Built for adoption, not just demonstration

AgentGate is published as the open-source Python package **`agentgate-py`**. It integrates with the OpenAI SDK and LangChain/LangGraph tool execution paths and can also protect individual functions. It supports scoped policy configuration and structured decision logs, including optional asynchronous Supabase audit logging. The goal is to put an enforceable boundary around an existing agent without forcing a developer to rebuild the entire application around a security prototype.

It also expresses a useful product instinct: a guardrail is more likely to be used when its interface is simple, its fast path is genuinely fast, and its decisions leave a trace.

**Good 20-second explanation:** “AgentGate is a Python firewall for AI agents. Instead of just checking what the model says, it intercepts the tool calls the agent is about to execute and checks them against permitted scope. I published it on PyPI, built integrations for common agent frameworks, and tested it against hundreds of adversarial evaluations rather than relying on a few demo attacks.”

**What this demonstrates:** security boundary design, SDK and framework integration, deterministic/semantic tradeoffs, developer experience, profiling, adversarial testing, and packaging an idea as reusable infrastructure.

---

## The deeper connection

AgentGate asks: **Should this agent action be allowed to happen?**  
BreakWater asks: **If something happened, can the system truthfully report what happened—and can we trust the test judging it?**

Both projects resist a common temptation in AI development: trusting the visible output too much. In one case, the visible output is an agent's message. In the other, it is a benchmark's PASS. Arhaam designs around the hidden mechanism, introduces failure deliberately, measures behavior, and packages evidence so others can verify the result.

**A strong, fair summary for an interviewer:** Arhaam has built real software around two critical bottlenecks in reliable AI: constrained action and credible evaluation. The work combines hands-on systems engineering with a researcher's habit of asking what would falsify the claim.
