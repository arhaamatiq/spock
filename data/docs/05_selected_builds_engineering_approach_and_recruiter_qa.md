# Selected Builds, Engineering Approach & Recruiter Conversations

*How Arhaam thinks beyond his flagships—and how to understand the value he would bring to an engineering team.*

## The larger portfolio is intentional

AgentGate, BreakWater, and LLM unlearning are the strongest proof of Arhaam's current technical direction. But they are not the whole picture. He has also built tools for multi-source AI research, practical RAG applications, developer-facing interfaces, and student-team workflows.

These projects add important range. They show he can work not just on low-level reliability or research, but on the product around the engineering: defining an information problem, orchestrating multiple inputs, exposing useful explanations, and building an interface someone can use.

## HireSignal — turning scattered hiring activity into structured intelligence

**Repository:** https://github.com/arhaamatiq/hire-signal  
**Area:** agent orchestration, information retrieval, asynchronous workflows, explainability

The initial insight behind HireSignal is simple: organizations often reveal their technical direction through the people they are recruiting. A sudden cluster of model-inference roles suggests one set of priorities; a pattern of alignment or evaluation roles suggests another. Those signals are public, but scattered across job boards, funding reports, and research publications.

Arhaam built HireSignal to pull those fragments into an interpretable view. It researches AI companies across hiring data, public news, and academic activity, then produces a ranked intelligence brief about what a company appears to be building and where its hiring priorities may be moving.

The system uses a **seven-node LangGraph workflow**. A coordinator dynamically fans out research across companies, parallel workers fetch available sources, downstream steps extract skills and themes, deterministic scoring ranks signals, and LLM-generated narratives synthesize what the evidence suggests. A FastAPI endpoint and Streamlit interface make the results accessible beyond the workflow itself.

One of the most important design choices is that the **0–100 intent score is explainable**. Its component signals are exposed rather than being left as an opaque number authored by a language model. The LLM interprets and narrates the evidence, while the ranking has a defined scoring structure. The system also incorporates source-level retry behavior and structured logs, recognizing that information-gathering pipelines need to degrade gracefully when a source fails.

Why this is interesting to a technical recruiter: HireSignal demonstrates stateful orchestration, parallel fan-out, multi-source ingestion, structured synthesis, and an understanding that an answer is more useful when a person can inspect its basis.

Why this is interesting to a product engineer: it turns information overload into a specific decision-support workflow. The purpose is not “use agents because agents are fashionable.” It is “take a research task that requires several kinds of evidence and make its result consistent, inspectable, and easy to consume.”

**Good short answer:** “HireSignal uses LangGraph to research several AI companies in parallel, combining hiring, news, and research signals into transparent intent scores and intelligence briefs. I kept the scoring deterministic and the explanation layer separate, so users can see why a company was ranked a certain way.”

The score is designed as an interpretable **research-prioritization signal**: it tells users what evidence drove a ranking and gives them a better place to investigate.

## DiscoverAI — from a business-development problem to an interactive AI product

**Repository:** https://github.com/arhaamatiq/discover-ai  
**Area:** AI-assisted research, async backend engineering, streaming interfaces, internal tools

DiscoverAI addresses a related but more operational problem. Before an engineering organization approaches a potential partner, it needs a convincing answer to “what could we actually build for this company, and why should they care?” A generic list of AI ideas is not a useful answer.

The product researches a company through multiple public signals and turns those findings into grounded project opportunities. Its backend performs parallel asynchronous research; the output is synthesized into an interactive ideation experience using streaming updates and a visual canvas. Instead of forcing users to wait for one large response, the interface makes ideas visible as they arrive and organizes their relationships.

The human factors are as important as the architecture. The tool was designed for an organization in which the people who know what makes a persuasive outreach idea are not necessarily the people writing the application. Editable guidance allows non-engineers to shape the criteria used for generated proposals.

That choice communicates a useful engineering instinct: **the best abstraction is sometimes an organizational one**. Give the right people control over the part they understand, and let software take care of the repetitive mechanics.

DiscoverAI also expresses Arhaam's interest in AI interfaces that feel native to the user's task. Not every intelligent product should be a chat window. Sometimes a canvas, a live stream, a research brief, or a structured decision record better fits how a human actually works.

**Good short answer:** “DiscoverAI is an internal research and ideation tool for Applied Engineering. It brings together company signals, uses AI to develop grounded project ideas, and streams them onto a visual canvas. The goal is to turn vague brainstorming into evidence-backed opportunities that a team can actually pitch and build.”

## Spock AI — making a portfolio conversational without losing its evidence

**Repository:** https://github.com/arhaamatiq/spock  
**Experience:** https://arhaamatiq.com/spock-ai  
**Area:** RAG, document retrieval, chat interfaces, backend integration

Spock AI is the conversational assistant embedded into Arhaam's portfolio. Its job is to help a visitor explore his work by asking the questions they actually care about: how BreakWater handles failures, what AgentGate protects, what he discovered in his research, what he led at Applied Engineering, and where to find more evidence.

Its implementation uses a document-based retrieval-augmented generation pipeline rather than expecting a general-purpose model to know the person behind the portfolio. It includes **ChromaDB-backed retrieval, persistent indexed knowledge, history-aware question handling, streaming generation, and an API layer**. The source knowledge is organized into curated files that can be updated and re-ingested as the portfolio evolves.

The project is a small but expressive example of a larger product belief: useful intelligence should be available in the interface that makes sense for the task. A personal portfolio is often a collection of links and summaries. Spock turns it into a guided conversation while keeping the underlying claims anchored to the work.

Spock is also a demonstration of taste. The best answer is not the longest, the most technical, or the most flattering; it is the answer that gets a recruiter or engineer quickly to a compelling, defensible detail and a place to explore further.

**Good short answer:** “Spock is the RAG-powered assistant on Arhaam's portfolio. Instead of making visitors hunt through project pages, it lets them ask questions about his work and pulls the relevant context into the conversation.”

## The practical RAG work sits alongside the research tools

The **ScoreGetter RAG customer-service agent**, delivered through Applied Engineering, adds an important dimension that should not be lost beneath the flashier research and infrastructure projects. It involved a real client, a three-developer team, Docker and GitHub Actions deployment workflows, and evaluation with RAGAS. On the documented 40-question synthetic set, the system measured **0.83 faithfulness, 0.81 answer relevancy, and 0.89 context recall**.

Spock and ScoreGetter are not the same project. The former is a personal portfolio retrieval experience. The latter is a client-facing customer-service system delivered through Applied Engineering. Together they show both the ability to engineer RAG pipelines and the sense to evaluate grounding and retrieval rather than merely celebrating fluent answers.

## The philosophy across his software

### 1. The system must be able to explain its outcome

Arhaam consistently favors outputs that carry usable evidence. AgentGate produces authorization decisions and action logs. BreakWater produces replayable verifier findings and trial provenance. HireSignal exposes score components. His research distinguishes multiple causes behind one recovery metric. This is not decoration around the answer: it is part of what makes the answer useful.

### 2. The happy path is where testing begins, not where it ends

He deliberately tests stale workers, repeated executions, incorrect programs, ambiguous permissions, failed tool calls, and misleading research metrics. The point is not pessimism. A system that survives difficult but plausible conditions is more valuable than one that performs beautifully on a scripted demo.

### 3. Deterministic software and model judgment should do different jobs

An LLM is useful when meaning must be interpreted. It should not replace ordinary mechanisms that are cheaper, more stable, and easier to audit. AgentGate uses direct checks before escalating ambiguity to a model. HireSignal separates deterministic scoring from LLM-written narrative. The same instinct appears in his research: statistical analysis and controls should constrain interpretation.

### 4. First principles beat inherited assumptions

If an agent is making decisions, Arhaam asks where permission is actually enforced. If a benchmark scores agents, he asks whether the grader deserves authority. If a model “recovers” knowledge, he asks how the experiment distinguishes knowledge from verbal output. These are practical framing choices that redirect implementation toward the actual source of uncertainty.

### 5. Usability is an engineering concern

Some of his strongest work is deep infrastructure, but his view of engineering is not limited to backends and benchmark harnesses. An installable Python package, a clear research narrative, a visual canvas, a company-intelligence brief, or a conversational portfolio can determine whether someone understands or adopts the system. He cares about both the invisible mechanism and the visible experience.

### 6. Build → test → revise → communicate

This cycle appears in his earliest robotics projects, his UL Solutions workflow improvements, his student-team client delivery, and his research. The sophistication of the tools changed; the discipline of producing something tangible, learning from it, and explaining the result did not.

## Strengths and technical range, without empty buzzwords

**Languages and development:** Python; TypeScript and JavaScript; SQL; Bash; Git-based development and collaborative software workflows.

**Systems and backends:** FastAPI, Docker, PostgreSQL, asynchronous/concurrent execution, distributed job coordination, failure recovery, service interfaces, CI/CD, Linux, object storage through S3-compatible APIs.

**Agent and application engineering:** LangGraph, LangChain, OpenAI-compatible model interfaces, tool-call interception, RAG, ChromaDB, Supabase, streaming experiences, React and Next.js.

**ML and experimentation:** PyTorch, CUDA, Hugging Face Transformers, open-weight LLM evaluation, relearning/unlearning workflows, adversarial testing, reproducible experiments, measurement controls, RAGAS.

These categories describe areas of demonstrated project experience, not equal depth in every technology. His standout strengths are how the parts come together into a system and how he tests its behavior.

## Recruiter and engineering Q&A

### “Tell me about yourself.”

Arhaam is an SJSU Applied Mathematics & Computational Science student focused on reliable AI systems. He builds across distributed infrastructure, agent safety, and empirical ML research. His work includes AgentGate, an open-source agent firewall on PyPI; BreakWater, a fault-tolerant AI evaluation and verifier-auditing platform; research on how LLM unlearning is measured; a software engineering internship at UL Solutions; and founding a 65+ member student engineering organization that delivers software.

### “What's the single most impressive technical project?”

For distributed-systems teams, start with **BreakWater**: worker leases, controlled crash recovery, idempotent results, and adversarial benchmark validation. For agent-infrastructure teams, start with **AgentGate**: pre-execution authorization, fast deterministic checks, and hundreds of adversarial evaluations. For ML research teams, start with the **unlearning measurement audit** and the reproduction of published baselines. The answer should match what the interviewer cares about rather than pretending every role values exactly the same proof.

### “What's something you built that other people depended on?”

At UL Solutions, he improved testing pipelines used in operational workflows. Through Applied Engineering, he led a three-developer team to deliver a client RAG agent. He also publishes reusable developer infrastructure through AgentGate and has contributed a benchmark-verifier fix upstream through the BreakWater investigation.

### “Where have you worked with real data or real measurements?”

His UL work had measured improvements in setup time and throughput. AgentGate has a documented adversarial evaluation program. BreakWater reports database-checked crash recovery and verifier test results. The ScoreGetter RAG agent was assessed through RAGAS on a stated dataset. His unlearning research relies on controlled experiments, reproduced baselines, and transparent distinctions between competing interpretations of a metric.

### “How do you approach a problem you're unfamiliar with?”

He starts by identifying the object the system is supposed to produce and what would make that outcome invalid. Then he separates what can be checked deterministically from what requires empirical investigation or judgment. He prefers to get a baseline running, understand failure modes, build an instrument for observing them, and iterate against evidence rather than intuition alone.

### “What has leadership taught him?”

That technical quality depends on more than individual ability. On a team, ambiguity around scope, expectations, interfaces, and ownership becomes engineering risk. Recruiting people is only the first step; the harder work is creating conditions in which they can make and deliver something together.

### “What is he especially interested in learning next?”

Deeper ML systems and reliable agent architectures: evaluation under adversarial conditions, the interfaces between models and tools, scalable execution, and how local-versus-global objectives affect multi-agent exploration. His current foundation includes both hands-on infrastructure and empirical ML work, with a particular interest in roles that demand both.

### “Why would he fit on a high-caliber engineering team?”

Because he has demonstrated the combination that hard engineering work rewards: the willingness to implement, the habit of testing assumptions, the patience to trace elusive failures, and the ability to explain what is actually proven. He has produced reusable open-source software, investigated real benchmark flaws, worked in industry, delivered a client system with a team, and pursued ML research that challenges superficial interpretations of results.

### “Where should I start if I only have five minutes?”

Visit **https://arhaamatiq.com**. Open **BreakWater** if you want evidence of systems engineering and open-source benchmark work. Open **AgentGate** if you work with tools and autonomous agents. Open **Tracing What Models Forget** if you're evaluating his research judgment. Read the **Applied Engineering** experience for leadership and end-to-end delivery.

## A closing portrait

Arhaam is most interesting at the boundary between *ambition* and *proof*. He is attracted to ambitious, frontier-facing problems, but his strongest work has not been about claiming that everything is solved. It has been about constructing the system, identifying where trust breaks, and gathering evidence strong enough to explain what really happened.

That's the common thread from early robotics to client software, from Python test pipelines to agent firewalls, from Docker worker failures to LLM unlearning experiments. He wants to build important things. He also wants to know—precisely, reproducibly, and without hand-waving—whether they work.

**One-line takeaway:** A builder's instinct, a systems engineer's discipline, and a researcher's respect for evidence.
