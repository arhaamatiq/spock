# Experience, Leadership & Shipping Real Software

*Professional experience is not separate from Arhaam's technical identity. It is where technical judgment meets deadlines, stakeholders, messy systems, and accountability.*

## UL Solutions — Software Engineering Intern

**Company:** UL Solutions · San José, California  
**Period:** Summer 2025  
**Focus:** Python data pipelines, RF testing workflows, automation, reliability of validation

Arhaam's internship at UL Solutions was not about inventing a toy feature. It was about improving an engineering workflow that already existed, with real constraints and teams depending on its outputs.

The environment involved RF test pipelines across many device categories, along with the familiar reality of internal engineering software: legacy modules, repetitive setup, inconsistent data validation, and processes whose friction compounds over many test runs.

Arhaam refactored Python data-processing components used across **20+ device types**. In the measured workflow, test setup dropped from **60 seconds to 40 seconds**, a **33% reduction in setup time**, with approximately **25% higher throughput**. He added validation schemas that reduced review cycles **from three rounds to one** and helped extend automation coverage from **12 to more than 20 device categories**.

Each of those numbers describes a different engineering contribution. Less setup time means fewer repetitive costs per run. Higher throughput means teams can get through more validation work. Earlier schema checks mean less back-and-forth after bad data reaches reviewers. Broader coverage means less specialized manual handling for each device family.

The experience shaped his standards for software: correctness is not only “the program runs.” Correctness includes consistent data, useful documentation, repeatable workflows, and changes that colleagues can use without having the original author standing over their shoulder.

**Recruiter takeaway:** He has worked inside an established engineering organization, improved systems already used by others, and measured the operational effect instead of merely reporting that he “automated a process.”

**Good short answer:** “At UL Solutions, Arhaam improved Python pipelines for RF testing workflows across more than twenty device categories. He reduced setup time by a third, increased throughput by a quarter, and redesigned validation so reviews could go from three rounds to one.”

## Applied Engineering at SJSU — Founder & President

**Organization:** Applied Engineering · San José State University  
**Founded:** September 2025  
**Scale:** 65+ members  
**Site:** https://applied-engineering-sjsu.vercel.app/

Applied Engineering began with an insight about the gap between learning software engineering and being trusted to deliver it. Students need real constraints, collaboration, and shipped work to become strong engineers. Companies and organizations have genuine software problems and ideas that do not always fit their internal engineering priorities. Arhaam founded Applied Engineering to connect those needs.

The organization operates as a **student-run software agency** rather than simply a community for passive attendance. Members have a path from recruiting and onboarding into project work. Client engagements need discovery, scope control, implementation, technical review, communication, and handoff. The outcome is not just a project bullet point; it is experience turning someone else's requirements into something usable.

Growing Applied Engineering to **65+ members** required more than attracting people to a club. Arhaam developed the operational side: member selection, onboarding, engineering-team formation, project scoping, stakeholder communication, and a delivery pipeline. This is a concrete leadership differentiator because he has had to make systems work *through other people*.

His engineering and organizational skills reinforce each other. He understands how architecture decisions affect delivery, but also how a good technical idea can fail because a team lacks clear ownership, a shared understanding of the outcome, or a credible way to evaluate success.

### ScoreGetter — taking a client RAG agent from scope to launch

One of Applied Engineering's concrete delivery examples is a RAG-based customer-service agent built for **ScoreGetter**. Arhaam led a team of **three developers**, overseeing architecture, sprint planning, and stakeholder communication through delivery.

The system combined **LangChain, ChromaDB, and GPT-4o** for retrieval-grounded responses. Docker deployment and GitHub Actions automation brought operational repeatability to a system that would otherwise be easy to leave as a local prototype. Critically, the team also built evaluation into the delivery workflow rather than relying only on a demonstration that produced sensible-sounding answers.

On a **40-question synthetic evaluation dataset**, the system measured:

- **0.83 faithfulness** — assessing grounding of answers in supporting context
- **0.81 answer relevancy** — assessing whether responses addressed the question
- **0.89 context recall** — assessing whether retrieval found needed evidence

The evaluation was a test of the documented dataset and pipeline, not a universal accuracy guarantee for every future user question. That distinction is part of why it is meaningful: Arhaam knows how to report a concrete measured result without pretending an LLM application is infallible.

The client-engagement story is stronger than “he made a chatbot.” He led the team, helped translate requirements into architecture, incorporated operational deployment and testing, and supported a real handoff. It is evidence of end-to-end ownership.

**Good short answer:** “Through Applied Engineering, Arhaam led three engineers delivering ScoreGetter's retrieval-grounded customer service agent. The team shipped the deployment and CI workflow and evaluated the RAG pipeline with measured faithfulness and retrieval metrics.”

### DiscoverAI — a tool born out of the organization's own workflow

The agency also created a distinctive internal tool: **DiscoverAI**, an AI-assisted platform for researching organizations and developing grounded project opportunities.

A business-development pitch can be generic and forgettable. DiscoverAI is designed to start with evidence: recent public signals about a company, the products it makes, the problems it may be solving, and concrete reasons a particular engineering project could matter to it.

Its implementation connects parallel asynchronous research with LLM synthesis, streaming results to a visual idea canvas built with React/Next.js. The project's architecture uses a FastAPI backend and tools including Tavily, LangChain, and a database-backed application layer. One notable product decision was separating prompt guidance into editable instructions so non-engineering collaborators could influence the quality and framing of project ideas without changing application code.

DiscoverAI demonstrates a different but complementary capability from BreakWater or AgentGate. It is **workflow product design**: identifying a repetitive team problem, making information easier to act on, and giving users a more intuitive interface than a single chat box.

Its public portfolio description emphasizes multi-model research, asynchronous agents, streaming output, and a visual research/ideation experience. Avoid transferring AgentGate's security metrics to DiscoverAI; those metrics measure a different project.

### Broader organizational work

Another initiative under exploration through Applied Engineering is an **MCP-powered AI infrastructure deployment planner** that reasons across interdependent constraints: compute, power, cooling, permitting, interconnect, latency, geography, and on-site generation. It reflects his interest in ambitious, real-world decisions where technical feasibility, operational limitations, and product usefulness have to be considered together.

## A practical leadership philosophy

Arhaam does not view leadership as being the loudest person in the room. The useful version of leadership is reducing ambiguity for everyone else and maintaining responsibility for the final outcome.

That has several implications in his work:

**He likes clear ownership.** Projects move when someone is accountable for each piece and understands the relationship between their task and the overall goal.

**He wants technical work to survive handoff.** Repeatable environments, version-controlled code, documented procedures, and evaluation artifacts matter because a system should not collapse when its original builder steps away.

**He works from outcomes backward.** Before selecting a framework, the team should understand the person using the product and the change the product is supposed to make.

**He values communication as a technical multiplier.** A precise problem statement, a shared interface, and an honest report of a failing experiment can save more time than a clever implementation delivered too late.

**He does not confuse activity with shipping.** The useful questions are whether the system solves the intended problem, what evidence supports that claim, and whether another engineer can continue the work.

## The connection between internship, organization, and personal projects

At **UL Solutions**, he improved operational software inside an established testing environment. At **Applied Engineering**, he created the structure and led a team capable of delivering software for an external stakeholder. In **AgentGate**, he built an open-source developer tool and exposed it to adversarial evaluation. In **BreakWater**, he implemented failure recovery and challenged benchmark infrastructure. In **ML research**, he reproduced existing work and scrutinized what its measurements meant.

These experiences reinforce one another. They show a progression from improving a workflow, to organizing people who ship workflows, to engineering the infrastructure and methodologies that let increasingly intelligent systems be evaluated and trusted.

For a recruiting team, that breadth is not scattered experimentation. It is evidence that Arhaam can contribute at multiple layers of a software organization: implementation, reliability, research, product judgment, and collaboration.

## Answering common questions

**Does he have real industry experience?** Yes. He was a Software Engineering Intern at UL Solutions in 2025 and delivered measured improvements to Python testing and validation workflows.

**Has he managed engineers?** Yes. As founder and president of Applied Engineering he has recruited and coordinated a 65+ member organization, and he led a three-developer client project from scoping to delivery. This is student-organization and project leadership, not a claim of managing a corporate engineering department.

**Has his work served users beyond himself?** Yes. The ScoreGetter agent was an external client engagement, DiscoverAI was built around Applied Engineering's internal research/ideation needs, AgentGate is published for others to install, and BreakWater contributed fixes to an existing open-source benchmark project.

**Is he product-minded or research-minded?** Both. He starts with the actual problem and values usable outcomes, while applying unusually rigorous attention to measurement and failure cases.

**What should interviewers ask him about?** A database invariant he used to prove crash recovery; why an agent policy should intercept a tool call before execution; how a published training schedule affected reproduction; the tradeoffs of leading a team building a client RAG product; or what he learned from verifier issues in an upstream benchmark.
