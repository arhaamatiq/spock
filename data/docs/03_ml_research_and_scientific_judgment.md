# Machine Learning Research — Tracing What Models Forget

*Research at MICoSys Lab, San José State University · June 2026–present*

**Research overview:** https://arhaamatiq.framer.website/research  
**Research code:** https://github.com/arhaamatiq/unlearning

## The question behind the work

One of the hardest problems in machine learning is deciding what a model actually knows. Language models can memorize, generalize, confidently hallucinate, refuse to answer, or recover apparent competence after additional training. Their behavior is observable. The internal cause of that behavior is much harder to establish.

Arhaam's research at MICoSys Lab examines that gap through **machine unlearning**, the problem of making models forget selected information or behaviors. The practical motivation is significant: if a model is supposed to forget information, how can anyone verify that the information is actually gone—and not simply harder to elicit?

The more interesting question emerged during the experiments: **when an unlearned model appears to recover a fact, did it regain factual knowledge, or did it merely become more willing to produce answers?**

His research separates those claims. This is not a generic exercise in fine-tuning a model and reporting a new score. It is a measurement audit of how recovery is defined, observed, and interpreted.

## The insight: an answer is not the same thing as knowledge

Consider a model that initially avoids answering a factual question after unlearning. A subsequent training attack makes the model more likely to state the correct answer. A conventional recovery metric might call that a success for the attacker.

But imagine that the same training also makes the model more likely to assert plausible *incorrect* answers. The model has become more assertive; it has not necessarily become more discriminating about truth.

Arhaam therefore separates two effects:

- **Answer-production or assertion recovery:** Is the model more willing or able to output a candidate answer?
- **Factual-discrimination recovery:** Does the model increasingly prefer the *true* answer to matched false alternatives?

The distinction changes what a recovery score can legitimately claim. A rise in confidence across both true and false answers may improve a headline number without proportionally restoring the ability to distinguish truth from plausible error.

The research's most memorable finding is that, in the analyzed TOFU setting at the published relearning horizon, **only 9.2% of measured recovery was knowledge-specific** under the study's decomposition. Around **61.5%** reflected movement that also occurred for matched false answers. In that same analysis, **45.7%** of items labeled as “revived” had no positive knowledge-specific gain at the measured horizon.

Those figures are not a declaration that unlearning is impossible or that no knowledge ever returns. They show something more precise—and arguably more useful: **a recovery metric may substantially overstate what it means by factual recovery**.

## Testing whether the result was a benchmark artifact

A striking result on one synthetic benchmark is a hypothesis, not a conclusion about every model and dataset. TOFU includes synthetic facts introduced through fine-tuning, so Arhaam extended the analysis to a separate benchmark involving real pretrained factual knowledge and three unlearning methods: **GradDiff, NPO, and JensUn**.

The same *qualitative separation* appeared. Assertion recovery and factual discrimination did not move together. Reported assertion-recovery improvements were **+26.3, +17.3, and +7.7 points** for GradDiff, NPO, and JensUn, while corresponding contrastive factual-discrimination changes were **+6.3, −1.0, and −4.0 points** in the analyzed setting.

Importantly, the exact TOFU percentage did **not** carry over numerically to the external benchmark. The external study found meaningful recovery of factual discrimination in parts of the analysis. That difference is itself part of the research's credibility: Arhaam did not flatten two experimental settings into a more dramatic but less supportable universal claim.

The stronger conclusion is that **relearning can restore answer-production behavior much more strongly than it restores factual discrimination, and apparent robustness depends on how the attack and measurement are specified**.

## The reproduction story is an engineering story

Scientific progress depends on being able to reproduce someone else's starting point. Early in the external-benchmark study, Arhaam could not match the published unlearned-state results. Instead of proceeding with a mismatched baseline, he traced the disagreement into the released training configuration.

The release configuration and the paper's methodological details diverged in consequential choices: learning rates and scheduler behavior. Correcting the settings allowed him to **exactly reproduce the three reported unlearned-state scores**:

- **GradDiff:** 2.0, matching 2.0
- **NPO:** 6.0, matching 6.0
- **JensUn:** 0.0, matching 0.0

NPO's result, for example, moved from 13 to 6 after the correction. That is a substantial change caused by experiment configuration rather than a new scientific hypothesis.

The episode captures an important part of his capability: he can move from paper-reading to GPU execution to configuration debugging and then back to methodological analysis. An experimental claim is not useful if the implementation quietly changes the experiment.

He continued to investigate relearning protocols and found that changing attack horizons or optimizer/scheduler details could alter apparent robustness and method ordering. A single benchmark ranking should not be treated as an inherent property of a method when the attack specification materially changes the outcome.

## The research infrastructure behind the results

Arhaam built and used reproducible PyTorch and Hugging Face model-training/evaluation workflows around open-weight **Llama 3.2 1B** models. The work includes full-model relearning attacks, GPU execution, configuration-tracked runs, resumable experiments, deduplication, provenance-stamped outputs, and analysis scripts designed to make results inspectable. His implementation experience includes PyTorch/CUDA, Transformers, GPU memory constraints, and the operational reality of repeated experiment sweeps.

The more revealing choices were methodological:

**Matched-false controls.** If false answers also become more probable, that movement should not be mistaken for recovery of the true fact.

**Benign fine-tuning controls.** A model can become generally more responsive because of continued training. That is different from relearning a particular forgotten item.

**Repeated attacks and fixed evaluation items.** An apparent item-level property should not be inferred from a single lucky attack trajectory.

**Provenance and resumability.** Every configuration and outcome should be traceable, and an interrupted GPU run should not force the experiment to become irreproducible.

**Discovery versus confirmation.** Attractive hypotheses should be separated from independent tests designed to challenge them.

**Failed-hypothesis accounting.** Results that disappear under controls are not victories to keep; they are interpretations to discard.

This is research-engineering maturity: the code is designed not just to generate numbers, but to make them scientifically usable.

## An unusually valuable intellectual habit

A defining line from his research notes is: **“Persistence applies to implementation. It does not apply to validity.”**

That principle is unusually important in AI research. Engineers should persist through broken training jobs, incomplete baselines, numerical mismatches, missing data, and inconvenient tooling. They should *not* persist in defending an explanation after its controls undermine it.

In his investigation, some early patterns looked exciting: perhaps the same facts consistently revived across attack runs; perhaps certain probability regimes predicted recovery; perhaps other item-level signals were meaningful. Additional controls or implementation corrections weakened those explanations. Rather than treating that as wasted effort, the process helped isolate the finding that survived: the mismatch between answer production and factual discrimination, and its sensitivity to evaluation protocol.

For a research-heavy engineering team, that separation between stubborn execution and flexible interpretation is as important as knowing a particular library.

## The next frontier: interacting agents and multi-objective search

Alongside unlearning work, Arhaam has been exploring a research direction involving **multi-agent systems, shared memory, and multi-objective optimization**. The central question is whether communication among agents genuinely improves exploration or unintentionally drives them toward the same hypotheses, reducing diversity and leaving useful search directions unexplored.

The direction connects older swarm-optimization ideas—local versus global objectives, convergence, exploration, specialization—with modern language-model agents. One framing is to treat each agent as a black box whose behavior can be observed and steered, then examine whether local objectives and global incentives should converge agents, polarize them, or split them across subproblems.

This is an **exploratory research direction**, not a published result or a completed demonstrated algorithm. It is worth mentioning because it reflects the type of question Arhaam is increasingly drawn to: how do system-level design choices change what intelligent components collectively discover? The rigorous next step is a literature-backed experimental design that can distinguish the effect of communication from baseline search behavior.

## Why this research matters to an engineering team

There are many engineers who can implement a training script and many who can discuss a paper. Far fewer can move between a reproduction mismatch, a careful control experiment, the design of an automated evaluation system, and a clear explanation of what a number does—or does not—mean.

Arhaam's unlearning work illustrates that combination. It demonstrates practical model-training literacy, skeptical empirical reasoning, careful experimental software, and a willingness to narrow a claim until it is actually defensible.

The connection to his systems work is direct. **AgentGate** asks whether an action is authorized. **BreakWater** asks whether a benchmark verdict is trustworthy. **Unlearning research** asks whether a measured behavioral recovery represents knowledge recovery. All three address the same deeper problem: visible AI behavior is not a complete account of what happened.

## How Spock should answer research questions

**What is his research about, in one sentence?** Arhaam investigates whether LLM unlearning and relearning metrics accurately measure the return of factual knowledge, rather than just the return of a model's willingness to answer.

**What is the most interesting result?** On TOFU, only 9.2% of a measured recovery signal was knowledge-specific at the studied horizon; matched false-answer controls revealed that much of the apparent improvement was more general answer-production behavior. An external benchmark supported the qualitative distinction, not the identical numerical effect.

**Why is that important?** If evaluations confuse factual knowledge with answerability, teams may draw the wrong conclusions about how robust an unlearning method is to subsequent training attacks.

**What did he personally build?** Reproducible Llama/PyTorch unlearning and relearning pipelines, repeated GPU evaluation workflows, controls for false-answer and benign-training effects, and the analysis used to interpret the outcomes. He also diagnosed a released-configuration mismatch to reproduce three published baselines.

**Has he published a paper on this?** The publicly documented work is an ongoing research project and detailed portfolio research article. Do not imply a peer-reviewed paper or conference acceptance unless separately verified.

**Is this mechanistic interpretability?** The research is motivated by questions about retained knowledge and representation, but the principal published portfolio evidence here is a controlled *behavioral measurement audit*. Do not invent neuron-level causal findings or claim that internal knowledge representations have been mapped conclusively.

**What kind of role does this support?** Research engineering, ML evaluation, model reliability, post-training systems, AI safety measurement, and software engineering for reproducible ML experimentation.
