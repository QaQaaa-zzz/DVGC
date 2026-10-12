# Reference and structural audit for the English JIT manuscript

Verified on 2026-10-12. This is an editorial reference note, not a research-state ledger. It uses the supplied RAS PDF and primary publication records. No manuscript, experiment, or model was changed. Bibliographic records are in [references_verified.bib](references_verified.bib).

## 1. Resolved reference papers

**Supplied RAS paper.** Benyan Huo, Long Yu, Yanhong Liu, and Zhang Chen, “Hierarchical residual reinforcement learning based path tracking control method for unmanned bicycle,” *Robotics and Autonomous Systems*, **190**, 104996, 2025. DOI: [10.1016/j.robot.2025.104996](https://doi.org/10.1016/j.robot.2025.104996). The [publisher record](https://www.sciencedirect.com/science/article/abs/pii/S092188902500082X) agrees with the supplied eight-page PDF.

Local source: `/home/qy/下载/RAS车.pdf`; SHA-256 `b7e1b45f89fcfcb7f9ccc8d5d45892124cbc75b1f69fecf1a651ab7c7ac3db14`. Text was extracted with `pdftotext -layout`; title, authors, DOI, sections, equations, captions, tables, and conclusion were inspected. The paper reports simulation, with real-world transfer left for future work.

**“ridelines” resolves to LineRides.** Seungeun Rho, Shamel Fahmi, Jeonghwan Kim, Arianna Ilvonen, Sehoon Ha, and Gabriel Nelson, “LineRides: Line-Guided Reinforcement Learning for Bicycle Robot Stunts,” 2026. The [author-submitted record](https://arxiv.org/abs/2605.05110), [v2 text](https://arxiv.org/html/2605.05110v2), and [PDF](https://arxiv.org/pdf/2605.05110v2) were inspected. Version 2 is dated 8 May 2026; the record identifies RA-L publication and the PDF is an accepted preprint. IEEE volume, pages, and journal DOI remain unverified. Use the verified arXiv identity, not guessed journal fields.

## 2. What to learn from the supplied RAS paper

| Part and location | Observed writing function | Application to JIT |
|---|---|---|
| Introduction, §§1.1–1.3, pp. 1–2 | Motivation, focused prior work, then numbered contributions. The technical gap leads directly to the proposed architecture. | Open with recovery after disturbed jumping and the difficulty of converting a local recovery trajectory into an independently executing policy. Introduce the E–G–π design before experimental history. |
| Bicycle model, §2, p. 2 | Geometry and physical variables precede the simplified dynamics, state definition, virtual input, and linear model, Eqs. (1)–(6). | Give the actual JIT robot, observation/action interface, task phases, and success definition before introducing learning losses. Use the source task's physics; do not import an LQR bicycle approximation. |
| Method overview, §3 and Fig. 2, pp. 2–3 | A system diagram explains the order of signals before controller derivations. | Walk through the user's E–G–π diagram once: exploration state, recovery proposal, verified demonstration, policy update, and explicit feedback. Distinguish a training-time teacher from the deployed policy. |
| Attitude and position control, §§3.1–3.2, pp. 3–4 | Each module states its purpose, variables, controller, learned correction, and reward. Equations are followed immediately by parameter meanings. | Organize subsections by exploration, recovery generation, student learning, and feedback. Each needs input, output, training data, objective, and one reason for its role. |
| Training, §3.3, p. 4 | Training order and the frozen lower controller are stated separately from the controller equations. | Separate the generic outer loop from the exact frozen protocol used by each reported experiment. A small protocol table can distinguish continuous PPO from supervised mechanism studies. |
| Experiments, §§4.1–4.3, pp. 4–7 | The attitude, position, and terrain studies correspond to method components. Baseline choices are explained before interpreting plots. | Name the question each comparison answers: teacher recovery, student transfer, skill retention, or a mechanism. Pair numerical tables with actual trajectories. |
| Tables 1–7; Figs. 4–11 | Configuration tables support reproduction; performance tables use physical metrics; learning curves and trajectory plots answer different questions. | Keep task success, action error, reward, and cost in distinct columns or panels. State the denominator and evaluation role in the caption. |
| Conclusion, §5, p. 7 | Returns to the architecture and observed outcomes, then gives concrete remaining work. | End with the strongest result established by the JIT evidence, followed by the specific unresolved step. Avoid a chronological recap of every run. |

The useful model is the paper's argument structure, not its wording or numerical settings. Its residual architecture, steering-angle output, TD3 training, PyBullet setup, and path-tracking metrics do not describe JIT. Its nonlinear-to-linear derivation is useful because that derivation supports its controllers; adding a similar derivation to JIT without using it would add length without explanation. Its component comparisons also contain different training and observation conditions, so they should not serve as a template for asserting equal-cost comparisons.

## 3. What to learn from LineRides

The sequence is introduction, related work, robot, problem formulation, method, experiments, and conclusion. The introduction pairs concrete difficulties with design choices; equations define the representation and control interface; experiments follow the claimed functions. Each stunt has its own policy combining driving and stunt modes. Guidelines shape training rewards rather than policy inputs. Five behaviors are shown in simulation and three on hardware. These distinctions matter when citing it as task context. [Original article](https://arxiv.org/html/2605.05110v2).

For JIT, adopt the question-to-design-to-experiment organization. Do not describe a local verified recovery sequence as an independently learned skill, or use another platform's hardware demonstration as evidence for JIT deployment.

## 4. Recommended English journal structure

This is a writing recommendation, not an instruction to execute missing experiments.

1. **Introduction.** Establish the physical task, identify the recovery-learning problem, explain the E–G–π mechanism, and state contributions supported by the current evidence. Keep run names, software audits, and authorization details out of the opening.
2. **Related work.** Group sources by their relationship to the problem: dynamic bicycle skills; exploration and restart-based learning; action-sequence generation; learning from teacher and student state distributions. Explain the relationship rather than listing algorithms in publication order.
3. **Problem formulation.** Specify the robot interface, complete closed-loop state, disturbances, recovery event, frozen source policy, and evaluation distributions. Define success before recovery expansion. Distinguish physical state, policy observation, and saved simulator context.
4. **JIT framework.** Present the complete pipeline figure, then exploration, conditional recovery generation and verification, student learning with retention, and learning-progress feedback. Finish with one outer-loop algorithm and a protocol table.
5. **Experimental protocol.** Identify the evaluated models and controls, data roles, selection rule, repeated evaluation, cost accounting, and statistical unit. Use one main comparison table and a compact configuration table.
6. **Results and analysis.** Organize by recovery generation, independent student transfer and retention, and mechanisms of transfer failure. Include the formal M0–M4 comparisons only as results become available. An unfinished comparison remains a clearly marked draft placeholder.
7. **Discussion and conclusion.** Explain what the evidence establishes about recovery learning and what it leaves unresolved. Put implementation failures and exact artifact identities in the reproducibility appendix while retaining their charged cost in the experiment accounting.

### Introduction paragraph roles

| Paragraph | Single message | Evidence or citation needed |
|---|---|---|
| 1 | A nominal jumping policy must also handle states displaced from its learned motion. | Define the task and cite directly relevant bicycle-skill work. |
| 2 | Obtaining a feasible recovery trajectory and learning reliable independent recovery are distinct problems. | Link exploration and imitation-learning literature to the current problem; avoid asserting that all prior methods fail. |
| 3 | JIT connects disturbance exploration, short recovery generation, and policy learning through verified demonstrations. | Explain the concrete modules and diagram, with no effectiveness claim inferred from architecture alone. |
| 4 | The evaluation separates teacher recovery, student transfer, and retention. | State the principal observed result and its actual experimental scope. |
| 5, if useful | List the contributions concisely. | Each contribution must map to a method definition, an evaluated comparison, or a documented empirical finding. |

Use direct topic sentences and stable terminology. A paragraph should develop one technical point; its next sentence should explain a cause, mechanism, consequence, or piece of evidence. Avoid generic openings about rapid AI progress. Also avoid repeating an audit disclaimer after every observation: define the scope once, then report the result precisely.

### Mathematical exposition

The main text needs equations that establish the scientific method, rather than every implementation detail:

- Define the augmented task state and observation map, including the history needed by the policy. Give the normalized action vector and physical command mapping once.
- Define the complete-task recovery event and the paired gained/lost counts. Make clear which policy and evaluation distribution appear in each quantity.
- Define the teacher's finite action sequence and its handoff to the frozen source policy. The handoff changes the controller, not the physical state or task clock.
- Give the conditional denoising objective and the executed sampling procedure. Keep G parameters and student parameters distinct. Cite DDPM, Diffusion Policy, and DDIM at their corresponding roles.
- Specify the fixed candidate selection and verification rule, including the source branch. An optimization over the finite candidate set is not a global feasibility certificate.
- Present the student PPO, demonstration, and retention objective. Define its sampling distributions, stopped reference policy, coefficients, and schedule clock. Use a separate equation or protocol row for the supervised diagnostic, rather than silently treating it as a PPO result.
- Define the two explicit feedback paths and data admission rule used by the stated outer-loop version. The combined M4 ablation tests their joint effect.

Define every symbol at first use and immediately explain what each equation does operationally. State all units and time indices. Keep gradient-cosine diagnostics, precise numerical tolerances, hashes, and exhaustive event flags in the appendix. Do not add a Lyapunov or reachability argument unsupported by the actual method. Recheck signs and normalization against source code; neither reference paper's equations should be transplanted into JIT.

### Experimental story and visual hierarchy

| Scientific question | Main evidence | Best visual |
|---|---|---|
| Can a teacher provide recovery unavailable to the source at the same state? | Fixed-source, same-state G/Noise/source comparison with all roots and fixed repetitions. | Teacher qualification matrix plus representative actual XY trajectories, with the complete matrix available. |
| Does that guidance improve an independent student? | Matched training budget, full-task student evaluation, gained/lost outcomes, and retention. | One principal student comparison table and learning curves versus actual cost. |
| Why can fitting improve while closed-loop transfer fails? | Same-observation action errors, paired trajectories, and retained counterexamples. | One combined mechanism figure; distinguish same-state error from same-tick differences along different trajectories. |
| What does feedback contribute? | Frozen M4 comparison against the complete framework. | Paired per-seed outcomes; do not split causal credit between its two disabled paths. |
| Does the frozen policy transfer? | Matching Actor, normalizer, task criterion, and target evaluation. | Same-policy trajectory and success comparison; keep unrelated historical demonstrations separate. |

Lead with a system/task figure and the method diagram. Make trajectory axes physical and comparable. Use common scales, readable legends, consistent model names, and captions that identify the model, denominator, data role, and repeated-evaluation rule. Reward curves supplement task outcomes; they do not replace them. A local root label is an index, not a training seed.

Move date-based stage names, budget reservations, process failures, desktop notifications, hashes, and machine paths to a linked appendix or supplement. Preserve the underlying evidence and failed attempts. Reducing their prominence in the scientific narrative does not remove their cost or change their interpretation.

## 5. Verified central bibliography and intended citation use

The seven works below complement the two supplied/style references. Metadata comes from author-submitted or official proceedings/publisher records, not search aggregators. The absence of a DOI field in the BibTeX means no publisher DOI was verified here, not that none can exist.

| Key | Verified reference and primary record | Appropriate use in JIT |
|---|---|---|
| `schulman2017ppo` | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017; v2, 28 August 2017. [arXiv](https://arxiv.org/abs/1707.06347). | Attribute the PPO surrogate and on-policy optimizer. PPO itself is not a JIT contribution. |
| `ho2020ddpm` | Jonathan Ho, Ajay N. Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. *Advances in Neural Information Processing Systems* 33:6840–6851, 2020. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), [publisher BibTeX](https://proceedings.neurips.cc/paper_files/paper/10298-/bibtex). | Attribute the denoising diffusion training formulation, not recovery feasibility or policy-retention properties. |
| `song2021ddim` | Jiaming Song, Chenlin Meng, Stefano Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021; arXiv:2010.02502. [Author record](https://arxiv.org/abs/2010.02502), [conference entry](https://openreview.net/forum?id=St1giarCHLP). | Attribute the DDIM sampling family. The OpenReview page was browser-challenged; author metadata and ICLR venue were confirmed on arXiv. |
| `chi2023diffusionpolicy` | Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin CM Burchfiel, Shuran Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS XIX, 2023. DOI [10.15607/RSS.2023.XIX.026](https://doi.org/10.15607/RSS.2023.XIX.026); [official proceedings](https://www.roboticsproceedings.org/rss19/p026.html). | Situate conditional action-sequence generation. JIT uses the generator as a verified teacher, so avoid implying identical deployment architecture. The author spelling follows the proceedings BibTeX. |
| `ross2011dagger` | Stephane Ross, Geoffrey Gordon, Drew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, PMLR 15:627–635, 2011. [PMLR](https://proceedings.mlr.press/v15/ross11a.html). | Explain why learning on teacher-state data can differ from performance on learner-induced states. A bounded self-visit repair is not automatically a full DAgger implementation or its guarantee. |
| `ecoffet2021goexplore` | Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. *First return, then explore*. *Nature* 590:580–586, 2021. DOI [10.1038/s41586-020-03157-9](https://doi.org/10.1038/s41586-020-03157-9); [publisher](https://www.nature.com/articles/s41586-020-03157-9). | Position remembering useful states and returning before further exploration. Do not conflate JIT snapshot reconstruction with exact simulator-state restoration or inherit Go-Explore's empirical results. |
| `florensa2017reverse` | Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel. *Reverse Curriculum Generation for Reinforcement Learning*. CoRL, PMLR 78:482–495, 2017. [PMLR](https://proceedings.mlr.press/v78/florensa17a.html). | Relate restart-state support to curriculum learning. JIT's fixed reset mixture must not be called this method's performance-adaptive reverse curriculum. |

### Four additional references used by the English revision

| Key | Verified reference and primary record | Appropriate use in JIT |
|---|---|---|
| `fahmi2026bicycle` | Shamel Fahmi, Arianna Ilvonen, Samuel Zapolsky, Yu-Ming Chen, Ravi Boggavarapu, Paul Drews, Ashwin Khadke, Dean Molinaro, Kaiyu Zheng, Alfred Rizzi, Gabriel Nelson. *Bicycle Acrobatics with Reinforcement Learning*. arXiv:2608.00880, 2026; submitted 1 August 2026. [Author-submitted record](https://arxiv.org/abs/2608.00880). No journal venue was verified. | Establish bicycle acrobatics as a related task. Its skill policies and state-dependent orchestrator do not establish a single universal JIT policy or JIT's hardware performance. |
| `johannink2019residual` | Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, Sergey Levine. *Residual Reinforcement Learning for Robot Control*. ICRA, 6023–6029, 2019. DOI [10.1109/ICRA.2019.8794127](https://doi.org/10.1109/ICRA.2019.8794127); [author preprint](https://arxiv.org/abs/1812.03201), [author-institution record](https://tore.tuhh.de/entities/publication/0f20de6e-0305-4e82-9cde-345323da5fd0). | Cite additive learned corrections to an existing controller. JIT's teacher–student learning structure should not be described as an additive residual unless its actual action interface is additive. |
| `schulman2016gae` | John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR, 2016. [Original paper](https://arxiv.org/abs/1506.02438), [official ICLR 2016 archive](https://iclr.cc/archive/www/2016.html). | Attribute generalized advantage estimation. Use conference year 2016; 2015 is the first arXiv year. The original PDF explicitly identifies ICLR 2016. |
| `agarwal2021rliable` | Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. *Advances in Neural Information Processing Systems* 34:29304–29320, 2021. [Official proceedings](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html), [publisher BibTeX](https://proceedings.neurips.cc/paper_files/paper/13867-/bibtex). | Motivate explicit uncertainty and appropriate statistical units. Three repeated evaluations of a fixed policy do not become independent training runs by citing this work. Its multi-task aggregate metrics should not be imported mechanically into one binary recovery task. |

For Residual RL, the IEEE page required a browser challenge. DOI, conference, page range, year, and author order were independently checked against the [publisher-deposited Crossref record](https://api.crossref.org/works/10.1109/ICRA.2019.8794127) and TUHH's institutional record. TUHH's rendered citation contains `60023`, but its structured start-page field and IEEE's deposited page range both give **6023**. The BibTeX follows the deposited family-name split `Ojea, Juan Aparicio`. Rliable's author names follow the original paper; its publisher BibTeX omits some middle initials. DDPM's author name follows the proceedings display, while the BibTeX export abbreviates it to Ajay Jain. These are metadata variants, not different works.

### Numeric mapping for the English rewrite

The author requested the following order. This is the target reference map for the English rewrite, not a renumbering of the preserved Chinese draft. Citation numbering must be checked again against the completed English manuscript before the Word build.

| Number | BibTeX key | Short identity | Publication year |
|---|---|---|---|
| [1] | `huo2025hrrl` | Huo et al., hierarchical residual bicycle path tracking | 2025 |
| [2] | `fahmi2026bicycle` | Bicycle Acrobatics | 2026 |
| [3] | `rho2026linerides` | LineRides | 2026 |
| [4] | `ecoffet2021goexplore` | First return, then explore | 2021 |
| [5] | `florensa2017reverse` | Reverse Curriculum Generation | 2017 |
| [6] | `johannink2019residual` | Residual Reinforcement Learning for Robot Control | 2019 |
| [7] | `ho2020ddpm` | Denoising Diffusion Probabilistic Models | 2020 |
| [8] | `song2021ddim` | Denoising Diffusion Implicit Models | 2021 |
| [9] | `chi2023diffusionpolicy` | Diffusion Policy | 2023 |
| [10] | `ross2011dagger` | DAgger / no-regret imitation learning | 2011 |
| [11] | `schulman2017ppo` | Proximal Policy Optimization Algorithms | 2017 |
| [12] | `schulman2016gae` | Generalized Advantage Estimation | 2016 |
| [13] | `agarwal2021rliable` | Statistical Precipice / rliable | 2021 |

This file covers all thirteen requested references. It does not claim an exhaustive novelty search or validate additional references not listed here. The metadata audit does not establish that every possible statement citing a listed paper is supported; sentence-level support is a separate manuscript check.

## 6. Author-facing final checks

- **Contribution:** the opening states the recovery-learning problem and concrete method, with each effectiveness statement mapped to an executed comparison.
- **Clarity:** one paragraph develops one message; the figure, equations, and algorithm use the same symbols and ordering.
- **Evidence:** teacher success, student success, newly acquired recovery, and lost prior skill have different labels and denominators.
- **Evaluation:** independent training seeds, repeated fixed rollouts, and multiple states from one ancestor are distinguished; selected and final checkpoints are identified.
- **Method:** generative recovery, source-policy continuation, learning, and feedback are explicit. Historical protocol differences are stated once in a table or appendix.
- **Reference fidelity:** the RAS study is simulation path tracking; task/interface details are not imported into JIT. Unverified publication fields remain blank. Source sentences and figures are not copied.

Writing guidance applied: `research-paper-writing` (introduction, method, experiments) and `academic-paper/references/writing_quality_check.md`. The complete-paper pipeline was not invoked for this bounded reference-audit assignment.

## 7. Independent English-manuscript review, 12 October 2026

Reviewed [the English manuscript](../JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.md), Sections 1–8 and Appendices A–B, while the author was integrating the final equation checks. This is a focused editorial and evidence review, not a five-reviewer assessment. The findings below record the initial review requests; their presence here does not mean they remain unfixed in later manuscript revisions. No manuscript or execution source was edited by this reviewer.

### Overall assessment

The English version now reads primarily as a methods paper: the physical problem leads to the exploration, teacher, verification, student, and feedback definitions; the results ask whether recovery guidance transfers to an independent student. The contribution can be stated without treating implementation records as performance evidence. The current negative student results are visible in the abstract, principal results table, discussion, and conclusion. The five-method experiment and same-policy transfer remain explicit unfilled templates. Citation uses [1]–[13] match the verified map above; no unsupported first-in-field claim was found.

The primary unresolved scientific claim remains complete-framework performance, not bibliographic completeness. The present evidence supports repeatable local recovery witnesses and a measurable teacher-to-student transfer gap. It does not yet establish aggregate diffusion superiority, reliable skill retention, a feedback benefit, or final-policy transfer. The manuscript generally respects these boundaries.

### Required accuracy and protocol clarifications sent to the author

1. **Table 3: source TRAIN and SOLVER_DEV are historical screening labels.** The T1 result archive says those original-policy root evaluations were not repeated in the paired PPO endpoint experiment. The zeros are consistent with D1's failure-selected root set, but the table caption must not imply a new three-policy paired evaluation for those two rows. Mark them `0 (historical screening)` or with a footnote: “Original-policy TRAIN and SOLVER_DEV labels are inherited from the earlier failure-screening study; they were not re-evaluated in this pilot. Complete DEV was evaluated for all three policies.” Source: [T1 result index](../../evidence/retention_next_20261010/T1_paired_ppo_002/INDEX.md).

2. **Section 6.2: distinguish the comparison screen from shared evaluation.** The initial sentence “The same mechanism-screening rule was applied to both arms” overstates symmetry: the teaching arm was required to exceed the control. Replace it with “A predeclared continuation screen compared the teaching arm with the control.” Shared evaluation and retention checks can be stated separately. This keeps the numerical negative result intact without making the paper sound like an authorization report.

3. **Equation (21): limit its separate-source requirement to the original-root D1 qualification.** The student-visited queries evaluated the source-only lane and fixed teacher winner three times at each new context. They did not additionally perform an independent three-repeat source rollout at every new query. The new-demo admission then required a historically stable-failure student ancestor and a failed current full student rollout. A suitable clarification is: “Equation (21) describes the original-root qualification study. At student-visited query states, repeated source-only lanes provide the local source comparison; admission additionally requires a consistently failed ancestor under the earlier student and a failed current student rollout.” Evidence: `retention_self_visit.py:290–303`, `retention_self_visit_learning.py:98–105,248–257`, and [query records](../../evidence/retention_next_20261010/self_visit/query_matrix.json). Ancestor `pi0_labels` in inherited capture metadata are not independent evaluations at the new query state.

4. **Report the measured pilot cost once.** The original draft gives the two 32,000-transition learning budgets and a general cost formula but omits the completed stage's total cost. Suggested appendix sentence: “The paired pilot and its fixed evaluations and engineering checks charged 328,308 environment control steps, excluding earlier development stages and generator pretraining.” This is T1_002's total; the previous T1_001 engineering failure charged a separate 200 steps and must not be silently included in that number. Source: [T1 cost accounting](../../evidence/retention_next_20261010/T1_paired_ppo_002/INDEX.md).

5. **Table 1: explain feedback under a fixed random explorer.** In M2, a fixed random exploration policy cannot change behavior in response to the student-quality reward. If “Both paths” denotes available measurements rather than effective learning updates, say so. The retained student-to-generator data path is distinct from an inactive student-to-explorer update. This clarification was also sent to the method reviewer; it should follow the frozen experiment definition rather than silently changing it.

### Concrete journal-style revisions

- **Abstract ending.** Avoid ending with a statement about empty panels. Suggested replacement: “The studies reveal a transfer gap between repeatable teacher recoveries and independent student performance, motivating matched-budget evaluation of the complete feedback loop.” Keep the actual 12/24, 4/8, and 1/8 results. The placeholder status is already explicit in the experimental section.
- **Introduction ending.** The sentence about reporting available evidence “without substituting” pilots for a planned comparison is editorial process language. A tighter scientific paragraph is: “The development studies establish repeatable local teacher recoveries and identify student absorption as the current limitation. The experimental design separates the effects of exploration, generative teaching, and feedback under matched evaluation conditions.” Follow this with the existing section roadmap if needed.
- **Contribution 3.** Consider emphasizing the evaluated empirical analysis instead of presenting a standard auxiliary-loss combination as novelty: “We provide a paired analysis of recovery generation, independent student absorption, and prior-skill retention, exposing cases where lower action error does not yield stronger closed-loop recovery.” This is supported by the current studies and makes the paper's present contribution more specific.
- **Section 6.2 screening paragraph.** Suggested compact replacement after giving the thresholds once: “The teaching arm fails the predeclared screen: it neither reaches the required TRAIN and solver counts nor exceeds the control in paired DEV net gain. The pilot therefore provides no absorption advantage for the additional demonstration term.” The final sentence need not discuss using a larger campaign to amplify the result.
- **Section 6.3 wording.** Prefer “executed simulated action windows” to “real action windows” in Section 4.3. This avoids an unintended suggestion of hardware data. Describe action-chain agreement as an engineering consistency result, without claiming contact-trajectory equivalence.

These are targeted edits. The method, equations, actual negative results, and unfilled scientific comparisons should remain. There is no need to add process logs, authorization history, or redundant cautionary sentences to the main text.

### Numerical checks and interpretation boundaries

- The T1 table agrees with the compact evidence: stable TRAIN 1/8 versus 1/8; SOLVER_DEV first pass 1/7 versus 1/7; stable DEV 56/64 versus 54/64; strict paired gain/loss 10/3 versus 6/0. The source has 35 stable successes and 14 ambiguous conditions. Therefore paired net gains +7 and +6 must not be replaced by total stable-count differences +21 and +19.
- The supervision study's first-pass 56/64 versus source 38/64, with 20 gains and two losses, is correctly separated from three-repeat stable PPO outcomes. The 4/8 stable absorption result and root exchange are retained, including the ambiguous `(1,0,0)` root.
- The common observation corpus contains 7,291 records: 525 old-demo, 228 added-demo, 6,289 retention, and 249 old-stable-student trajectory observations. Same-observation inference errors and same-time actions along different closed-loop trajectories remain distinct.
- T0's independent CPU-to-saved maximum discrepancy is 0.000741865486, below its frozen tolerance 0.00559464097. Independent CPU versus native CPU differs by at most 2.68220901e-7. The shared CPU-to-saved discrepancy must not be attributed to a particular GPU kernel without further evidence. [T0 source report](../../evidence/retention_next_20261010/T0_saved_diagnostics_002/INDEX.md).
- The T1 action-error values, median weighted demo/PPO gradient norm ratio 1.8815%, and 37/80 negative gradient cosines agree with [the saved T1 report](../../evidence/retention_next_20261010/T1_paired_ppo_002/INDEX.md). They do not isolate clipping, Adam, capacity, or contact sensitivity as a cause.
- The 525-observation export precheck and its maximum action discrepancy of 4.77e-7 establish tested inference agreement, not physical transfer. Earlier target-trained policies remain outside the proposed zero-target-training comparison.

### Reference delivery and checks

All thirteen unique BibTeX keys have balanced braces and are mapped above; UTF-8, final newlines, and trailing whitespace were checked. A numbered [1]–[13] insertion snippet was delivered to the author at `/tmp/jit_refs_en.md`. No BibTeX renderer was available on this reviewer's PATH, so these checks are not a claim of a renderer-tested bibliography. The author controls final manuscript insertion, rendering, and the subsequent citation-order check.

**Integration recheck.** A subsequent read of the author's manuscript confirms that all five accuracy requests above were incorporated: historical-source footnote, asymmetric continuation-screen wording, original-root versus student-visited verification, Appendix B.6 pilot cost, and M2's student-to-generator feedback label. The author also inserted the numbered bibliography; the complete set of body citation numbers is exactly 1–13, and the reference entries match the delivered mapping. All relative links in this audit resolve. The optional prose revisions remain editorial suggestions, not unresolved evidence blockers. The final document build and visual layout are outside this reviewer's file ownership.
