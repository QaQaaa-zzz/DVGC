# JIT method and equation audit

Audited 2026-10-12, repository HEAD `2bfaac6c88314e71865a6694b1fed3650953959d`. This is a source audit, not an additional experiment. No environment, GPU, or training was started. Repository references below are relative to `JIT/src/jit_dvgc/`; line numbers refer to this revision. **Brax** and **RSL** references refer to the actual runtime installation under `/home/qy/mujoco_playground/.venv/lib/python3.12/site-packages/`, not a different Conda environment.

## 1. Scope, notation, and evidence boundaries

Use distinct indices: physical control time \(t\); outer iteration \(k\); diffusion noise level \(n\); completed student transitions \(u\); optimizer update \(v\). The controller period is 0.02 s, so an H=16 prefix lasts 0.32 s. A root is a saved **closed-loop context** \(\chi\), not merely a generalized-position vector. It includes physical coordinates/velocities, observation FIFO and validity, previous action, phase/event/recovery counters, RNG, and task time. Context preservation is required at teacher handover and snapshot resets.

Let \(o\in\mathbb R^{76}\) denote the Actor observation, \(p\in\mathbb R^{106}\) the privileged Critic observation, and \(a\in[-1,1]^4\) the normalized absolute action in steer/rear-drive/hip/knee order. The Actor observation is three 25-dimensional history frames plus the jump signal. Each frame contains gravity(3), angular velocity(3), acceleration(3), steer/hip/knee positions(3) and velocities(3), wheel velocities(2), previous normalized action(4), forward velocity, obstacle-relative x, height, and validity. The Critic appends qpos(12), qvel(11), RPY(3), world root linear velocity(3), and illegal-contact(1). Some quantities are simulator-derived; this interface is not demonstrated to be sensor-complete on hardware. Sources: `observation.py:33–58,73–146`.

For source statistics \((\mu_0,s_0)\), write \(N_0(o)=(o-\mu_0)/s_0\). There is **no extra observation clip or epsilon added at inference** in this path. Stored standard deviations already belong to the frozen state. Generator conditioning uses its checkpoint's fixed original-source Actor statistics, including when the historical student changed. T1 freezes the entire original Actor **and Critic** running-statistics state, including count. Sources: `generative_bridge/worker.py:34–48`; `generative_bridge/diffusion.py:74–80,103`; `ppo.py:92–101`; Brax `training/acme/running_statistics.py:302–318`; `generative_bridge/student_clock.py:19–43`.

| Mechanism | Execution evidence / correct scope |
|---|---|
| E collection → frozen-root G search → student learning → delayed E/G updates | Implemented and executed in the historical continuous lineage, including R73. This establishes execution of the loop, not reliable net improvement. |
| Source-normalizer-frozen student with true four-pool terminal resets | Implemented and executed in T1_002; both arms completed 32,000 transitions and 80 optimizer updates. This was a one-seed negative mechanism screen, not a successful end-to-end JIT campaign. |
| Self-visited teacher queries and supervised data augmentation | Executed diagnostic and subsequent learning subexperiment. E and G were frozen; this is a test of the G→student transfer link. |
| Five methods × three independent training seeds / full external-disturbance branch | Planned formal evaluation; not completed evidence. Formal campaign is closed/paused, final TEST remains unopened. Do not describe these comparisons as results. |

Live authority: shared ledger `PROJECT_STATE.md`, revision 270 near lines 2257–2263. Runtime references: `generative_bridge/closed_loop.py:345–384`, `retention_t1.py:321–371`, `retention_t1_runtime.py:13–29,111–145`. Historical R73 used evolving student statistics and its then-current reset/continuation protocol; do not retrospectively assign T1's frozen-statistics/four-pool formulation to it.

## 2. E: observations, pulse distribution, reward, and complete loss

### Local evidence encoder

The raw E observation is 386-dimensional: privileged base 106, 16 neighbor rows of dimension 17, and eight statistics. A neighbor row is \([r_j,\ell_j,m_j]\), where \(r_j\in\mathbb R^{12}\) is a normalized relative physical coordinate, \(\ell_j\in\{0,1\}^4\), and \(m_j\) is validity. **The validity bit is not an MLP input**:

\[
h_N=\frac{\sum_{j=1}^{16}m_jg_\phi(\operatorname{where}(m_j,[r_j,\ell_j],0))}{\max(1,\sum_jm_j)},\qquad x=[N_E(p),h_N,s_N].
\]

The row MLP is 16→64→64 with ELU; Actor and Critic have separate encoders and separate 178→128→128→128 heads. E outputs four means and four state-dependent log standard deviations; the Critic outputs one scalar. Under `train_history_v1`, evidence flags are current-source success/failure, historical Actor success, verified-teacher success. Unknown or conflicting labels are not failure flags. Eight statistics are log(1+count), current-source success/failure fractions, and nearest distance, computed for near/far radius sets. Nearest empty-distance sentinel is 2. Neighbor selection uses same phase, Chebyshev distance ≤1 in fixed halfwidth units; statistics additionally use distance ≤2. Up to 16 near rows are selected. The evidence map is immutable during collection.

Coordinates are x/y/z, vx/vy/vz, roll/pitch/yaw, and angular rates; widths are `(0.2,0.05,0.1,0.4,0.2,0.6,2,5,5,10,20,20)` in their stored metre/second/degree units. Sources: `neighborhood.py:16–24,138–166,174–220`; `neighborhood_network.py:13–32`; `rsl_pulse.py:65–98,122–181`.

### Pulse and probability space

\[
z_t\sim\mathcal N(\mu_\phi(x_t),\operatorname{diag}(e^{2\eta_\phi(x_t)})),\quad
\delta_t=0.25\tanh z_t,\quad
a_t=\operatorname{clip}(\bar\pi_k(o_t)+m_t^{\rm pulse}\delta_t,-1,1).
\]

Here \(\bar\pi_k\) means deterministic `tanh(mean)` action. The scheduled pulse has three real control steps, onset in {0,5,10,15}. The request and effective change \(a_t-\bar\pi_k(o_t)\) differ under saturation. E optimizes **latent Gaussian** log probabilities and latent entropy. No tanh Jacobian or final action-clipping density is used for E. This is a valid declared latent-action objective, but it is not the entropy of bounded actuator actions. Only actual learned pulse samples with resolved eligible feedback enter PPO; padding and uniform-branch samples do not. Sources: `rsl_pulse.py:148–181,194–227`; `generative_bridge/explorer_admission.py:131–171,175–183`. R73 collection's frozen `hyperparameters.json` declares `uniform_episode_fraction=0.0`, not the older 80/20 mixture.

### Delayed outcome reward

Let \(C_i\) be the set of real pulse cells, \(S_k\) the frozen source's earlier ledger, and \(n(c)\) the number of current-batch episode sets containing new cell c. For an eligible nonempty set,

\[
N_i=\begin{cases}
\frac{0.02}{|C_i|}\sum_{c\in C_i\setminus S_k}\frac1{n(c)},&C_i\setminus S_k\ne\varnothing,\\
-0.02,&C_i\setminus S_k=\varnothing.
\end{cases}
\]

Let y_pre/y_post be valid independent student labels and a_acc the experimental continuation acceptance decision. With an attempted learning/evaluation outcome,

\[
Q_i=2\mathbf1\{y_{\rm pre}=0,y_{\rm post}=1,a_{\rm acc}=1,\text{source recheck}\ne1\}
-0.1\mathbf1\{y_{\rm pre}=0,y_{\rm post}=0\}.
\]

Then \(R_i^E=-2\) for direct physical failure after at least one applied pulse action with stage reached; otherwise \(R_i^E=N_i+Q_i\) for eligible rows. Direct failure overrides novelty, including a failure before any quantizable cell. Initial success/final success is eligible with zero quality term. Initial failure with missing adoption decision or unresolved/unattempted final outcome is excluded. **Teacher success contributes no quality bonus.** A historically failed root found source-recoverable on recheck cannot earn the +2 incremental-conversion credit. The novelty ledger update unions all stage-reached cell sets, not just positive or PPO-eligible outcomes. Sources: `generative_bridge/rewards.py:4–27`; `discovery_reward.py:13–46`; `generative_bridge/closed_loop.py:32–43,358–370`.

This reward is a delayed binary outcome credit; it is not a derivative of global success rate, theoretical learning progress, or a teacher-rescue reward. `successor_adopted` historically means experimental continuation, not formal deployment adoption.

### E PPO objective and coefficients

All executed eligible pulse actions in episode i receive return \(R_i^E\), equivalent to terminal attribution with \(\gamma_E=\lambda_E=1\). Set \(A_{it}=R_i^E-V_{\nu,old}(x_{it})\), standardized over all valid samples (Torch sample standard deviation; a singleton gets zero). With latent probability ratio \(\rho_{it}\),

\[
L_E=-\mathbb E\min(\rho A,\operatorname{clip}(\rho,0.8,1.2)A)
+\mathbb E\max\{(V_\nu-R)^2,(V_{old}+\operatorname{clip}(V_\nu-V_{old},-0.2,0.2)-R)^2\}
-0.01\,\mathbb E\sum_{d=1}^{4}\tfrac12\log(2\pi e\sigma_d^2).
\]

There is **no additional 1/2 in E's value term**; value coefficient is 1.0. R73 used up to four epochs, minibatch 256, initial/max LR 0.001, max gradient norm 1.0. RSL target KL=0.01 changes LR: divide by 1.5 above twice target, multiply by 1.5 below half target (positive KL), internal floor 1e-5, experiment ceiling 0.001. A separate post-epoch KL>0.03 stops further epochs. It is incorrect to state “hard KL stop at 0.01.” Sources: `rsl_pulse.py:257–295`; RSL `rsl_rl/algorithms/ppo.py:274–312`; frozen R73 `round_0073/collection/hyperparameters.json` and `explorer_update/hyperparameters.json`.

## 3. G: conditioning, complete diffusion equations, architecture, update

Use clean sequence \(U\in[-1,1]^{16\times4}\), and noisy sequence \(X_n\); this avoids calling both the clean action and the first noisy index “U_0”. Define

\[
f_j=\cos^2\left(\frac{j/100+0.008}{1.008}\frac\pi2\right),\quad j=0,\ldots,100;
\quad\beta_n=\min(1-f_{n+1}/f_n,0.999),\quad
\bar\alpha_n=\prod_{q=0}^{n}(1-\beta_q),\quad n=0,\ldots,99.
\]

For \(n\sim\operatorname{Uniform}\{0,\ldots,99\}\), \(\epsilon\sim\mathcal N(0,I_{16\times4})\),

\[
c=N_G(o),\quad X_n=\sqrt{\bar\alpha_n}U+\sqrt{1-\bar\alpha_n}\epsilon,\qquad
L_G=\mathbb E_{(o,U),n,\epsilon}\frac1{64}\|\epsilon_\psi(X_n,c,n)-\epsilon\|_F^2.
\]

The loss averages over all batch, time and action dimensions. There is no per-dimension sum replacing the mean, success classifier, classifier-free guidance, or reward-gradient term. Conditioning occurs once at the query root using its real 76-D pre-action observation. In self-visited queries this is the newly captured observation, not ancestor coordinates copied as metadata.

The U-Net has temporal lengths 16→8→4→8→16, widths 64/128/256/128/64, two residual blocks at each level, skip concatenation, nearest-repeat upsampling followed by convolution, GroupNorm(8 groups, epsilon 1e-5), and SiLU. Observation embedding is 76→128→256; time embedding uses 128 sin/cos features with \(\omega_j=\exp[-\log(10000)j/63]\), j=0…63, then 128→256→256. The concatenated 512-D condition generates FiLM scale/shift. A residual block's conditioned activation is \(\operatorname{SiLU}[(1+s(c,n))\odot\operatorname{GN}(\operatorname{Conv}x)+b(c,n)]\), followed by another conv/GN/SiLU and residual addition. Sources: `generative_bridge/network.py:6–50`; `generative_bridge/diffusion.py:8–17,97–108`.

### Exact deterministic DDIM20

Initialize \(X_{99}\sim\mathcal N(0,I)\). Visit **99,94,…,4**, not 99,…,0. At each n,

\[
\hat U=\operatorname{clip}\left(\frac{X_n-\sqrt{1-\bar\alpha_n}\epsilon_{\bar\psi}(X_n,c,n)}{\sqrt{\bar\alpha_n}},-1,1\right),\quad
\tilde\epsilon_n=\frac{X_n-\sqrt{\bar\alpha_n}\hat U}{\sqrt{1-\bar\alpha_n}},
\]
\[
a_{prev}=\begin{cases}\bar\alpha_{n-5},&n\ge5,\\1,&n=4,\end{cases}\qquad
X_{prev}=\sqrt{a_{prev}}\hat U+\sqrt{1-a_{prev}}\tilde\epsilon_n.
\]

The residual noise must be **recomputed after clipping**. There is no new stochastic noise in the reverse updates (eta=0). The final output is the last clipped clean estimate. Inference uses EMA \(\bar\psi\leftarrow0.999\bar\psi+0.001\psi_{new}\), updated after each accepted optimizer step. Sources: `generative_bridge/diffusion.py:5,20–30,108`.

G optimization is Adam with global gradient clip 1.0 and decoupled weight decay 1e-4. Pretraining supports up to 20k updates, first 1000 linear-warmup updates to 1e-4, then cosine to 1e-5. Incremental training uses LR=1e-5, batch 256, at most 2000 updates; params, EMA, optimizer, RNG, fixed normalizer, and counter continue together. With no new admitted data, training and RNG advancement are skipped. R73 uses `last_valid`: final durable full state supplies inference EMA; old fixed generator-dev MSE is monitoring only. Earlier `fixed_dev_best` behavior exists but must not be attributed to the frozen R73 teacher. Sources: `generative_bridge/diffusion.py:33–66,140–156,295–377`; R73 `round_0073/production.json`.

The corpus admits TRAIN/generator_train full-success trajectories only, from bootstrap Actors, replay-verified teachers, or independently successful **adopted** Actors with matching identities. It excludes nonfinite/incomplete/failed/timeout records, DEV/TEST, and unavailable pre-action observations. Sampling is source group (history .5 / new teacher .25 / new adopted Actor .25, renormalized over nonempty groups), then uniform ancestor, uniform trajectory, uniform valid H16 window. Windows may cross prefix/tail or physical phase boundaries but not resets/terminal padding. Loss itself is unweighted after this sampling. Sources: `generative_bridge/feedback_data.py:5–50,53–111`; `generative_bridge/data.py:7–59`.

## 4. Teacher execution, qualification, and demonstration mass

G supplies 16 independent H16 proposals. A batch has 17 worlds: closed-loop source lane 0 and 16 proposal lanes. Candidate execution is \(a_{t+\ell}=U_\ell\) for \(0\le\ell<h\), then deterministic source Actor thereafter, with no reset at handover. Standard h=16; self-visited j∈{4,8,12} uses h=16−j while still generating 16 actions. Context and episode clock continue; these queries do not acquire a fresh physical episode allowance merely by resetting a local evaluator index. Sources: `generative_bridge/rollout.py:14–26,45–62,102–114`; `retention_self_visit.py:228–307`.

Only complete physical successes enter ranking:

\[
c^*=\operatorname{lexargmin}_{c:S_c=1}\left(\sum_{\ell=0}^{T_c-1}\|a_\ell^c-a_{\ell-1}^c\|_2^2,\ -\sum_{\ell=0}^{T_c-1}r_\ell^c,\ c\right).
\]

The first difference uses the root's true previous action. Cost includes the complete successful source tail, not just generated prefix. Engineering/incomplete evaluations are errors, not unsuccessful searches. Winner identity and proposals are frozen before repeats. Repeats preserve the original whole batch, lane order, and logical RNG mapping; repeated deterministic checks are not independent draws establishing a success probability. Sources: `generative_bridge/teacher.py:5–33`; `generative_bridge/production.py:436–493,535–593`; `generative_bridge/teacher_runtime.py:36–49`.

**Different executed qualification contracts must remain distinct:**

* Historical R73: independent source recheck, source comparison in 17-world search, then **one** complete-layout replay of the fixed winner. Source success/conflict quarantines incremental credit; selected-candidate replay failure rejects it. No automatic re-selection.
* D1 original roots: independent source labels 000, same-layout source labels 000, fixed winner labels 111 over search+two repeats; all complete. Thus \(w_i^{D1}=\prod_{r=1}^{3}(1-y^0_{ir})(1-y^{src}_{ir})y^T_{ir}\) is valid **for this D1 screen**, with additional TRAIN/no-ambiguity admission. Sources: `retention_next_runtime.py:84–116,161–181`.
* Self-visited query: same-layout source 000 and fixed winner 111; **no additional independent-source triple at the new query state**. Thus \(w_{ij}^{SV}=\mathbf1\{\text{eligible stable-failed ancestor, current student failure}\}\prod_{r=1}^{3}(1-y^{src}_{ijr})y^T_{ijr}\). All 24 declared query slots must be resolved before data freezing; only the fixed second-confirmation winner trace is exported. Sources: `retention_self_visit.py:44–64,290–307`; `retention_self_visit_learning.py:198–232,245–269`.

D1 Noise comparison uses 16 candidates plus a source lane separately for each method:

\[
e_0=\xi_0,\quad e_\ell=0.8e_{\ell-1}+\sqrt{1-0.8^2}\xi_\ell,\quad
U_\ell^{Noise}=\operatorname{clip}(a_\ell^{ref}+0.2e_\ell,-1,1),\quad\xi_\ell\sim\mathcal N(0,I_4).
\]

Reference is the saved real source prefix; if shorter than 16, its final action is repeated only to define proposals. This padding is not a successful demonstration and source lane still acts closed-loop. `G_only` means only G obtained fixed-winner repeated qualification; it does not mean Noise never succeeded in any search/replay. Source: `retention_next_runtime.py:136–180`; D1_001 `plan.json` config (noise_rho=.8, noise_scale=.2). The older helper default of 15 colored candidates in a combined 32-world pool is a different protocol.

Demonstrations consist of actual pre-action observations paired with actual executed normalized actions, including successful tail. No learner's first j actions are prepended to a self-visited teacher trace. For the current augmented set, with A=8 ancestors, \(M_i\) trajectories at ancestor i, \(S_{iq}\) available real segments, and \(n_{iqs}\) real frames in segment s,

\[
P(i,q,s,t)=\frac{1}{8M_iS_{iq}n_{iqs}}.
\]

When both prefix/tail exist, each gets half its trajectory mass. Old observations/actions are preserved; trajectory balancing redistributes their sampling mass within ancestors, so frame counts are not sampling proportions. Current data has 525 old +228 new=753 real frames; keep data is the independently locked 6289-observation set. Sources: `retention_self_visit_learning.py:234–298`; T1_002 `training/PPO_demo_keep/training_contract.json` and referenced demo/keep manifests.

## 5. Student: exact current PPO + demo + keep formulation

### Policy and normalization

Actor: 76→256→256→256→8, Swish hidden activations; split eight outputs into mean and scale logits. Critic: 106→256→256→256→1. Despite the generic factory's `state_dependent_std=False` default, its **tanh_normal branch uses an ordinary 8-output MLP**, so both output sets depend on observation:

\[
z\sim\mathcal N(\mu_\theta(N_0(o)),\operatorname{diag}\sigma_\theta^2(N_0(o))),\quad
\sigma_\theta=\operatorname{softplus}(s_\theta)+0.001,\quad a=\tanh z,\quad\bar\pi_\theta(o)=\tanh\mu_\theta(N_0(o)).
\]

The last expression is Brax's deterministic `mode` convention, not a claim that tanh(mean) is mathematically the density mode after transformation. For PPO,

\[
\log\pi_\theta(a|o)=\sum_d\left[\log\mathcal N(z_d;\mu_d,\sigma_d^2)-\log(1-\tanh^2z_d)\right],
\quad\widehat H_\theta=\sum_d\left[\tfrac12\log(2\pi e\sigma_d^2)+\log(1-\tanh^2\tilde z_d)\right],\quad\tilde z\sim\mathcal N(\mu,\sigma^2).
\]

The entropy is a Monte Carlo Jacobian-corrected estimate; this differs from E's latent Gaussian entropy. Sources: `ppo.py:79–101`; Brax `training/networks.py:540–589`, `training/agents/ppo/networks.py:54–84,91–159`, `training/distribution.py:74–92,138–172`.

### Exact return and advantage convention

Use \(d_t^{env}=1-\texttt{data.discount}_t\), truncation \(\tau_t\), \(m_t=1-\tau_t\), and \(d_t=d_t^{env}(1-\tau_t)\). The trainer enables `bootstrap_on_timeout`: before loss calculation it sets

\[
\tilde r_t=r_t+\gamma\,\texttt{time\_out}_t\,V^{beh}(p_t),\qquad r'_t=0.1\tilde r_t.
\]

This uses the **stored pre-step rollout value**, not a fresh terminal-observation value. It does not clear truncation, and the reward scale is applied after this addition. Do not silently replace it with a textbook terminal-state bootstrap or claim ideal unbiased time-limit handling. References: `unified_formal.py:549–583`; Brax `training/agents/ppo/train.py:599–609`; `networks.py:73–83`; `losses.py:212–214`.

With value estimates recomputed for the current loss parameters, let \(V_T\) be the value at the unroll's last next-observation, set \(b_T=0\), \(v_T^\lambda=V_T\), and recurse:

\[
\delta_t=m_t[r'_t+\gamma(1-d_t)V_{t+1}-V_t],\quad
b_t=\delta_t+\gamma\lambda(1-d_t)m_tb_{t+1},\quad
v_t^\lambda=\operatorname{sg}(V_t+b_t),
\]
\[
A_t=\operatorname{sg}\{m_t[r'_t+\gamma(1-d_t)v_{t+1}^\lambda-V_t]\},\qquad
\hat A_t=\frac{A_t-\operatorname{mean}(A)}{\operatorname{std}(A)+10^{-8}}.
\]

The policy advantage is the second expression, **not simply b_t**. Normalization is per loss minibatch over its unroll elements; JAX uses population standard deviation. Returns are not standardized. \(\gamma=.995,\lambda=.97\). Sources: Brax `training/agents/ppo/losses.py:38–100,221–237`.

### Exact loss and clocks

With \(\rho_t=\exp[\log\pi_\theta(a_t|o_t)-\log\pi_{beh}(a_t|o_t)]\),

\[
L_{PPO}=-\mathbb E\min(\rho_t\hat A_t,\operatorname{clip}(\rho_t,0.9,1.1)\hat A_t)
+0.25\mathbb E(V_\nu(p_t)-v_t^\lambda)^2-0.001\mathbb E\widehat H_\theta.
\]

The value coefficient is **0.25**: code uses 0.5 times default `vf_loss_coefficient=.5`. Current T1 does not enable clipped-value loss. Sources: Brax `losses.py:245–275`, `train.py:207–213`; no override in `unified_formal.py:549–583`.

\[
L_{demo}=\mathbb E_{(o,a^T)\sim\mathcal D_T}\tfrac14\|\bar\pi_\theta(o)-\operatorname{sg}(a^T)\|_2^2,\qquad
L_{keep}=\mathbb E_{o\sim\mathcal D_K}\tfrac14\|\bar\pi_\theta(o)-\operatorname{sg}(\bar\pi_0(o))\|_2^2,
\]
\[
L_{student}=L_{PPO}+\lambda_D(u)L_{demo}+0.2L_{keep},\qquad
\lambda_D(u)=0.2+(0.05-0.2)\operatorname{clip}(u/128000,0,1).
\]

Demo and keep observations are sampled separately, with replacement, batch 256 each using their frozen sampling probabilities. They do not become PPO transitions, critic targets, or diffusion updates. Full Actor+Critic gradient is jointly clipped at norm .75 before Adam. Current T1 starts both arms from original pi0 Actor **and Critic**, with fresh Adam/RNG and whole normalizer frozen; Critic remains trainable. LR=7.5e-6; 128 worlds×25 steps=3200 collected transitions per rollout, eight minibatches of 16 unrolls, one pass, hence 10 rollouts and 80 actual Adam updates at 32k. The clock is completed physical transitions stamped before each SGD block, not BC update count or normalizer.count. Thus first block uses u=3200 and lambda=.19625; endpoint uses u=32000 and lambda=.1625, not .05. PPO_keep sets both demo endpoints to zero; other contracts match. Sources: `generative_bridge/student.py:23–42,127–150`; `policy_retention.py:7–10`; `generative_bridge/student_clock.py:6–16`; `retention_t1_runtime.py:13–29`; `retention_t1.py:289–304`; frozen T1_002 arm contracts.

The equal TRAIN reset law is \(\mathcal D_{reset}=\frac14(\mathcal D_{nominal}+\mathcal D_{random}+\mathcal D_{pi0-success}+\mathcal D_{teacher-recoverable})\); within each selected pool rows are uniform. Applies at real terminal resets as well as initialization. No online E pulse is injected in T1. The two snapshot pools retain their recorded RNG/context; complete-start pools receive new reset RNG. Sources: `retention_reset.py:6–78`; `retention_t1_runtime.py:32–52`.

## 6. What the feedback arrows mean in the manuscript

E→root is a real three-action intervention and saved context; G→demo is a finite physically evaluated prefix plus source tail; demo→student is deterministic-action MSE alongside on-policy PPO; student→E is delayed independently evaluated outcome credit on current collection actions; student→G is admitted successful **adopted-Actor** training trajectories. These arrows are implemented. Their existence does not demonstrate joint net capability growth or retention.

In the historical alternating round the source Actor and G inference state are frozen during collection/search, student training/evaluation completes before outcome feedback is assigned, E uses that collection exactly once, and G then updates from admitted successful data. Source: `generative_bridge/closed_loop.py:358–384`; `generative_bridge/explorer_admission.py:186–205`. T1 freezes E/G and therefore does not test whether learned exploration and feedback improve a complete new training run. Its negative result constrains the transfer/retention subproblem only. The multi-seed formal comparison and quality-feedback ablation remain proposed switches/protocols, not executed confirmation.

The endpoint success predicate, physical action map, and piecewise task reward belong in the manuscript's separately audited physical Appendix. They must remain distinct from E's discovery reward and the student's scaled learning reward. Neither success of the teacher, lower diffusion MSE, lower imitation loss, larger visitation extent, nor a positive training reward certifies independent policy improvement.

## 7. Final English manuscript review (2026-10-12)

Reviewed the revised `JIT_GENERATIVE_BRIDGE_MANUSCRIPT_EN.md`, Sections 3–5 and Appendices A–B, against the sources above. The previous substantive corrections are now present: neighbor validity is not an encoder input; DDIM recomputes residual noise after clean clipping and uses 99,…,4 with final signal coefficient one; original-root and self-visited qualification contracts are distinguished; student value targets use the current loss-call Critic; the value coefficient is .25; timeout preprocessing is described as implemented; the global-gradient clip no longer inserts an invented epsilon; and the diffusion schedule, E clipped-value/latent-entropy loss, and Noise process are included. Appendix A correctly identifies its angular-rate penalty as the world-frame angular-velocity sensor component rather than an Euler-angle derivative.

Two remaining corrections were sent to the manuscript owner before finalization:

1. **Feedback ablation table:** fixed-random exploration M2 cannot receive an effective student→E parameter update. Its remaining feedback is successful-student→G. The Noise teacher M3 has no learned G to update, so its remaining feedback is student-quality→E. Both entries currently saying “Both paths” contradict the stated interventions. This is a causal-description correction; it does not change the proposed protocols.
2. **Reward time indexing:** Eq. (1) uses `chi_t --a_t→ chi_(t+1)`, whereas the physical signals in Eq. (50) are read after executing the action (`unified_env.py:548–562,607–630`). Posture/height/velocity/angular velocity/actuator-force/joint-velocity and the updated upstream jump flag therefore need t+1 indices (or an explicit separate post-transition notation), while the action difference remains a_t−a_(t−1). Event indicators denote events on that transition.

Formula-completeness edits also sent: specify the finite-unroll boundary A_T^GAE=0 and the terminal bootstrap V_T; identify Eq. (58)'s g_P as the clipped policy surrogate gradient if matching recorded diagnostics, with entropy separate (`generative_bridge/student.py:153–165`); replace “log-count” with log(1+count) (`neighborhood.py:211`); give the AR recursion's domain ell=1,…,15; and include source-recheck≠1 in Eq. (12)'s positive-credit indicator if the equation is intended to stand without its subsequent cancellation sentence. Existing prose correctly states the cancellation.

The review found no additional claim that T1 proves complete-loop improvement, no inferred teacher→student success, and no use of numerical repeats as independent training seeds. The formal comparisons remain explicitly pending. These are source-level checks, not new numerical or physical validation.

**Final readback closure:** All seven items above have now been corrected in the manuscript readback with SHA-256 `8f5595c4072413e18f696a9107a15e580e7dc592a2f598dfafaebd5b0448825b`: Table 1 gives the one applicable feedback path for M2/M3; Eq. (50) uses post-transition physical signals and Eq. (51) defines transition events; Appendix B states the finite-unroll boundary; Eq. (58) separates the policy surrogate and entropy gradients; neighbor counts are log(1+count); the Noise recursion domain is explicit; Eq. (12) defines q_i using acceptance and the source-recheck exclusion. No mathematical or causal-description blocking issue remains from this method audit. Later editorial/source changes require their own delivery hash; this closure applies to the stated readback, not an unobserved future rendering.
