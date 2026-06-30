# Hierarchical Multi-Agent MCTS for Safety-Critical Coordination in Mixed-Autonomy Roundabouts

**Zhihao Lin¹, Jianglin Lan¹§, Shuo Liu², Zhen Tian¹, Dezong Zhao¹, and Chongfeng Wei¹**

¹ James Watt School of Engineering, University of Glasgow, Glasgow G12 8QQ, United Kingdom  
² Boston University, Brookline, MA, USA  
§ Corresponding author: Jianglin.Lan@glasgow.ac.uk

*arXiv:2509.01856v4 [eess.SY] 24 Feb 2026*

---

## Abstract

Navigating unsignalized roundabouts in mixed-autonomy traffic presents significant challenges due to dense vehicle interactions, lane-changing complexities, and behavioral uncertainties of human-driven vehicles (HDVs). This paper proposes a safety-critical decision-making framework for connected and automated vehicles (CAVs) navigating dual-lane roundabouts alongside HDVs. We formulate the problem as a multi-agent Markov Decision Process and develop a hierarchical safety assessment mechanism that evaluates three critical interaction types: CAV-to-CAV (C2C), CAV-to-HDV (C2H), and CAV-to-Boundary (C2B). A key contribution is our lane-specific uncertainty model for HDVs, which captures distinct behavioral patterns between inner and outer lanes, with outer-lane vehicles exhibiting 2.3× higher uncertainty due to less constrained movements. We integrate this safety framework with a multi-agent Monte Carlo Tree Search (MCTS) algorithm that employs safety-aware pruning to eliminate high-risk trajectories while maintaining computational efficiency. The reward function incorporates Shapley value-based credit assignment to balance individual performance with group coordination. Extensive simulation results validate the effectiveness of the proposed approach under both fully autonomous (100% AVs) and mixed traffic (50% AVs + 50% HDVs) conditions. Compared to benchmark methods, our framework consistently reduces trajectory deviations across all AVs and significantly lowers the rate of Post-Encroachment Time (PET) violations, achieving only 1.0% in the fully autonomous scenario and 3.2% in the mixed traffic setting.

**Index Terms:** Autonomous vehicles, decision making, mixed traffic, Monte Carlo tree search, risk assessment

---

## I. Introduction

Navigation at roundabouts presents unique challenges for Connected and Autonomous Vehicles (CAVs), particularly in mixed traffic where both CAVs and human-driven vehicles (HDVs) must safely coordinate through complex circular geometries and multiple merging points. The difficulty lies in managing several critical interactions simultaneously while ensuring safety and efficiency in a highly dynamic setting with continuous merging, lane-changing, and exit decisions. This challenge is further amplified by intricate interaction patterns induced by roundabout-specific geometric constraints and yielding rules. Therefore, a comprehensive understanding of both deterministic CAV behaviors and uncertain HDV behaviors is essential.

Traditional approaches to roundabout management often rely on rule-based decision-making methods, which generate conflict-free navigation sequences through preset yielding regulations and lane assignment strategies. These methods struggle to capture the complex decision-making behaviors of human drivers, where yielding involves significant behavioral variability. Conventional priority-based strategies ensure safety by enforcing strict yielding rules, but may substantially increase the requirement for onboard communication quality. More sophisticated rule-based approaches incorporating machine learning and static- and dynamic-constraint-based optimization have been proposed, yet their effectiveness has only been verified in simple single-lane roundabouts.

The application of machine learning to roundabout navigation has advanced significantly, yet critical gaps remain between theory and practical deployment. While deep reinforcement learning methods, including multi-agent deep deterministic policy gradient approaches, have shown success in controlled simulations, their performance degrades when faced with the geometric diversity and behavioral uncertainty of real-world roundabouts. The core challenge lies not in learning capability itself, but in the mismatch between underlying assumptions and the requirements of safety-critical traffic coordination. Specifically, neural network-based approaches exhibit three key limitations. First, the exponential growth of the state-action space with vehicle density makes it computationally prohibitive to cover all safety-critical scenarios during training. Second, limited interpretability prevents real-time safety verification, a non-negotiable requirement in mixed autonomy. Third, the sim-to-real gap is severe in roundabouts due to the interplay between lane-changing decisions and continuous control, leading to unexpected behaviors when policies encounter novel traffic patterns.

In complex environments like roundabouts, vehicle interactions are often modeled using game-theoretic reasoning, aiming to capture both cooperative and competitive behaviors among drivers. Frameworks such as Nash and Stackelberg games enable multi-agent planning under strategic assumptions. However, real-world scenarios introduce significant deviations from these assumptions—human drivers may act irrationally, interpret gaps inconsistently, or delay exit decisions, leading to mismatches between model predictions and actual behavior. Furthermore, the non-uniqueness of game-theoretic solutions can hinder reliable coordination, particularly in multi-lane roundabouts where rapid decision-making is crucial for maintaining safety and efficiency.

Beyond learning and game-theoretic approaches, safety-critical control methods have also been widely investigated to provide formal safety guarantees. Representative examples include Control Barrier Functions (CBFs), reachability analysis, and formal methods. In contrast, our work addresses safety through a Monte Carlo Tree Search (MCTS) framework with hierarchical risk assessment, tailored to mixed traffic at unsignalized roundabouts.

MCTS has emerged as a promising approach by marrying the learning-based and game-theoretic methods for interactive navigation. Unlike traditional DRL which requires extensive offline training, MCTS can efficiently explore the action space through online planning, making it particularly suitable for the dynamic and geometry-dependent nature of roundabout environments. The algorithm's inherent ability to balance exploration and exploitation makes it suitable for handling the uncertainties in mixed roundabout traffic, where vehicles are required to continuously make decisions about lane positioning, gap acceptance, and exit timing. However, current MCTS implementations often fall short in addressing comprehensive safety considerations and face significant scalability challenges in multi-agent scenarios involving multiple lanes and exit options.

This paper introduces a safety-critical multi-agent MCTS framework for coordinating mixed traffic at dual-lane roundabouts. The framework addresses the unique challenges posed by roundabout geometry, including lane-specific uncertainty modeling, exit proximity effects, and complex interactions between inner and outer lane vehicles. Unlike prior MCTS-based planners that apply uniform safety checks or treat all traffic participants identically, our approach introduces a structurally differentiated design: safety assessment, uncertainty modeling, and reward allocation are tailored to the specific interaction type (C2C, C2H, and C2B) and lane context, enabling these components to reinforce each other through hierarchical integration rather than operating as independent modules.

**The main contributions are summarized as follows:**

- We propose a safety-critical multi-agent MCTS framework for dual-lane roundabouts, unifying safety pruning, uncertainty modeling, and reward shaping into an interaction-aware tree search pipeline for mixed traffic.
- We develop a structurally differentiated safety assessment mechanism that handles various interactions via lane-dependent risk metrics and conservative thresholds for human uncertainty, validated through ablation studies.
- We introduce two key structural components for robust coordination: (i) an adaptive HDV prediction model capturing lane-specific uncertainty and exit proximity effects, and (ii) a Shapley value-based reward allocation scheme enabling contribution-aware credit assignment and reducing collision rates in cooperative roundabout navigation.

---

## II. Safety-Critical Decision-Making Framework

We consider unsignalized dual-lane roundabouts, where CAVs must navigate among other CAVs and HDVs without traffic signals. This task requires handling both predictable autonomous behaviors and uncertainty in human driving patterns.

To model this interaction-rich environment, we cast the problem as a multi-agent Markov Decision Process (MDP), formally defined as ⟨S, A, T, R, γ⟩. Here, S represents the joint state space capturing critical vehicle-level information such as positions, velocities, and heading angles; A defines the joint action space, including bounded control inputs for acceleration and steering; T describes the transition dynamics; R is the reward function guiding agent behavior; and γ is the discount factor used for long-term planning.

The main complexity arises from three intertwined interaction types: C2C, C2H, and C2B interactions. Within this framework, the goal is to determine an optimal policy π* that ensures both safety and efficiency in navigation. Denoting π ∈ Π as a policy in the admissible space, s_t as the system state at time step t, and R(s_t, π(s_t)) as the instantaneous reward, the optimization problem can be expressed as:

$$\pi^* = \arg\max_{\pi \in \Pi} \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \mathcal{R}(s_t, \pi(s_t))\right] \tag{1}$$

The architecture is composed of four key modules. First, the multi-agent MDP model is established, including formal definitions of states, actions, and dynamic models with embedded safety constraints (Sec. III-A). Next, we introduce a hierarchical safety mechanism that considers layered safety constraints across vehicle types and environmental boundaries, using dynamically updated safety thresholds and predictive risk assessment for HDVs (Sec. III-B). Building on this foundation, we develop a safety-aware multi-agent MCTS algorithm with safety-encoded tree nodes, UCB-based exploration–exploitation balancing, and rollout-based policy extraction with backpropagation (Sec. IV-A). Finally, we formulate a multi-objective reward function that jointly accounts for A2A, A2H, and A2R safety, motion efficiency, and dynamic feasibility, yielding an integrated optimization scheme (Sec. IV-B).

### Table I: Summary of Key Notation

| Symbol | Description |
|--------|-------------|
| **State and Control** | |
| s_i = [r_i, θ_i, v_i, φ_i, l_i]ᵀ | State vector of vehicle i |
| u_i = [a_i, φ̇_i, δ_i]ᵀ | Control input of vehicle i |
| C, H | Set of CAVs / HDVs |
| N, M | Number of CAVs / HDVs |
| **Safety Assessment** | |
| d(s_i, s_j) | Min. distance between vehicles i, j |
| d_safe(s_i, s_j) | Adaptive safe distance |
| d_min | Min. base safe distance |
| d_c2b(s_i) | Min. distance to roundabout boundary |
| Q^cc_risk, Q^ch_risk | C2C / C2H overall risk metric |
| Q^cc_th, Q^ch_th | Safety thresholds for C2C / C2H |
| C_ih | C2H collision probability |
| **HDV Uncertainty Modeling** | |
| f_IDM(s_h) | IDM nominal prediction for HDV h |
| Σ^lane_h | Lane-specific covariance matrix |
| M_lane(l_h) | Lane multiplier matrix |
| M_exit(E_h) | Exit proximity multiplier |
| **MCTS and Reward** | |
| N_n, Q_n | Node visit count / cumulative reward |
| e_exp | UCB exploration constant |
| K | Number of MCTS iterations |
| φ_i | Shapley value of agent i |
| λ^t_i | Cooperation coefficient |

---

## III. Safety-Critical Decision Making System

### A. Multi-Agent MDP Formulation for Dual-Lane Roundabout

We formulate the dual-lane roundabout navigation problem involving N CAVs and M HDVs as a multi-agent MDP: ⟨S, A, T, R, γ⟩. For the i-th vehicle, the state vector s_i consists of its polar coordinates (r_i, θ_i), velocity v_i, heading angle φ_i, and lane index l_i. The control input u_i includes acceleration command a_i, steering rate φ̇_i, and lane-changing decision δ_i. The joint state and action spaces are defined as:

$$\mathcal{S} = \prod_{i=1}^{N+M} \left\{ s_i = \begin{bmatrix} r_i \\ \theta_i \\ v_i \\ \phi_i \\ l_i \end{bmatrix} \in \mathbb{R}^5 \;\middle|\; \begin{array}{l} r_i \in [r_{\text{inner}}, r_{\text{outer}}], \\ \theta_i \in [0, 2\pi), \\ v_i \in [0, v_{\max}], \\ \phi_i \in [-\pi, \pi], \\ l_i \in \{0, 1\} \end{array} \right\} \tag{2}$$

$$\mathcal{A}_i = \left\{ u_i = \begin{bmatrix} a_i & \dot{\phi}_i & \delta_i \end{bmatrix}^\top \in \mathbb{R}^3 \;\middle|\; \begin{array}{l} |a_i| \leq a_{\max}, \\ |\dot{\phi}_i| \leq \dot{\phi}_{\max}, \\ \delta_i \in \{-1, 0, 1\} \end{array} \right\} \tag{3}$$

where r_inner and r_outer are the inner and outer lane radii, l_i = 0 represents the inner lane, l_i = 1 represents the outer lane, and δ_i denotes the lane-changing decision (−1: move inward, 0: maintain, 1: move outward). The roundabout navigation decisions must satisfy the following safety constraints on state transitions and inter-vehicle distances:

$$\mathcal{S}_{\text{safe}} = \{s \in \mathcal{S} \mid d(s_i, s_j) \geq d_{\text{safe}}, \forall i, j \in \mathcal{C} \cup \mathcal{H}\}$$

$$d(s_i, s_j) = \min_{p_i \in P(s_i),\, p_j \in P(s_j)} \|p_i - p_j\|_2 \tag{4}$$

Navigation decisions must also satisfy constraints over T:

$$s_{t+1} \in \mathcal{S}_{\text{safe}},\; \forall t \in [0, T],\quad |v_i| \leq v_{\max},\quad d_{c2b}(s_i) \geq d_{\min} \tag{5}$$

The CAV state transitions can be precisely calculated using control inputs and dynamics as follows:

$$s_{t+1} = \Phi(s_t, u_t) \tag{6}$$

with the transition function Φ(s_t, u_t) = {g_i(s_{i,t}, u_{i,t})}^N_{i=1}. The vehicle kinematic model of vehicle i ∈ C is:

$$g_i(s_{i,t}, u_{i,t}) = \begin{bmatrix} \text{sat}[r_{\text{inner}}, r_{\text{outer}}]\!\left(r_{i,t} + \frac{v_{i,t}\sin(\phi_{i,t})}{r_{i,t}}\Delta t\right) \\ \text{wrap}[0,2\pi]\!\left(\theta_{i,t} + \frac{v_{i,t}\cos(\phi_{i,t})}{r_{i,t}}\Delta t\right) \\ \text{sat}[0, v_{\max}](v_{i,t} + a_{i,t}\Delta t) \\ \text{wrap}[-\pi, \pi](\phi_{i,t} + \dot{\phi}_{i,t}\Delta t) \\ \text{LC}(l_{i,t}, \delta_{i,t}, Z_{i,t}) \end{bmatrix} \tag{7}$$

where Δt is the time step, sat[a,b](·) keeps values within bounds, wrap[a,b](·) handles angle continuity, and LC(l_{i,t}, δ_{i,t}, Z_{i,t}) is the lane-changing function that depends on current lane l_{i,t}, decision δ_{i,t}, and safety conditions Z_{i,t}.

### B. Hierarchical Safety Assessment for Roundabout Navigation

To ensure safe navigation in dual-lane roundabouts, we develop a hierarchical safety assessment framework that evaluates three critical interaction types.

#### Table II: Safety Distance Adjustment Factors

| Factor | Expression | Purpose |
|--------|-----------|---------|
| α_v | 1 + β_v\|Δv_ij\|/v_ref | Velocity adjustment |
| α_φ | 1 + β_φ\|Δφ_ij\|/π | Heading adjustment |
| α_l | 1 + β_l\|l_i − l_j\| | Lane difference |
| α_z | 1 + Σ_Ω 𝟙_Ω(s_i, s_j) | Zone-based risk |

#### 1) C2B Safety Assessment

The C2B safety assessment focuses on spatial constraints by partitioning the roundabout environment into the interaction area Ω_int and approach/exit areas Ω_app:

$$\Omega_{\text{int}} = \{s \in \mathcal{S} \mid r_{\text{inner}} \leq r \leq r_{\text{outer}}\}$$
$$\Omega_{\text{app}} = \{s \in \mathcal{S} \mid r < r_{\text{inner}} \lor r > r_{\text{outer}}\} \tag{8}$$

The safety level is evaluated through the minimum distance to roundabout boundaries d_c2b(s_i) and its corresponding penalty function φ_c2b(d):

$$d_{c2b}(s_i) = \min_{p \in P(s_i)} \text{distance}(p,\, \partial\Omega_{\text{circ}} \cup \partial\Omega_{\text{appr}})$$

$$\phi_{c2b}(d) = \begin{cases} -\infty, & \text{if } d \leq d_{\min} \\ -\beta\!\left(\dfrac{d_{\min}}{d}\right)^2, & \text{if } d_{\min} < d \leq d_{\text{safe}} \\ 0, & \text{if } d > d_{\text{safe}} \end{cases} \tag{9}$$

#### 2) C2C and C2H Safety Assessment

The adaptive safety distance accounts for multiple risk factors via a base value and adjustment terms:

$$d_{\text{safe}}(s_i, s_j) = \max\{d_{\min},\, \kappa_v|\Delta v_{ij}|\} \cdot \prod_{k \in K} \alpha_k(s_i, s_j) \tag{10}$$

Based on the distance measure in (4) and safety threshold in (10), we define two complementary risk metrics.

The **instantaneous risk** captures immediate collision threats:

$$r_{\text{inst}}(s_i, s_j) = \exp\!\left(\frac{d_{\text{safe}} - d_{\min}}{d_{\text{safe}}}\right) \cdot \left(1 + \frac{|\Delta v_{ij}|}{v_{\max}}\right) \tag{11}$$

The **temporal risk** aggregates future collision probabilities with time discounting:

$$R_T(s_i, s_j) = \frac{1}{T}\sum_{t=1}^{T} \frac{1}{1+t} \cdot \rho(d_t, d_{\text{safe}}) \tag{12}$$

where ρ(d, d_safe) = max{0, (1 − d/d_safe)²}.

For C2C interactions, the overall safety level is:

$$Q^{cc}_{\text{risk}}(s_i, s_j) = w^{cc}_1\, r_{\text{inst}}(s_i, s_j) + w^{cc}_2\, R_T(s_i, s_j), \quad i, j \in \mathcal{C} \tag{13}$$

#### 3) Lane-Specific HDV Uncertainty Modeling

For C2H interactions, we adopt the Intelligent Driver Model (IDM) as the nominal behavior predictor and embed it within a probabilistic framework to account for behavioral deviations. The uncertainty model is:

$$P(\hat{s}_h \mid s_h) = \mathcal{N}\!\left(f_{\text{IDM}}(s_h),\, \Sigma^{\text{lane}}_h(t, l_h, E_h)\right) \tag{14}$$

$$\Sigma^{\text{lane}}_h(t, l_h, E_h) = \Sigma_{\text{base}}(t) \cdot M_{\text{lane}}(l_h) \cdot M_{\text{exit}}(E_h) \tag{15}$$

The lane-specific multiplier matrix is:

$$M_{\text{lane}}(l_h = 0) = \text{diag}(0.3,\, 0.5,\, 0.4,\, 0.6,\, 0.2) \tag{16}$$

$$M_{\text{lane}}(l_h = 1) = \text{diag}(1.2,\, 1.5,\, 1.3,\, 1.4,\, 1.8) \tag{17}$$

The scaling factors are motivated by the geometric asymmetry of roundabout lanes: inner-lane vehicles are constrained by a tighter turning radius that naturally limits lateral and heading deviations, whereas outer-lane vehicles have greater freedom for radial displacement and exit maneuvers.

The exit proximity multiplier M_exit(E_h) accounts for increased uncertainty near exits:

$$M_{\text{exit}}(E_h) = I + \sum_{k=1}^{N_{\text{exit}}} \xi_k \exp\!\left(-\frac{|\theta_h - \theta_{\text{exit},k}|^2}{2\sigma^2_{\text{exit}}}\right) J_k \tag{18}$$

The base covariance matrix Σ_base(t) is defined as:

$$\Sigma_{\text{base}}(t) = \begin{bmatrix} \sigma^2_r t + \epsilon^2_r t^2 & 0 & \rho_{rv}\sigma_r\sigma_v t & 0 & 0 \\ 0 & \sigma^2_\theta t + \epsilon^2_\theta t^2 & 0 & \rho_{\theta\phi}\sigma_\theta\sigma_\phi t & 0 \\ \rho_{rv}\sigma_r\sigma_v t & 0 & \sigma^2_v & 0 & 0 \\ 0 & \rho_{\theta\phi}\sigma_\theta\sigma_\phi t & 0 & \sigma^2_\phi & 0 \\ 0 & 0 & 0 & 0 & \sigma^2_l \end{bmatrix} \tag{19}$$

Over a 12-second horizon, outer-lane radial uncertainty reaches approximately 5.8σ compared to 2.5σ for the inner lane, while angular uncertainties show similar disparities (4.0σ outer vs. 2.0σ inner).

We bound HDVs' reachable state space as:

$$\mathcal{S}^t_h = \left\{ \hat{s}_h \in \mathbb{R}^5 \;\middle|\; \begin{array}{l} \|p_h - p_h(t)\| \leq (v_{\max} + \sigma_v)t, \\ |v_h| \leq v_{\max} + 2\sigma_v, \\ |\phi_h| \leq \pi, \\ l_h \in \{0, 1\} \end{array} \right\} \tag{20}$$

The collision probability for C2H interactions is:

$$C_{ih} = \int_{\hat{s}_h \in \mathcal{S}^t_h} \psi(\hat{s}_i, \hat{s}_h) \cdot \mathcal{N}\!\left(f_{\text{IDM}}(s_h), \Sigma^{\text{lane}}_h\right) d\hat{s}_h \tag{21}$$

The safety level for C2H interactions is:

$$Q^{ch}_{\text{risk}}(s_i, s_h) = w^{ch}_1\, r_{\text{inst}}(s_i, \hat{s}_h) + w^{ch}_2\, R_T(s_i, s_h) + w^{ch}_3\, C_{ih} \tag{22}$$

---

## IV. Multi-Agent MCTS Solution Approach

### A. Multi-Agent MCTS for Roundabout Navigation

Building on the safety assessment framework, we propose a structured tree search approach where the risk assessment functions in (13) and (22) are used to evaluate safety at each node and prune unsafe nodes that exceed predefined safety thresholds (Q^cc_th for C2C interactions and Q^ch_th for C2H interactions).

Let T be the search tree whose node n ∈ T is defined as: n = (d_n, p_n, C_n, N_n, Q_n, u_n, ξ_n). A node is considered safe (ξ_n = "safe") if:
- Q^cc_risk ≤ Q^cc_th for C2C interactions
- Q^ch_risk ≤ Q^ch_th for C2H interactions
- d_c2b ≥ d_min

We set the safety thresholds to Q^cc_th = 0.8 (C2C) and Q^ch_th = 0.6 (C2H), using a more conservative margin for human-driven interactions due to higher prediction uncertainty. Following standard surrogate safety analysis, we adopt a PET threshold of 1.0 s.

The safety-critical multi-agent MCTS algorithm uses a depth-first search strategy with safety validation at each node expansion:

$$\text{Search}(n_t) = \begin{cases} \text{Expand}(n_t) \cup \text{Rollout}(n_t), & \text{if new \& safe} \\ \text{Search}(\text{UCB}(n_t)), & \text{if visited} \\ \text{Terminate}, & \text{if unsafe} \end{cases}$$

The selection of nodes for expansion is governed by the UCB formula:

$$\text{UCB}(n) = \frac{Q_n}{N_n} + e_{\text{exp}}\sqrt{\frac{\ln N_{p_n}}{N_n}} \tag{23}$$

### B. Roundabout-Specific Reward Function Design

#### 1) Group-Aware Reward Formulation

We adopt a coalition-based design using the Shapley value φ_i to distribute the global reward among cooperative vehicles:

$$\phi_i(v) = \sum_{c \subset \mathcal{C} \cup \mathcal{H},\, i \in c} \frac{(|c|-1)!(n-|c|)!}{n!}\left[v(c) - v(c \setminus \{i\})\right]$$

The global reward at each timestep is then defined as:

$$\mathcal{R}(\mathbf{s}_t, \mathbf{u}_t) = \sum_{i \in \mathcal{C}} \phi_i \cdot \mathcal{R}_i(\mathbf{s}_t, \mathbf{u}_t) \tag{24}$$

Each CAV's individual reward R_i balances its own performance and the impact on others:

$$\mathcal{R}_i = \frac{Q^{\text{self}}_i + \lambda^t_i Q^{\text{other}}_i}{1 + \lambda^t_i(N-1)} \tag{25}$$

where Q^self_i = Q^i_safety + Q^i_eff + Q^i_comfort + Q^i_lane, and Q^other_i = Σ_{j≠i}(Q^j_safety + Q^j_eff + Q^j_lane).

#### 2) Reward Components

Each CAV's self-reward comprises four components:

$$Q^i_{\text{safety}} = -w_{c2c} Q^{cc}_{\text{risk}}(s_i) - w_{c2h} Q^{ch}_{\text{risk}}(s_i) - w_{c2b}\phi_{c2b}$$

$$Q^i_{\text{eff}} = -w_v(v_i - v_{\text{des}})^2 - w_p\|p_i - p_{\text{ref}}\|^2$$

$$Q^i_{\text{comfort}} = -w_a|a_i|^2 - w_\phi|\dot{\phi}_i|^2$$

$$Q^i_{\text{lane}} = Q^i_{\text{position}}(l_i) - w_{\text{trans}}|\delta_i|^2 + Q^i_{\text{exit}}(\theta_i) \tag{26}$$

### C. Optimization Policy

The optimality of our MCTS solution for roundabout navigation can be formally characterized by:

$$u^* = \arg\max_{u \in \mathcal{A}_{\text{joint}}} \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t_r \sum_{i \in \mathcal{C}} \phi_i \mathcal{R}_i(s_t, u(s_t)) \;\middle|\; u\right] \tag{27}$$

subject to the following constraints at each time step t:

**Safety Constraints:**
- (C2C) Q^cc_risk(s_i, s_j) ≤ Q^cc_th, ∀i, j ∈ C
- (C2H) Q^ch_risk(s_i, s_h) ≤ Q^ch_th, ∀i ∈ C, h ∈ H
- (C2B) d_c2b(s_i) ≥ d_min, ∀i ∈ C

**Dynamic Constraints:**
- (Velocity) v_i ∈ [0, v_max], ∀i ∈ C
- (Acceleration) |a_i| ≤ a_max, ∀i ∈ C
- (Steering) |φ̇_i| ≤ φ̇_max, ∀i ∈ C

### D. Computational Complexity Analysis

The worst-case overall complexity across K MCTS iterations can be coarsely estimated as:

$$\mathcal{O}\!\left(K \cdot |\mathcal{A}|^N \cdot d_{\max} \cdot NM \cdot d^2\right) \tag{29}$$

which grows exponentially with the number of agents. In practice, however, the actual computational cost is significantly lower due to: (1) safety pruning that eliminates unsafe branches early; (2) UCB-guided selective search that limits unnecessary expansions; and (3) matrix-based parallel implementation of safety and reward computations.

### Table III: Computational Complexity Comparison

| Method | Complexity | Operations |
|--------|-----------|------------|
| Joint exhaustive search | \|A\|^(N·H) | ≈ 10^85 |
| Game-theoretic Nash | O(\|A\|^(NH) · I_Nash) | > 10^87 |
| MCTS (no pruning) | K · \|A\| · N · H | ≈ 3.2 × 10^5 |
| MCTS + safety pruning | (1 − ρ) · K · \|A\| · N · H | ≈ 2.2 × 10^5 |

The average planning time is approximately 58 ms per decision step, well within a typical planning cycle of Δt = 0.2 s. Safety pruning reduces the effective search space by approximately 1.4× compared to unpruned MCTS.

---

## V. Experimental Evaluation

Simulations are conducted in MATLAB 2024a to evaluate the proposed approach for safe and efficient autonomous driving at a signal-free, dual-lane roundabout. We compare the proposed method with several advanced optimization algorithms, including the Stackelberg game approach and the Nash equilibrium method.

### A. Case 1: Dual-Lane Roundabout (ROP = 100%)

The experimental evaluation begins with a baseline scenario featuring a 100% rate of penetration (ROP) at a signal-free, dual-lane roundabout. Four CAVs simultaneously approach the roundabout from different directions, creating a complex multi-agent coordination challenge.

The vehicles maintain stable speeds ranging from 3 to 5 m/s, requiring only minimal speed modulation for safe interaction. Our approach consistently yields smaller trajectory deviations from the reference trajectories compared to Nash and Stackelberg baselines.

The proposed method with a maximum tree depth of d_max = 8 delivers the best outcomes, with no instances of PET falling below the critical threshold of 1.0 second. In comparison:
- Baseline approach: 8.8% violations
- Nash method: 13.2% violations
- Stackelberg method: 26.1% violations

#### Table IV: Comparison of Algorithm Performances in Case 1

| Methods | Average Speed (m/s) | Average Trajectory Deviation (m) | Collision Rate (%) |
|---------|--------------------|---------------------------------|-------------------|
| Stackelberg | 5.22 ± 1.34 | 0.89 ± 0.84 | 16.0 |
| Nash | 5.16 ± 1.01 | 0.74 ± 0.59 | 11.0 |
| Baseline | 5.44 ± 1.10 | 0.65 ± 0.47 | 13.0 |
| d_max = 3 | 5.27 ± 1.03 | 0.57 ± 0.38 | 4.0 |
| **d_max = 8** | **5.31 ± 0.89** | **0.45 ± 0.39** | **0.0** |

### B. Case 2: Dual-Lane Roundabout (ROP = 50%)

To assess robustness under mixed traffic conditions, we conduct experiments with a 50% CAV penetration rate, where four CAVs and four HDVs coexist and interact at the roundabout.

The velocity trajectories exhibit greater variability—ranging between 3–5 m/s—and wider confidence intervals, reflecting the influence of HDV-induced uncertainty. Our method with d_max = 8 achieves 3.1% PET violations compared to:
- Baseline: 13.3% violations
- Nash: 15.2% violations
- Stackelberg: 28.0% violations

#### Table V: Comparison of Algorithm Performances in Case 2

| Methods | Average Speed (m/s) | Average Trajectory Deviation (m) | Collision Rate (%) |
|---------|--------------------|---------------------------------|-------------------|
| Stackelberg | 5.25 ± 1.31 | 1.14 ± 1.06 | 33.0 |
| Nash | 5.12 ± 1.24 | 0.85 ± 0.68 | 29.0 |
| Baseline | 4.83 ± 0.82 | 0.81 ± 0.65 | 17.0 |
| d_max = 3 | 5.16 ± 0.95 | 0.76 ± 0.62 | 5.0 |
| **d_max = 8** | **5.23 ± 0.94** | **0.63 ± 0.59** | **2.0** |

**Effect of CAV Penetration Rate:** At low penetration rates (20%–33.3%), PET distributions are more variable with low violation rates (1.0%–3.7%). In the medium range (42.8%–57.1%), violation rates initially rise to 3.7% at 42.8%, then stabilize around 3.2%. At high penetration rates (66.7%–100%), violation rates drop to 0.9% and PET distributions narrow.

### C. Sensitivity and Ablation Analysis

#### Table VI: Sensitivity and Ablation Analysis

| Setting | Collision Rate (%) | PET Viol. Rate (%) | Traj. Dev. (m) | Min. Distance (m) |
|---------|-------------------|--------------------|----------------|-------------------|
| **Min. Safe Distance d_min (Default d_min = 2.0 m)** | | | | |
| d_min = 1.0 m | 9.0 | 8.2 ± 0.9 | 0.56 ± 0.39 | 1.31 ± 0.74 |
| **d_min = 2.0 m** | **2.0** | **3.1 ± 0.7** | **0.73 ± 0.61** | **2.72 ± 0.93** |
| d_min = 3.0 m | 1.0 | 2.4 ± 0.5 | 1.42 ± 0.85 | 3.61 ± 1.12 |
| **Reward Allocation Strategy (Default Shapley)** | | | | |
| Equal sharing | 14.0 | 10.2 ± 2.1 | 1.63 ± 0.85 | 1.12 ± 0.71 |
| Distance-weighted | 7.0 | 5.3 ± 1.6 | 1.12 ± 0.78 | 2.36 ± 0.87 |
| **Shapley value** | **2.0** | **3.1 ± 0.7** | **0.73 ± 0.61** | **2.71 ± 0.93** |
| **HDV Uncertainty Model (Default Lane-Specific)** | | | | |
| Deterministic IDM | 0.0 | 1.9 ± 0.5 | 0.52 ± 0.43 | 2.54 ± 0.98 |
| Lane-agnostic | 11.0 | 8.7 ± 1.8 | 1.66 ± 0.82 | 1.25 ± 0.67 |
| **Lane-specific** | **2.0** | **3.1 ± 0.7** | **0.73 ± 0.61** | **2.71 ± 0.93** |

**Key findings:**

1. **Minimum Safe Distance:** The default d_min = 2.0 m achieves an effective balance between safety and efficiency, maintaining a low collision rate (2.0%) with moderate trajectory deviations (0.73 m).

2. **Reward Allocation Strategy:** The Shapley value method significantly outperforms both alternatives, reducing collision rates to 2.0% and achieving the best minimum distance (2.71 m). Equal sharing, which ignores individual contributions, leads to the highest collision rate (14.0%).

3. **HDV Uncertainty Model:** The deterministic IDM achieves zero collisions in simulation, but offers no robustness against real-world behavioral deviations. The lane-agnostic model paradoxically increases collisions to 11.0% due to excessive conservatism causing hesitation and deadlock. The lane-specific model reduces collisions to 2.0% while providing structured robustness margins calibrated to lane-dependent behavioral variability.

---

## VI. Conclusion

This paper introduces a safety-critical decision-making framework for autonomous vehicles navigating unsignalized, dual-lane roundabouts by integrating Monte Carlo Tree Search (MCTS) with a hierarchical risk assessment strategy. The framework offers three major innovations: a multi-agent MCTS structure for scalable and efficient action space exploration, a hierarchical safety assessment mechanism for robust spatiotemporal risk evaluation, and an adaptive reward function that effectively balances safety and efficiency.

Experimental results confirm the effectiveness of the proposed method under varying autonomous vehicle penetration rates. In fully autonomous settings (100% CAVs), the framework achieves reduced trajectory deviations and eliminates PET violations when compared to baseline approaches. In mixed traffic scenarios (50% CAVs + 50% HDVs), the framework delivers even greater improvements by reliably handling the uncertainty introduced by human drivers, while maintaining low deviation and high safety margins.

Sensitivity and ablation analyses further demonstrate robustness to key parameter choices and confirm that each proposed component—Shapley value allocation and lane-specific uncertainty modeling—contributes meaningfully to overall performance.

We note that the current framework assumes accurate state observations and plans independently per CAV without inter-vehicle communication. Future work will focus on improving computational scalability through parallelized search strategies, extending the framework to diverse roundabout geometries, and integrating data-driven behavior prediction with validation under realistic sensing noise and communication delays.

---

## References

[1] S. Alkheder, F. Al-Rukaibi, and A. Al-Faresi, "Driver behavior at kuwait roundabouts and its performance evaluation," *IATSS Research*, vol. 44, no. 4, pp. 272–284, 2020.

[2] L. Zhang, Y. Dong, H. Farah, and B. van Arem, "Social-aware planning and control for automated vehicles based on driving risk field and model predictive contouring control," in *Proc. IEEE SMC*, 2023, pp. 3297–3304.

[3] E. Galceran et al., "Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction," in *Proc. RSS*, 2015, p. 6.

[4] E. Polders et al., "Identifying crash patterns on roundabouts," *Traffic Inj. Prev.*, vol. 16, no. 2, pp. 202–207, 2015.

[5] C. Badue et al., "Self-driving cars: A survey," *Expert Syst. Appl.*, vol. 165, p. 113816, 2021.

[6] J. Pérez et al., "Autonomous driving manoeuvres in urban road traffic environment: A study on roundabouts," in *Proc. IFAC World Congress*, vol. 44, 2011, pp. 13795–13800.

[7] J. Choi and D.-K. Kim, "Calibration and validation of the rule-based human driver model for car-following behaviors at roundabout," *Asian Transp. Stud.*, vol. 10, p. 100129, 2024.

[8] Y. Shi et al., "Safety-enhanced behavioral decision strategy for intelligent vehicles under roundabout scenarios," *Inf. Sci.*, p. 122367, 2025.

[9] R. Azimi et al., "V2V-intersection management at roundabouts," *SAE Int. J. Passeng. Cars–Mech. Syst.*, vol. 6, pp. 681–690, 2013.

[10] X. Gong, P. Lyu, and B. Wang, "Cooperative motion planning and decision making for cavs at roundabouts," *IEEE Internet Things J.*, vol. 11, no. 19, pp. 32205–32220, 2024.

[11] Z. Lin et al., "A conflicts-free, speed-lossless kan-based reinforcement learning decision system for interactive driving in roundabouts," *IEEE Trans. Intell. Transp. Syst.*, vol. 25, pp. 1–14, 2025.

[12] S. Lu et al., "A multi-agent federated reinforcement learning-based cooperative vehicle-infrastructure control approach framework for roundabouts," *IEEE Trans. Intell. Transp. Syst.*, pp. 1–16, 2025.

[13] B. Peng et al., "Communication scheduling by deep reinforcement learning for remote traffic state estimation with bayesian inference," *IEEE Trans. Veh. Technol.*, vol. 71, no. 4, pp. 4287–4300, 2022.

[14] Z. Lin, J. Lan, and X. Zhao, "Kan-lstm enhanced multi-agent advantage actor-critic reinforcement learning for autonomous ramp merging," *IEEE Trans. Veh. Technol.*, pp. 1–12, 2025.

[15] P. Cai et al., "Dq-gat: Towards safe and efficient autonomous driving with deep q-learning and graph attention networks," *IEEE Trans. Intell. Transp. Syst.*, vol. 23, no. 11, pp. 21102–21112, 2022.

[16] Z. Tian et al., "Efficient and balanced exploration-driven decision making for autonomous racing using local information," *IEEE Trans. Intell. Veh.*, pp. 1–17, 2024.

[17] L. Ferrarotti et al., "Autonomous and human-driven vehicles interacting in a roundabout: A quantitative and qualitative evaluation," *IEEE Access*, vol. 12, pp. 32693–32705, 2024.

[18] Y. Zhang et al., "Adaptive decision-making for automated vehicles under roundabout scenarios using optimization embedded reinforcement learning," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 32, no. 12, pp. 5526–5538, 2021.

[19] F. Konstantinidis et al., "Parameter sharing reinforcement learning for modeling multi-agent driving behavior in roundabout scenarios," in *Proc. IEEE ITSC*, 2021, pp. 1974–1981.

[20] Z. Tian et al., "Balanced reward-inspired reinforcement learning for autonomous vehicle racing," in *Proc. L4DC*, 2024, pp. 628–640.

[21] J. Zhu et al., "Bi-level ramp merging coordination for dense mixed traffic conditions," *Fundam. Res.*, 2023.

[22] Z. Kherroubi et al., "Novel decision-making strategy for connected and autonomous vehicles in highway on-ramp merging," *IEEE Trans. Intell. Transp. Syst.*, vol. 23, no. 8, pp. 12490–12502, 2022.

[23] H. Wang et al., "Interpretable decision-making for autonomous vehicles at highway on-ramps with latent space reinforcement learning," *IEEE Trans. Veh. Technol.*, vol. 70, no. 9, pp. 8707–8719, 2021.

[24] N. Ding et al., "Multi-vehicle coordinated lane change strategy in the roundabout under internet of vehicles based on game theory," *IEEE Trans. Ind. Informat.*, vol. 16, no. 8, pp. 5435–5443, 2019.

[25] M. Pourabdollah et al., "Calibration and evaluation of car following models using real-world driving data," in *Proc. IEEE ITSC*, 2017, pp. 1–6.

[26] Y. Bie, Y. Ji, and D. Ma, "Multi-agent deep reinforcement learning collaborative traffic signal control method," *Transp. Res. Part C*, vol. 164, p. 104663, 2024.

[27] P. Hang et al., "Driving conflict resolution of autonomous vehicles at unsignalized intersections: A differential game approach," *IEEE/ASME Trans. Mechatron.*, vol. 27, no. 6, pp. 5136–5146, 2022.

[28] J. Zhang et al., "Distributed model-free sliding-mode predictive control of discrete-time second-order nonlinear multiagent systems with delays," *IEEE Trans. Cybern.*, vol. 52, no. 11, pp. 12403–12413, 2022.

[29] S. Liu et al., "Iterative convex optimization for model predictive control with discrete-time high-order control barrier functions," in *Proc. ACC*, 2023, pp. 3368–3375.

[30] S. Liu, W. Xiao, and C. Belta, "Feasibility-guaranteed safety-critical control with applications to heterogeneous platoons," in *Proc. IEEE CDC*, 2024, pp. 8066–8073.

[31] S. Bansal et al., "Hamilton-jacobi reachability: A brief overview and recent advances," in *Proc. IEEE CDC*, 2017, pp. 2242–2253.

[32] C. B. Browne et al., "A survey of monte carlo tree search methods," *IEEE Trans. Comput. Intell. AI Games*, vol. 4, no. 1, pp. 1–43, 2012.

[33] D. Lenz, T. Kessler, and A. Knoll, "Tactical cooperative planning for autonomous highway driving using monte-carlo tree search," in *Proc. IEEE IVS*, 2016, pp. 447–453.

[34] P. Zhou, X. Sun, and T. Chai, "Enhanced nmpc for stochastic dynamic systems driven by control error compensation with entropy optimization," *IEEE Trans. Control Syst. Technol.*, vol. 31, no. 5, pp. 2217–2230, 2023.

[35] J. Wurts, J. L. Stein, and T. Ersal, "Design for real-time nonlinear model predictive control with application to collision imminent steering," *IEEE Trans. Control Syst. Technol.*, vol. 30, no. 6, pp. 2450–2465, 2022.

[36] C. F. Hayes et al., "Risk aware and multi-objective decision making with distributional monte carlo tree search," *arXiv:2102.00966*, 2021.

[37] Z. Lin et al., "Safety-critical multi-agent mcts for mixed traffic coordination at unsignalized intersections," *IEEE Trans. Intell. Transp. Syst.*, pp. 1–15, 2025.

[38] P. Weingertner et al., "Monte carlo tree search with reinforcement learning for motion planning," in *Proc. ITSC*, 2020, pp. 1–7.

[39] M. Wang et al., "Speed planning for autonomous driving in dynamic urban driving scenarios," in *Proc. ECCE*, 2020, pp. 1462–1468.

[40] C.-K. Ho and C.-T. King, "Lac-rrt: Constrained rapidly-exploring random tree with configuration transfer models for motion planning," *IEEE Access*, vol. 11, pp. 97654–97663, 2023.

[41] Y. Gao et al., "Trajectory planning and tracking control of autonomous vehicles based on improved artificial potential field," *IEEE Trans. Veh. Technol.*, vol. 73, no. 9, pp. 12468–12483, 2024.

[42] R. Szczepanski, "Safe artificial potential field: Novel local path planning algorithm maintaining safe distance from obstacles," *IEEE Robot. Autom. Lett.*, vol. 8, no. 8, pp. 4823–4830, 2023.

[43] M. Treiber, A. Hennecke, and D. Helbing, "Congested traffic states in empirical observations and microscopic simulations," *Physical Review E*, vol. 62, no. 2, p. 1805, 2000.

[44] Y. Cui et al., "A game-theoretic framework of interaction and cooperative driving for cavs at mixed unsignalized intersections," *IEEE Internet of Things Journal*, vol. 13, no. 1, pp. 1524–1538, 2026.

[45] P. Hang et al., "Decision making of connected automated vehicles at an unsignalized roundabout considering personalized driving behaviours," *IEEE Trans. Veh. Technol.*, vol. 70, no. 5, pp. 4051–4064, 2021.

---

## Author Biographies

**Zhihao Lin** received the M.S. degree from the College of Electronic Science and Engineering, Jilin University, Changchun, China, in 2019. He is currently pursuing the Ph.D. degree with the James Watt School of Engineering, University of Glasgow, UK. His research interests focus on Structure-aware Reinforcement Learning and Multi-Agent Decision Making.

**Jianglin Lan** received the Ph.D. degree from the University of Hull in 2017. He has been a Leverhulme Early Career Fellow and Lecturer at the University of Glasgow since 2022. He was a Visiting Professor at the Robotics Institute, Carnegie Mellon University, in 2023. From 2017 to 2022, he held postdoc positions at Imperial College London, Loughborough University, and University of Sheffield. His research interests include AI, optimisation, control theory, and autonomy.

**Shuo Liu** (Student Member, IEEE) received his M.S. degree in Mechanical Engineering from Columbia University, New York, NY, USA, in 2020 and his B.Eng. degree in Mechanical Engineering from Chongqing University, Chongqing, China, in 2018. He is currently a Ph.D. candidate in Mechanical Engineering at Boston University, Boston, USA. His research interests include optimization, nonlinear control, deep learning, and robotics.

**Zhen Tian** received the B.Eng. degree in Electronic and Electrical Engineering from the University of Strathclyde, Glasgow, U.K., in 2020, and the Ph.D. degree from the College of Science and Engineering, University of Glasgow, Glasgow, U.K., in 2025. He is currently a postdoctoral researcher at the University of Glasgow. His research interests include interactive vehicle decision systems and autonomous racing strategies.

**Dezong Zhao** received the B.Eng. and M.S. degrees from Shandong University, Jinan, China, in 2003 and 2006, respectively, and the Ph.D. degree from Tsinghua University, Beijing, China, in 2010, all in Control Science and Engineering. He is a Reader in Autonomous Systems with the James Watt School of Engineering, University of Glasgow and a Turing Fellow with the Alan Turing Institute. He was awarded a Royal Society-Newton Advanced Fellow in 2020 and an EPSRC Innovation Fellow in 2018.

**Chongfeng Wei** received his Ph.D. degree in mechanical engineering from the University of Birmingham in 2015. He is now an Associate Professor (University Senior Lecturer) at University of Glasgow, UK. His current research interests include decision-making and control of intelligent vehicles, human-centric autonomous driving, cooperative automation, and dynamics and control of mechanical systems. He is also serving as an Associate Editor of IEEE TITS, IEEE TIV, IEEE TVT, and Frontier on Robotics and AI.
