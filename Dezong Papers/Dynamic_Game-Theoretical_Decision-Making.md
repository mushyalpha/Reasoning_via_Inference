# Dynamic Game-Theoretical Decision-Making Framework for Vehicle-Pedestrian Interaction with Human Bounded Rationality

**Meiting Dang, Dezong Zhao** *(Senior Member, IEEE)*, **Yafei Wang** *(Member, IEEE)*, **Chongfeng Wei\*** *(Member, IEEE)*

> \* Chongfeng Wei is the corresponding author.
>
> Meiting Dang, Dezong Zhao and Chongfeng Wei are with James Watt School of Engineering, University of Glasgow, Glasgow, G12 8QQ, United Kingdom (email: m.dang.1@research.gla.ac.uk, dezong.zhao@glasgow.ac.uk, chongfeng.wei@glasgow.ac.uk)
>
> Yafei Wang is with the School of Mechanical Engineering, Shanghai Jiao Tong University, Shanghai 200240, China (email: wyfjlu@sjtu.edu.cn)

*arXiv:2409.15629v1 [cs.RO] 24 Sep 2024*

---

## Abstract

Human-involved interactive environments pose significant challenges for autonomous vehicle decision-making processes due to the complexity and uncertainty of human behavior. It is crucial to develop an explainable and trustworthy decision-making system for autonomous vehicles interacting with pedestrians. Previous studies often used traditional game theory to describe interactions for its interpretability. However, it assumes complete human rationality and unlimited reasoning abilities, which is unrealistic. To solve this limitation and improve model accuracy, this paper proposes a novel framework that integrates the partially observable Markov decision process with behavioral game theory to dynamically model AV-pedestrian interactions at the unsignalized intersection. Both the AV and the pedestrian are modeled as dynamic-belief-induced quantal cognitive hierarchy (DB-QCH) models, considering human reasoning limitations and bounded rationality in the decision-making process. In addition, a dynamic belief updating mechanism allows the AV to update its understanding of the opponent's rationality degree in real-time based on observed behaviors and adapt its strategies accordingly. The analysis results indicate that our models effectively simulate vehicle-pedestrian interactions and our proposed AV decision-making approach performs well in safety, efficiency, and smoothness. It closely resembles real-world driving behavior and even achieves more comfortable driving navigation compared to our previous virtual reality experimental data.

**Index Terms:** Vehicle-pedestrian interaction, Decision-making, Behavioral game theory, Bounded rationality

---

## I. Introduction

Autonomous vehicle (AV) technology represents a transformative leap in the automotive sector, promising safer, more efficient, and more convenient transportation in the future [1]. As this technology advances and becomes increasingly integrated into practical applications, AVs will inevitably share the road with other road users, including pedestrians [2], [3]. However, the complexity and uncertainty inherent in human behavior present significant challenges for AV decision-making and motion planning, especially at unsignalized intersections where pedestrians are involved [4].

At these intersections, the priority of right-of-way is often ambiguous due to the absence of traffic signals, leading to potential road conflicts that further complicate AV's decision-making. Moreover, interactions between vehicles and pedestrians are interdependent and coupled. Pedestrians may exhibit various behaviors, like directly crossing the road or hesitating before crossing. In response, the AV must adjust its strategy accordingly. Conversely, AV actions can also influence pedestrian behavior, such as changing their crossing intention or walking speed based on the approaching AV's movements. Smooth interaction between AVs and pedestrians is essential on urban unsignalized roadways. Hence, this work focuses on AV decision-making in the AV-pedestrian interactions at an unsignalized intersection, where conflicts may arise.

Previous studies have investigated vehicle-pedestrian interactions, often relying on statistical methods [5], [6] or describing interactions as one-time events [7]. However, pedestrian behavior, characterized by unpredictability and dynamism, presents challenges for such approaches. Their movements can quickly change [8], introducing uncertainty into interactions that traditional methods struggle to capture. By considering uncertainty and dynamic interactions, the partially observable Markov decision process (POMDP) framework [9] provides a modeling approach for decision-making challenges that closely mirror real-world conditions. While widely applied to handle complex environments in vehicle-vehicle interactions, its potential in vehicle-pedestrian interactions remains underexplored.

Game theory is frequently used to model the interaction between vehicles and pedestrians. However, most studies assume that all players follow the Nash equilibrium [10], [11], possessing unlimited computational reasoning ability to compute optimal actions and perfect rationality to execute them, thus maximizing their utility function in decision-making. In reality, individuals often deviate from the Nash equilibrium due to cognitive limitations [12], unable to consistently calculate optimal actions or prone to make errors in complex scenarios. Hence, considering human reasoning levels and bounded rationality is essential to develop more accurate models of real-world behaviors.

To address these limitations, our study proposes a novel framework that combines POMDP and behavioral game theory to tackle the AV decision-making problem within complex and dynamic environments. Figure 1 shows our proposed framework for AV-pedestrian interaction at an unsignalized intersection. In this work, we employ the POMDP framework to dynamically model the decision-making process of the AV in an environment with incomplete information and uncertainty. Furthermore, we use a behavioral game theory model to describe AV-pedestrian interaction, both the AV and pedestrian modeled as dynamic-belief-induced quantal cognitive hierarchy (DB-QCH) models. At each time step, the AV model updates its beliefs about its opponent's reasoning level and rationality based on extended Bayesian Estimation. A trained neural network calculates the predicted optimal action, which is then translated to an action space using a Gaussian distribution function. An iterative reasoning model is established to deduce the optimal strategies for both oneself and the opponent at each level, computed via the Monte Carlo Tree Search (MCTS) method. While the pedestrian model is also constructed as a DB-QCH model, their action space remains fixed over time, which differs from the AV model.

To enhance humans' understanding of the interaction process and resolve intersection conflicts in a human-involved interactive environment, this work makes the following main contributions:

1. The POMDP framework and behavioral game theory are integrated to address the uncertainty and dynamic interaction between the AV and the pedestrian.
2. To accurately capture the decision-making processes, both the AV and pedestrian are modeled as DB-QCH models. This modeling approach provides a comprehensive understanding of interaction dynamics and facilitates more realistic simulations.
3. A trained neural network based on data from our previous experiments is developed to guide MCTS in exploring the continuous action space of AV, thereby facilitating effective and efficient decision-making.
4. This work introduces variables to quantify human bounded rationality and is the first to propose a dynamic updating mechanism for rational values based on the observed environment, enabling adaptive decision-making by AVs in real-time.

These concentrated efforts pave the way for an explainable and trustworthy AV decision-making system, leading to safer and more efficient navigation of AVs in such an interactive environment where humans are involved.

---

## II. Related Work

POMDP is a mathematical framework for modeling dynamic systems with imperfect observations, which is an extension of the Markov Decision Process [13]. In the context of self-driving decision problems, POMDP is commonly employed to capture the incomplete observability and uncertainty in the AV's surrounding environment. Previous studies have explored various sources of uncertainty in their POMDP models, categorized into offline and online methods based on their solution approaches [14]. For instance, [15] regarded pedestrians' target position as an unobservable variable in the POMDP model to capture the decision-making and planning behavior of autonomous vehicles navigating among many pedestrians. They employed an online planning method to solve this model. Similarly, an AV-pedestrian interaction model was proposed in [16] to address complex decision-making challenges arising from the uncertain crossing intention of pedestrians in urban environments by leveraging the POMDP framework. In contrast to these approaches, our study treats pedestrian reasoning levels and rationality degrees as unobservable information within the POMDP framework, similar to [17], [18] focusing on vehicle-vehicle interactions. However, our work innovatively introduces a dynamic mechanism to update rationality levels based on observed behavior and prior knowledge.

Game theory [19] serves as a valuable tool for modeling and analyzing conflicts among individuals, initially used in economics and now extended to vehicle-vehicle or vehicle-pedestrian interactions in the context of AVs. A Zebra Crossing Game was introduced to explore cyclist-vehicle interaction in Norway, demonstrating consistency between real crossing behavior and the solution derived from game theory [20]. Similarly, the 'sequential chicken' model was proposed to simulate space competition between vehicles and pedestrians at an unsigned intersection [21]. This model was further extended in [22] by employing empirical data and the Gaussian Process to fit the model's parameters. A recent study developed a Stackelberg game model based on the belief that players usually make sequential decisions in road conflicts rather than simultaneous responses [23]. Similar applications of the Stackelberg game model for simulating the interaction between vehicles and pedestrians are observed in [24] and [25].

The approaches mentioned earlier are all based on the conventional game theory model with players' complete rationality assumption. However, human behavior does not always conform to the predictions of the Nash equilibrium [26] in real-world situations due to bounded rationality and cognitive limitation. To relax the assumption of complete rationality, Chen et al. [27] combined evolutionary game theory with cumulative prospect theory to formulate an interactive decision model at uncontrolled mid-block crosswalks. This method can simulate different behaviors within a pedestrian group but requires numerous parameters for model fitting.

In contrast, behavioral game theory provides a more accurate predictor of human behavior in real-world scenarios. It outperforms conventional models in forecasting interaction outcomes [28]. The researchers argued that Nash equilibrium, normally with complete information may not sufficiently reflect the unpredictable actions of pedestrians at crosswalks. To simulate the joint behavior of pedestrians and vehicles, they proposed a game theoretical framework, namely logit quantal response equilibrium [29], [5] with incomplete information, replacing Nash equilibrium. Moreover, level-k reasoning softened the perfect rationality assumption of Nash equilibrium by assuming that agents have different levels of reasoning [30], applied in diverse vehicle interaction scenarios such as roundabouts [31], lane changes [32], and intersections [33]. However, if the opponent's cognitive hierarchy is not at level-(k − 1), the level-k model may not perform well in predicting its behavior. Another approach, the 'cognitive hierarchy' framework, allowed interaction with opponents of varying cognitive levels, not just one level below [34].

Despite previous efforts using quantal level-k game theory for vehicle-vehicle interactions [17], [18], fixed rational levels for humans were a limitation. The quantal cognitive hierarchy model has been demonstrated better performance in predicting human behaviors [35]. However, its application in the field of autonomous driving remains unexplored. Therefore, our work adopts a DB-QCH model to model the AV and the pedestrian, providing a more accurate description of AV-pedestrian interactions in urban areas.

---

## III. Problem Formulation

This work focuses on addressing the challenge posed by conflicts arising at the unsignalized intersection, where both the pedestrian and the AV intend to cross simultaneously. Specifically, it aims to develop continuous decision-making strategies for AVs navigating safely and efficiently through such a scenario. As the AV lacks knowledge about the opponent's intelligence level and rationality in a dynamic and interactive environment, we model the interaction between the AV and the pedestrian using a POMDP framework. The model is defined by the tuple:

$$\langle N, S, A, T, O, J, B \rangle$$

- **N = {0, 1}:** Represents the two players, where 0 denotes the AV and 1 denotes the pedestrian.
- **S:** A finite set of states, where $s_t \in S$ signifies the state of the environment at discrete time step $t$.
- **A = {A⁰, A¹}:** Defines the action space, with $A^0$ representing the AV's actions and $A^1$ denoting the pedestrian's actions.
- **T:** The state transition dynamics, expressed as $s_{t+1} = T(s_t, a^0_t, a^1_t)$ for an action pair $(a^0, a^1) \in A$. This function describes how the environment transitions from one state to another based on the actions of both players.
- **O = {O⁰, O¹}:** Represents the partially observable state. We assume that each agent's action can be observed, along with certain physical information (e.g., speed, acceleration, distance), while implicit information (e.g., reasoning level, rationality degree) remain unobservable.
- **J = {J⁰, J¹}:** The utility function for each agent. The utility $J^i_t = J^i(s_t, a^0_t, a^1_t)$, $i \in N$, depends on both the agent's action and the opponent's action.
- **B = {B⁰, B¹}:** The belief in the opponent's intelligence level and rationality, with $b^0_t \in B^0$ and $b^1_t \in B^1$.

For the AV model, the goal is to determine a sequence of optimal actions. The optimization problem can thus be formulated as follows:

$$\maximize_{\pi} \quad \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t J^0_t(s_t, a^0_t, a^1_t, b^0_t) \mid a^0_t \sim \pi\right]$$

$$\text{subject to} \quad s_{t+1} = T(s_t, a^0_t, a^1_t),$$
$$b^0_{t+1} = \rho(b^0_t, o^0_t),$$
$$a^0_t \in A^0,\ a^1_t \in A^1,$$
$$o^0_t \in O^0,\ b^0_t \in B^0 \tag{1}$$

where $\gamma$ represents the discount factor within the range of $(0, 1]$, while $\rho$ denotes the belief update function.

---

## IV. Methodology

This section provides a detailed description of the approaches we use to model the interaction between AVs and pedestrians at the unsignalized intersection and solve the above problem.

### A. Action Space Generation

For our AV model, the dynamic decision-making process aims to produce a sequence of expected accelerations. However, in actual scenarios, the AV's acceleration range exists in a continuous space, posing challenges for methods like MCTS, which typically excel in discrete action spaces. To address this, we employ a pre-trained neural network model to guide MCTS through the continuous action space of AVs.

We have opted for the long short-term memory (LSTM) network as our neural network model. LSTM, a subtype of recurrent neural network (RNN), is good at processing and predicting time series data, adeptly capturing temporal dependencies [36]. Unlike traditional RNNs, LSTM overcomes long-term dependency issues through its gate mechanisms (including input gate, forget gate, and output gate), effectively retaining and leveraging long-term information [37]. This gives LSTM a significant advantage in handling complex time series data in autonomous driving scenarios.

Acceleration prediction in autonomous driving presents a highly temporal problem, as a vehicle's acceleration depends not only on its current state but also on past states and actions. Traditional RNNs often encounter problems like gradient vanishing or exploding when dealing with long-term dependencies [38], making it difficult to effectively capture long-term dependency information. LSTM, with its unique gating mechanism, can maintain and transmit key information across lengthy time series, avoiding these shortcomings of traditional RNNs [39].

Given the temporal dynamics and complexity of acceleration prediction tasks, we choose LSTM as our preferred model for anticipating AV acceleration. Its ability to use historical data enhances prediction accuracy and stability, providing robust support for our decision-making system.

Training data for the LSTM model is sourced from our prior vehicle-pedestrian interaction experiments [40], conducted using virtual reality (VR) technology. This experiment yielded dynamic interaction data, including the absolute positions of pedestrians and vehicles, vehicle speeds, and driver inputs like steering, throttle, and brakes. Through data processing, we extracted relevant variables such as vehicle speed, acceleration, relative distances, time-to-arrivals, pedestrian speeds, and vehicle yielding status at each time step for every scenario.

This data underwent training in the LSTM model, which, post-training, can ingest state information at each time step and output the corresponding anticipated acceleration. This acceleration is treated as the mean of a Gaussian distribution, from which N accelerations are sampled, yielding N + 1 possible accelerations. Subsequently, MCTS is employed to explore these N + 1 actions, leveraging this neural network model's output as an initial guide to enhance MCTS's efficiency in navigating the continuous action space. Through this integration of the neural network and MCTS methods, we can improve the decision-making ability of autonomous vehicles in complex dynamic interactive environments.

In contrast, within the pedestrian model, the action space pertains to the pedestrian's speed. Since pedestrian speed can change rapidly, using a neural network model with Gaussian distribution sampling for action spaces, as used in the AV model, is less effective. Instead, we adopt a discrete action selection space to simplify our model. Starting from 0 m/s, we discretize the speed at 0.1 m/s intervals. Unlike the AV model, where elements in the action space dynamically change, the pedestrian's action space remains fixed at each time step.

### B. Dynamic-Belief-Induced Quantal Cognitive Hierarchy Model

#### 1) Quantal Cognitive Hierarchy Model

The Quantal Cognitive Hierarchy model is a behavioral game theory model used to describe the behavior of bounded rational individuals in games. It integrates the quantal response (QR) model into the traditional cognitive hierarchy (CH) model.

In the CH model, agents are characterized by different cognitive levels, each associated with varying degrees of reasoning abilities and consideration of others' behavior. Higher levels indicate greater reasoning capabilities and more consideration of opponents' actions. At each level, agents simulate their opponents' behavior under the assumption that opponents operate at lower levels. Each agent's cognitive level is denoted by $k$ (where $k = 0, 1, 2, \ldots$). Level-0 agents are regarded as non-strategic, generating their strategies independently and without considering opponents' behavior, often through uniform random selection or simple heuristic methods. Conversely, strategic agents at level-$k$ (where $k > 0$) engage in a more sophisticated decision-making process. They assume their opponents operate at level-$j$, where $j < k$, and respond accordingly with optimal strategies.

The QR model introduces the concept of bounded rationality, where agents do not always choose the optimal strategy but select strategies with certain probabilities based on expected payoffs when making decisions. In this model, bounded rationality is represented by the parameter $\lambda$ (where $\lambda \in [0, \infty)$), which measures the degree of rationality. A higher $\lambda$ indicates more rational behavior, while a lower value reflects greater randomness in decision-making. The probability $P(a_i)$ that an agent $i$ chooses a particular strategy $a_i$ given the opponent's action is described by the quantal response function:

$$P(a_i) = \frac{e^{\lambda Q(a_i, a_{-i})}}{\sum_{a'_i \in A} e^{\lambda Q(a'_i, a_{-i})}} \tag{2}$$

where $Q_i(a_i, a_{-i})$ is the expected payoff for agent $i$ when choosing strategy $a$.

Equation 2 shows that the probability of selecting a strategy increases with its expected payoff, meaning individuals are more likely to select a strategy with a higher expected payoff but may also opt for those with lower returns. As $\lambda$ approaches infinity, the model approximates perfect rationality, where the highest payoff strategy is always chosen. Conversely, when $\lambda$ is close to zero, the choice of strategy becomes completely random.

By combining ideas from the CH model and QR model, the QCH model offers insights into how individuals probabilistically select strategies at different cognitive levels, thereby enhancing our understanding of bounded rational behavior. In our study, we adopt the QCH model to represent the decision-making process for both the AV and the pedestrian. This model captures the varying levels of intelligence $k$ and rationality $\lambda$ of each opponent, which are unobservable to each other.

At each level-$k$, the AV evaluates its potential actions by calculating the expected payoff $Q$ for each action given the current state $s_t$. This evaluation also considers the pedestrian's policy from the preceding level-$(k-1)$. The AV then makes a quantal best response to the pedestrian's level-$(k-1)$ policy. The policy at each level for the AV and its opponent is developed through a sequential and iterative process, starting from level-0 to higher levels. In our study, we assume that the level-0 agent lacks understanding of pedestrian intentions or higher-level policies, and instead treats pedestrians as stationary obstacles to compute its actions. In contrast, the level-$k$ (where $k > 1$) agent regards its opponents as level-$(k-1)$ agents. Specifically, the quantal response function is used to compute the policy:

$$\pi^{i,k,\lambda_i}(a^i_j) = \frac{e^{\lambda_i Q_k(s_t, a^i_j, \pi^{-i,k-1,\lambda_{-i}})}}{\sum_{a' \in A^i} e^{\lambda_i Q_k(s_t, a', \pi^{-i,k-1,\lambda_{-i}})}} \tag{3}$$

After computing strategies for all levels, we can derive the optimal strategy using initial beliefs. Finally, we can determine the optimal action for the AV, selecting the action associated with the highest mixed strategy value.

---

**Algorithm 1: QCH Model Iterative Reasoning to Compute Optimal Action**

**Input:** $N$: Player set, $A$: Possible action set, $s_t$: Current state, $\pi^{i,0}$: The level-0 policy for agent $i$, $K$: Maximum cognitive level, $b_k$: Belief about the opponent's level, $\lambda$: Rationality degree

**Output:** Optimal action

```
Initialize agent_policy ← []
Initialize mix_policy ← []
Append π^{i,0} to agent_policy

for k = 1 to K do
    for each player i ∈ N do
        for each action a^i_j ∈ A^i do
            Compute the payoff Q_{i,k}(s_t, a^i_j, π^{-i,k-1,λ_{-i}})
            Compute the policy π^{i,k,λ_i}(a^i_j)
            Append π^{i,k,λ_i} to agent_policy

for k = 1 to K do
    mix_policy ← mix_policy + b_k[k-1] · agent_policy[k]

optimal_index ← argmax(mix_policy)
optimal_action ← A[optimal_index]
return optimal_action
```

---

#### 2) Dynamic Belief Update

For AVs, the opponent's reasoning level-$k$ and rationality degree $\lambda$ are not directly observable. Pedestrian behavior is dynamic and may constantly change. If AVs always use fixed values of pedestrian's cognitive states for best response calculation during interactions with pedestrians, they will be unable to effectively identify and adapt to changes in pedestrian behavior. The Bayesian approach allows AVs to continuously learn and update their beliefs about pedestrians' reasoning level-$k$ and rationality $\lambda$ during interactions.

At time step $t = 0$, the agent $i$ establishes an initial belief $b_{k,0}$ about the pedestrian's reasoning level, according to the initial environmental state and our prior experimental data on human-vehicle interactions [40]. Throughout the game reasoning process, its QCH model iteratively predicts the expected utility of the opponent's potential actions across each reasoning level $k$ for the next state $s_{t+1}$, alongside computing the associated probability $P(s_{t+1}, a^{-i}_{t+1}|k)$. Upon observing the opponent's latest action $a^{-i}_{t+1}$ at time step $t+1$, the agent model updates its belief $b_{k,t+1}$ concerning its opponent's reasoning level using the Bayesian equation:

$$P(k \mid s_{t+1}, a^{-i}_{t+1}) = \frac{P(s_{t+1}, a^{-i}_{t+1} \mid k)\, b_{k,t}(k)}{\sum_{k' \in \Theta} P(s_{t+1}, a^{-i}_{t+1} \mid k')\, b_{k,t}(k')} \tag{4}$$

where $P(k|s_{t+1}, a^{-i}_{t+1}) \in b_{k,t+1}$, and $\Theta$ represents all possible values for the reasoning level.

Our model differs from others by dynamically updating the belief about the opponent's rationality degree $\lambda$, rather than relying on constant values. Initially, agent $i$ has prior knowledge regarding the distribution of $\lambda$, denoted as $f_t(\lambda)$. Since $\lambda \in [0, \infty)$, it is treated as a continuous variable. Therefore, we use a Bayesian updating method suitable for continuous variables [41]:

$$f_{t+1}(\lambda \mid a^j_t) = \frac{P(a^j_t \mid \lambda)\, f_t(\lambda)}{\int_0^\infty P(a^j_t \mid \lambda')\, f_t(\lambda')\, d\lambda'} \tag{5}$$

Considering the varying reasoning level of agent $j$, this can be extended as:

$$f_{t+1}(\lambda \mid a^j_t, k) = \frac{P(a^j_t \mid k, \lambda)\, f_t(\lambda)}{\int_0^\infty P(a^j_t \mid k, \lambda')\, f_t(\lambda')\, d\lambda'} \tag{6}$$

The conjugate prior distribution proves highly effective in addressing the challenge of computing Equation 6 after multiple iterations [42]. Our work considers the following family of distributions [43]:

$$f(\lambda;\, Q, n_0, n_1, \ldots, n_K) = \frac{e^{\lambda Q} / \prod_{k=0}^{K}\left(\sum_{l=1}^{m} e^{\lambda Q_{a_l,k}}\right)^{n_k}}{\int_0^\infty e^{\lambda' Q} / \prod_{k=0}^{K}\left(\sum_{l=1}^{m} e^{\lambda' Q_{a_l,k}}\right)^{n_k} d\lambda'} \tag{7}$$

where $n_k \in \mathbb{N}$, $\forall k = 0, 1, 2, \ldots, K$, representing the number of occurrences agent $j$'s reasoning level corresponds to $k$.

**Theorem 1:** Given the prior distribution $f_t(\lambda; Q, n_0, n_1, \ldots, n_K)$, upon observing the action $a^j$ taken by the opponent at time step $t+1$, agent $i$ can update the belief as $f_{t+1}(\lambda; Q + Q_{a^j,k},\, n_0, n_1, \ldots, n_k + 1, \ldots, n_K)$.

When the distribution of the continuous variable $\lambda$ at the next time step $s_{t+1}$ is obtained, the expectation of rationality degree can be calculated using:

$$\mathbb{E}(\lambda) = \int_0^\infty \lambda\, f(\lambda;\, Q + Q_{a^j,k},\, n_0, n_1, \ldots, n_k + 1, \ldots, n_K)\, d\lambda \tag{8}$$

---

**Algorithm 2: Belief Update**

**Input:** $s_t$: Current state, $a^j$: Observed opponent's action, $Q_{a^j,k}$: Opponent's action $a^j$'s expected utility for each level $k$, $b_{k,t}$: Prior belief about reasoning level, $P(s_{t+1}, a^j|k)$: Probability for $a^j$ at each level $k$, $f(\lambda; Q, n_0, n_1, \ldots, n_K)$: Prior distribution about rationality

**Output:** Updated belief $b_{k,t+1}$, $b_{\lambda,t+1}$

```
for k = 0 to K-1 do
    P(k | s_{t+1}, a^{j}_{t+1}) = P(s_{t+1}, a^{j}_{t+1}|k) · b_{k,t}(k) /
                                   Σ_{k'∈Θ} P(s_{t+1}, a^{j}_{t+1}|k') · b_{k,t}(k')
    b_{k,t+1}(k) ← P(k | s_{t+1}, a^{j}_{t+1})

k ← argmax(b_{k,t+1})
E(λ) ← ∫₀^∞ λ · f(λ; Q + Q_{a^j,k}, n_0, ..., n_k+1, ..., n_K) dλ
b_{λ,t+1} ← E(λ)
return b_{k,t+1}, b_{λ,t+1}
```

---

Through dynamic belief updates, AVs can more accurately predict pedestrian behavior and adjust their strategies based on the latest beliefs. This dynamic update mechanism enables AVs to better adapt to changing environments, thereby behaving more intelligently and human-like.

### C. MCTS

Monte Carlo Tree Search is a heuristic search algorithm to predict future outcomes and optimizes decision-making by simulations. It includes four main steps: selection, expansion, simulation, and backpropagation [44]. In our study, MCTS is used to compute the anticipated payoff of each possible action for both the AV and pedestrian models at each level for each moment.

Specifically, the decision tree is initialized at every time step from the current state $s_t$. In the selection stage, we use the Upper Confidence Bound applied to Trees (UCT) formula to calculate the UCT value for each potential action in the action space and select the one with the highest UCT value for the next expansion. The UCT equation is shown below [45]:

$$\text{UCT}(s, a) = \bar{Q}_a + C\sqrt{\frac{\ln N_s}{N_a}} \tag{9}$$

where $\bar{Q}_a$ is the average utility of action $a$ at state $s_t$, $N_s$ is the total number of visits to the state $s$, $N_a$ is the number of times action $a$ was chosen, and $C$ is a constant that balances exploration and exploitation.

During the expansion step, new decision nodes are generated, corresponding to different actions that the AV and the pedestrian may take in the current state. The simulation step starts from the newly expanded node, where the actions of the agent and its opponent are randomly simulated in turn until reaching the terminal state or the maximum search depth. In this stage, a random strategy is used to select actions and simulate the opponent's strategies, estimating the potential value of the node. Notably, when calculating the optimal strategy for the agent at level-$k$, its opponent follows the policy of level-$(k-1)$; for level-0 agents, their opponents are regarded as static obstacles. Upon simulation completion, the rewards obtained are backpropagated to the root node, updating the node statistics information. After multiple iterations, the average cumulative utility for each action will be utilized to calculate the quantal response policy.

Through this process, MCTS evaluates the potential effects of different actions through extensive simulations without relying on specific domain knowledge, enabling AV to make efficient and safe decisions in complex and uncertain environments.

---

## V. Experiments and Results

### A. Experiment Setup

We conducted a series of simulation experiments for verification to evaluate the effectiveness of the AV model and decision-making algorithm we developed. We built a simulation scenario where the AV interacts with a pedestrian crossing the road. The AV is 5 m long and 2 m wide, driving on a single-lane road that is 3.65 m wide. The pedestrian's goal is to cross the road to reach the opposite side.

We previously conducted experiments on real human drivers and pedestrians interacting in a VR environment [40]. During these experiments, the driver continuously drove along the road at a random speed and interacted with the pedestrian crossing the road. We randomly selected 100 scenarios from these experimental data for this simulation experiment. The initial conditions of each scenario were input into our model for simulation, and each scenario was simulated 100 times to ensure the reliability and statistical significance of the results.

The initial data included the initial positions of the vehicle and the pedestrian, the vehicle's initial velocity and acceleration, and the pedestrian's velocity. In our models, we assume that the AV and pedestrian follow a straight line with only longitudinal movements. The acceleration range of the AV is set to [-5, 5] m/s², and the speed range of the pedestrian is [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] m/s. We assume that the maximum reasoning ability of the agent is $k_{\text{max}} = 2$, with an initial rationality degree value of $\mathbb{E}(\lambda) = 10$. Each time step is 0.8 s. The level-0 policy is non-strategic, which believes that the agent calculates its optimal strategy based on the assumption that its opponents are static obstacles.

### B. Results

This section will verify and evaluate our proposed model's performance through qualitative and quantitative analysis.

#### 1) Qualitative Analysis

Three simulation examples will be given to clearly and intuitively demonstrate how our AV model operates under different scenarios.

**Case 1:** This case illustrates an AV yielding to a pedestrian, randomly selected from a set of 100 scenarios. Initially, the AV is 46.309 m away from the pedestrian, traveling at a speed of 9.348 m/s with an acceleration of -0.11 m/s² towards the pedestrian. The pedestrian is located 2.564 m laterally from the AV on the road.

Under these initial conditions, the AV calculates a higher probability that the pedestrian is a level-0 agent, indicating that the pedestrian is more likely to cross the road. Therefore, the AV decides to decelerate to avoid collision. When the AV observes the pedestrian stepping into the lane at a speed of 1.2 m/s, it executes a more pronounced deceleration. At the same time, the AV updates its assessment of the pedestrian's rationality based on their actions, concluding that the pedestrian remains in a rational state.

As the pedestrian continues to cross and approach its destination, the AV gradually reduces its deceleration and eventually transitions back to acceleration. Notably, the AV does not come to a complete stop but maintains a reduced speed while the pedestrian crosses. This entire process demonstrates the AV's ability to yield to pedestrians by making real-time decisions, ensuring both safety and efficiency.

**Case 2:** This case illustrates a scenario where a pedestrian yields to an AV. Compared to Case 1, the initial longitudinal distance between the AV and the pedestrian is updated to 34 m with other conditions remaining the same.

At the first time step, the AV is 34 m away, moving at a speed of 9.348 m/s, and an acceleration of -0.11 m/s². Unlike Case 1, the AV determines that the pedestrian poses a lower probability of crossing and designates the pedestrian as a level-1 agent, signifying that the pedestrian will not recklessly enter the lane and yield to AV. As a result, the AV opts to maintain a slight deceleration in case the pedestrian suddenly changes their mind and steps into the lane. However, after observing that the pedestrian does not appear in the lane, the AV confirms that the pedestrian will indeed not cross and perceives its behavior as rational. At this point, the AV sustains a slight deceleration to ensure the pedestrian's safety and maintain smooth operation without significant speed adjustments.

**Case 3:** This case primarily aims to validate the AV's ability to identify irrational pedestrian behavior. The initial conditions are the same as Case 2, with the only difference being the replacement of the pedestrian model with a custom-designed action for the pedestrian. Specifically, we program the pedestrian to cross the road under these conditions at a speed of 1.2 m/s.

The AV, similar to Case 2, initially assumes a low likelihood of pedestrians crossing the road and thus only slows down slightly. However, when the pedestrian begins moving, the AV updates its estimation of the pedestrian being a level-0 agent based on their actions. At the same time, a marked drop in the pedestrian's rationality value indicates that the AV deems it irrational for the pedestrian to cross the road under the current circumstance. According to these judgments, the AV decelerates more sharply than in Case 2, resulting in a rapid decrease in vehicle speed. This indicates that the proposed AV model can effectively update its understanding of pedestrian rationality based on real-time behaviors and appropriately adjust its acceleration to prevent potential collisions.

#### 2) Quantitative Analysis

Following the analysis method in [48], we conduct the quantitative evaluation from three aspects: safety, efficiency, and smoothness. We input 100 scenarios into our model for simulation, with each scenario being simulated 100 times.

**Table I: Statistic Results of Our Proposed Approach Compared with VR Experiments**

| Metric | Category | VR Experiment | Ours |
|--------|----------|---------------|------|
| Safety | Collision rate | — | 0.15% |
| Efficiency | Yielding rate | 51% | 75.03% |
| Efficiency | Average vehicle speed (m/s) | 9.468 | 9.600 |
| Smoothness | Average vehicle jerk | 0.897 | 0.843 |
| Smoothness | Average maximum absolute acceleration/deceleration (m/s²) | 2.228 | 1.867 |

To evaluate driving efficiency, the vehicle yielding rate and average vehicle speed are considered. The yielding rate observed in the simulation (75.03%) is higher than in the VR experiment (51%). This suggests that the algorithm is more cautious, which may slightly affect the average speed. However, the average speed in the simulation (9.600 m/s) is slightly faster compared to the experiment (9.468 m/s), indicating that our approach can maintain efficiency even with a greater yielding rate. Two parameters, jerk and maximum absolute acceleration/deceleration, are used to evaluate smoothness. The lower average jerk value and average maximum absolute acceleration/deceleration value in Table I indicate that our proposed method can achieve smoother and more comfortable driving behavior.

In summary, the quantitative analysis shows that the proposed AV decision-making algorithm performs well in safety, efficiency, and smoothness. The similar average vehicle speed values between VR experimental data and simulation data indicate that our algorithm closely mimics real-world driving behavior. Additionally, the lower jerk values and maximum absolute acceleration/deceleration values in our simulations suggest that our method achieves smoother driving compared to the experimental data.

---

## VI. Conclusion

This paper proposes an innovative framework to address the decision-making challenges AVs face when interacting with pedestrians at the unsignalized intersection. First, we integrate the POMDP with behavioral game theory to model these interactions, capturing the uncertainty and dynamism of pedestrian behavior. Second, both the AV and pedestrian are modeled as DB-QCH models, accounting for human reasoning limitations and bounded rationality, thus enabling more realistic interaction simulations compared to traditional game theory approaches. Moreover, the dynamic updating mechanism for the opponent's rationality degree is introduced, which allows the AV to adjust its strategies based on real-time observations. Finally, a trained neural network is developed to guide MCTS within the AV's continuous action space, improving decision-making efficiency and effectiveness.

Simulation results demonstrate that our method excels in safety, efficiency, and smoothness, closely resembling real-world driving behavior. Although our model performs well, our current research is limited to a simple scenario of a single AV and a single pedestrian interaction. In the future, we will expand our scope to include the interaction between a single AV and multiple pedestrians, allowing the proposed AV decision-making algorithm to handle more complex scenarios.

---

## References

[1] A. Mehta et al., "Securing the future: A comprehensive review of security challenges and solutions in advanced driver assistance systems," *IEEE Access*, 2023.

[2] M. T. Rahman, K. Dey, S. Das, and M. Sherfinski, "Sharing the road with autonomous vehicles: A qualitative analysis of the perceptions of pedestrians and bicyclists," *Transportation Research Part F*, vol. 78, pp. 433–445, 2021.

[3] X. Li et al., "Sharing roads with automated vehicles: A questionnaire investigation from drivers', cyclists' and pedestrians' perspectives," *Accident Analysis & Prevention*, vol. 188, p. 107093, 2023.

[4] K. Yang et al., "Uncertainties in onboard algorithms for autonomous vehicles: Challenges, mitigation, and perspectives," *IEEE Transactions on Intelligent Transportation Systems*, vol. 24, no. 9, pp. 8963–8987, 2023.

[5] H. Li et al., "The role of yielding cameras in pedestrian-vehicle interactions at un-signalized crosswalks: An application of game theoretical model," *Transportation Research Part F*, vol. 92, pp. 27–43, 2023.

[6] K. Tian et al., "Deceleration parameters as implicit communication signals for pedestrians' crossing decisions and estimations of automated vehicle behaviour," *Accident Analysis & Prevention*, vol. 190, p. 107173, 2023.

[7] K. Tian et al., "Deconstructing pedestrian crossing decisions in interactions with continuous traffic: An anthropomorphic model," *IEEE Transactions on Intelligent Transportation Systems*, 2023.

[8] S. Rezwana and N. Lownes, "Interactions and behaviors of pedestrians with autonomous vehicles: A synthesis," *Future Transportation*, vol. 4, no. 3, pp. 722–745, 2024.

[9] P. Pouya and A. M. Madni, "Expandable-partially observable Markov decision-process framework for modeling and analysis of autonomous vehicle behavior," *IEEE Systems Journal*, vol. 15, no. 3, pp. 3714–3725, 2020.

[10] D. Zhu et al., "A two-stage safety evaluation model for the red light running behaviour of pedestrians using the game theory," *Safety Science*, vol. 147, p. 105600, 2022.

[11] L. Peters et al., "Contingency games for multi-agent interaction," *IEEE Robotics and Automation Letters*, 2024.

[12] J. R. Wright and K. Leyton-Brown, "Level-0 meta-models for predicting human behavior in games," in *Proc. 15th ACM Conference on Economics and Computation*, 2014, pp. 857–874.

[13] H. Kurniawati, "Partially observable Markov decision processes (POMDPs) and robotics," *arXiv preprint arXiv:2107.07599*, 2021.

[14] L. Burks et al., "HARPS: An online POMDP framework for human-assisted robotic planning and sensing," *IEEE Transactions on Robotics*, vol. 39, no. 4, pp. 3024–3042, 2023.

[15] H. Bai et al., "Intention-aware online POMDP planning for autonomous driving in a crowd," in *Proc. IEEE ICRA*, 2015, pp. 454–460.

[16] Y.-C. Hsu et al., "A POMDP treatment of vehicle-pedestrian interaction: Implicit coordination via uncertainty-aware planning," in *Proc. IEEE/RSJ IROS*, 2020, pp. 1984–1991.

[17] R. Tian et al., "Anytime game-theoretic planning with active reasoning about humans' latent states for human-centered robots," in *Proc. IEEE ICRA*, 2021, pp. 4509–4515.

[18] S. Dai, S. Bae, and D. Isele, "Game theoretic decision making by actively learning human intentions applied on autonomous driving," *arXiv preprint arXiv:2301.09178*, 2023.

[19] J. Von Neumann and O. Morgenstern, *Theory of Games and Economic Behavior*, 2nd ed., 1947.

[20] T. Bjørnskau, "The zebra crossing game–using game theory to explain a discrepancy between road user behaviour and traffic rules," *Safety Science*, vol. 92, pp. 298–301, 2017.

[21] F. Camara et al., "When should the chicken cross the road?: Game theory for autonomous vehicle-human interactions," 2018.

[22] F. Camara et al., "Empirical game theory of pedestrian interaction for autonomous vehicles," in *Proc. Measuring Behavior 2018*, pp. 238–244.

[23] R. E. Amini, A. Dhamaniya, and C. Antoniou, "Towards a game theoretic approach to model pedestrian road crossings," *Transportation Research Procedia*, vol. 52, pp. 692–699, 2021.

[24] X. Sun et al., "A study on pedestrian–vehicle conflict at unsignalized crosswalks based on game theory," *Sustainability*, vol. 14, no. 13, p. 7652, 2022.

[25] Y. Chen et al., "Interaction-aware decision-making for autonomous vehicles," *IEEE Transactions on Transportation Electrification*, vol. 9, no. 3, pp. 4704–4715, 2023.

[26] C. M. Harris, "Autonomous vehicle decision-making: Should we be bio-inspired?" in *TAROS 2017*, Springer, 2017, pp. 315–324.

[27] P. Chen, C. Wu, and S. Zhu, "Interaction between vehicles and pedestrians at uncontrolled mid-block crosswalks," *Safety Science*, vol. 82, pp. 68–76, 2016.

[28] A. H. Kalantari et al., "Driver-pedestrian interactions at unsignalized crossings are not in line with the Nash equilibrium," *IEEE Access*, 2023.

[29] Y. Zhang and J. D. Fricker, "Incorporating conflict risks in pedestrian-motorist interactions: A game theoretical approach," *Accident Analysis & Prevention*, vol. 159, p. 106254, 2021.

[30] L. Crosato et al., "Social interaction-aware dynamical models and decision-making for autonomous vehicles," *Advanced Intelligent Systems*, vol. 6, no. 3, p. 2300575, 2024.

[31] R. Tian et al., "Adaptive game-theoretic decision making for autonomous vehicle control at roundabouts," in *Proc. IEEE CDC*, 2018, pp. 321–326.

[32] S. Karimi, A. Karimi, and A. Vahidi, "Level-k reasoning, deep reinforcement learning, and monte carlo decision process for fast and safe automated lane change and speed management," *IEEE Transactions on Intelligent Vehicles*, vol. 8, no. 6, pp. 3556–3571, 2023.

[33] S. Fang et al., "Cooperative driving of connected autonomous vehicles in heterogeneous mixed traffic: A game theoretic approach," *IEEE Transactions on Intelligent Vehicles*, 2024.

[34] S. Li et al., "Decision making in dynamic and interactive environments based on cognitive hierarchy theory, Bayesian inference, and predictive control," in *Proc. IEEE CDC*, 2019, pp. 2181–2187.

[35] Y. Xu, S.-F. Cheng, and X. Chen, "Improving quantal cognitive hierarchy model through iterative population learning," *arXiv preprint arXiv:2302.06033*, 2023.

[36] F. M. Shiri et al., "A comprehensive overview and comparative analysis on deep learning models: CNN, RNN, LSTM, GRU," *arXiv preprint arXiv:2305.17473*, 2023.

[37] T. Qie et al., "A self-trajectory prediction approach for autonomous vehicles using distributed decouple LSTM," *IEEE Transactions on Industrial Informatics*, 2024.

[38] S. Das et al., "Recurrent neural networks (RNNs): Architectures, training tricks, and introduction to influential research," *Machine Learning for Brain Disorders*, pp. 117–138, 2023.

[39] W. Li and K. E. Law, "Deep learning models for time series forecasting: A review," *IEEE Access*, 2024.

[40] M. Dang et al., "Coupling intention and actions of vehicle–pedestrian interaction: A virtual reality experiment study," *Accident Analysis & Prevention*, vol. 203, p. 107639, 2024.

[41] A. Salmerón et al., "A review of inference algorithms for hybrid Bayesian networks," *Journal of Artificial Intelligence Research*, vol. 62, pp. 799–828, 2018.

[42] M. H. DeGroot, *Optimal Statistical Decisions*. John Wiley & Sons, 2005.

[43] Q. Guo and P. Gmytrasiewicz, "Modeling bounded rationality of agents during interactions," in *Workshops at the 25th AAAI Conference on Artificial Intelligence*, 2011.

[44] M. Świechowski et al., "Monte Carlo tree search: A review of recent modifications and applications," *Artificial Intelligence Review*, vol. 56, no. 3, pp. 2497–2562, 2023.

[45] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," in *European Conference on Machine Learning*, Springer, 2006, pp. 282–293.

[46] T. T. M. Tran, C. Parker, and M. Tomitsch, "A review of virtual reality studies on autonomous vehicle–pedestrian interaction," *IEEE Transactions on Human-Machine Systems*, vol. 51, no. 6, pp. 641–652, 2021.

[47] E. Ejichukwu et al., "Enhancing autonomous vehicle design and testing: A comprehensive review of AR and VR integration," *arXiv preprint arXiv:2404.19021*, 2024.

[48] D. Yang, K. Redmill, and Ü. Özgüner, "A multi-state social force based framework for vehicle-pedestrian interaction in uncontrolled pedestrian crossing scenarios," in *Proc. IEEE Intelligent Vehicles Symposium*, 2020, pp. 1807–1812.

---

## Author Biographies

**Meiting Dang** received the B.S. and M.S. degrees from Chang'an University, China, in 2017 and 2020, respectively. She is currently working toward the Ph.D. degree in the James Watt School of Engineering at the University of Glasgow, UK. Her research interests include decision-making and planning of autonomous vehicles, autonomous vehicle-pedestrian interaction modeling based on game theory, and machine learning.

**Dezong Zhao** (Senior Member, IEEE) received the B.S. and M.S. degrees from Shandong University, Jinan, China, in 2003 and 2006, respectively, and the Ph.D. degree from Tsinghua University, Beijing, China, in 2010. His research interests include connected and autonomous vehicles, low carbon vehicles, machine learning and nonlinear control theory and applications. Dr. Zhao is a Fellow of the Higher Education Academy and was the recipient of the Excellence 100 Campaign of Loughborough University.

**Yafei Wang** (Member, IEEE) received the B.S. degree from Jilin University, Changchun, China, in 2005, the M.S. degree from Shanghai Jiao Tong University, Shanghai, China, in 2008, and the Ph.D. degree from The University of Tokyo, Tokyo, Japan, in 2013. He is currently a Professor of automotive engineering with the School of Mechanical Engineering, Shanghai Jiao Tong University. His research interests include state estimation and control for connected and automated vehicles.

**Chongfeng Wei** (Member, IEEE) received his Ph.D. degree in mechanical engineering from the University of Birmingham in 2015. He is now a Senior Lecturer (Associate Professor) at University of Glasgow, UK. His current research interests include decision-making and control of intelligent vehicles, human-centric autonomous driving, cooperative automation, and dynamics and control of mechanical systems. He is also serving as an Associate Editor of IEEE TITS, IEEE TIV, IEEE TVT, and Frontier on Robotics and AI.
