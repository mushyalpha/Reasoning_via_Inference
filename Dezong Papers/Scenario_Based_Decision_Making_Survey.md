# Evaluating Scenario-based Decision-making for Interactive Autonomous Driving Using Rational Criteria: A Survey

**Zhen Tian†, Zhihao Lin†, Dezong Zhao⋆** *(Senior Member, IEEE)*, **Wenjing Zhao, David Flynn** *(Member, IEEE)*, **Shuja Ansari, and Chongfeng Wei** *(Member, IEEE)*

*James Watt School of Engineering, University of Glasgow, Glasgow, G12 8QQ, U.K.*  
⋆ Corresponding author: dezong.zhao@glasgow.ac.uk | † Equal contribution

*arXiv:2501.01886v1 [cs.RO] — 3 Jan 2025*

---

## Abstract

Autonomous vehicles (AVs) can significantly promote the advances in road transport mobility in terms of safety, reliability, and decarbonization. However, ensuring safety and efficiency in interactive driving within dynamic and diverse environments is still a primary barrier to large-scale AV adoption.

In recent years, deep reinforcement learning (DRL) has emerged as an advanced AI-based approach, enabling AVs to learn decision-making strategies adaptively from data and interactions. DRL strategies are better suited than traditional rule-based methods for handling complex, dynamic, and unpredictable driving environments due to their adaptivity. However, varying driving scenarios present distinct challenges, such as avoiding obstacles on highways and reaching specific exits at intersections, requiring different scenario-specific decision-making algorithms.

Many DRL algorithms have been proposed in interactive decision-making. However, a rationale review of these DRL algorithms across various scenarios is lacking. Therefore, a comprehensive evaluation is essential to assess these algorithms from multiple perspectives, including those of vehicle users and vehicle manufacturers.

This survey reviews the application of DRL algorithms in autonomous driving across typical scenarios, summarizing road features and recent advancements. The scenarios include **highways**, **on-ramp merging**, **roundabouts**, and **unsignalized intersections**. Furthermore, DRL-based algorithms are evaluated based on five rationale criteria: **driving safety, driving efficiency, training efficiency, unselfishness, and interpretability (DDTUI)**. Each criterion of DDTUI is specifically analyzed in relation to the reviewed algorithms. Finally, the challenges for future DRL-based decision-making algorithms are summarized.

**Index Terms:** Interactive autonomous driving, decision making, deep reinforcement learning, typical scenarios, rationale evaluation.

---

## I. Introduction

Autonomous vehicles (AVs) face significant challenges in making reliable decisions when interacting with human-driven vehicles (HDVs). This challenge is primarily due to the difficulty of accurately predicting the intentions of HDVs. Road traffic crashes cause significant fatalities and serious injuries, reflecting the global issue of millions of lives lost annually. Since 2021, over 900 Tesla crashes involving driver-assistance systems have been reported. Despite unresolved safety issues, the number of AVs is projected to surpass 50 million by 2024. These statistics underscore the critical need for improving safety in autonomous driving.

With a safe decision-making system, AVs have the potential to significantly decrease road crashes caused by human errors such as fatigue, distraction, and delayed reactions. Moreover, AVs are capable of making optimal decisions faster than human drivers, thereby enhancing traffic efficiency.

There are several typical driving scenarios, such as highways, roundabouts, on-ramping merging, and unsignalized intersections, each characterized by distinct road features and scenario-specific requirements. Operational decision support for AV driving includes perception, planning, and control modules:

- The **perception module** consists of onboard sensors that continuously perceive the surrounding environment, processed through algorithms such as YOLO methods.
- The **planning module** handles driving tasks based on scenario recognition. The motion planner generates discrete decisions and converts them into feasible trajectories.
- The **control module** receives feasible trajectories and generates control commands sent to the vehicle's actuators (steering wheel and pedals).

### Approaches to Interactive Driving

**Model-based approaches** include four main types:

1. **Trajectory prediction** of HDVs within a fixed time window — limited by the window potentially being shorter than lane-changing maneuvers.
2. **Robust control methods** (e.g., min-max model predictive control) — make excessively cautious decisions based on worst-case scenario assumptions, not suitable for most real traffic environments.
3. **Game theory** (cooperative and non-cooperative games) — relies on equilibrium models that fail to capture real-world driving complexities and uncertainties.
4. **Collision-avoidance and Voronoi diagram-based methods** — unable to safely respond to movable objects.

Real-world data shows that 79% of ADS-related accidents involve HDVs hitting AVs, and 21% involve AVs hitting HDVs, underscoring that collision-free interactions remain unresolved.

**Simple guidance methods**, such as the Artificial Potential Field (APF), guide the AV to a target lane using attractive and repulsive force fields. However, APF assumes uniform risk around the vehicle and is difficult to generalize across scenarios without prior environmental knowledge.

**Learning-based methods** — particularly Deep Reinforcement Learning (DRL) — enable AVs to learn and adapt to complex driving scenarios through iterative interactions and feedback. DRL combines deep neural networks with reinforcement learning to handle complex, real-time decision-making tasks by learning directly from perceptual inputs.

### Survey Scope

This survey reviews DRL-based algorithms for autonomous interactive driving, classified by scenario and evaluated for real-world adaptation across four typical scenarios: **highways, on-ramping merging, roundabouts, and unsignalized intersections**. Algorithms are evaluated using five key criteria (**DDTUI**):

1. Driving Safety
2. Driving Efficiency
3. Training Efficiency
4. Unselfishness
5. Interpretability

---

## II. Road Features and Driving Tasks

### A. Highways

Highways are fundamental components of road networks, designed to enable vehicle movement over long distances with minimal interruption. Their design focuses on safety, efficiency, and environmental impact. Notable examples include the U.S. Interstate Highway System and Germany's Autobahn.

**Driving task:** Balancing collision avoidance with surrounding HDVs while maintaining a consistently high speed. The AV must decide between cautious car-following (safer but less efficient) and lane-changing to maintain speed (more efficient but collision-prone).

### B. On-ramp Merging

Ramps enable smooth and safe transitions between different roadways, connecting surface streets with highways. Key design considerations include safety for vehicles accelerating/decelerating, traffic flow efficiency, and urban space constraints (e.g., cloverleaf interchanges, HOV lane ramps).

**Comparison with Highways:**

| Feature | Highway | On-ramp |
|--------|---------|---------|
| Function | High-speed, long-distance travel | Transition between road types |
| Design | Long straight stretches, multiple lanes | Curves and elevation changes |
| Speed | Constant high speeds | Acceleration/deceleration required |

**Driving task:** The AV must change into the main lane before the ramp ends, navigating around surrounding HDVs and a static obstruction in the ramp lane, while maintaining high driving speed.

### C. Roundabouts

Roundabouts improve traffic flow and reduce severe accidents via a central island and circular road layout. Examples include Folon's obelisk in Pietrasanta, Italy, and Place Charles de Gaulle in Paris.

**Driving task:** The AV must navigate to one of several exits (e.g., O1, O2, O3) from an entry port (e.g., EB4), choosing between inner and outer lanes. The inner lane offers efficiency (shorter path) but increases collision risk from rear vehicles.

### D. Unsignalized Intersections

Unsignalized intersections manage traffic flow from different directions without signal control, reducing congestion and enhancing safety. An example is the Diverging Diamond Interchange (DDI).

**Driving task:** The AV must navigate a multi-lane intersection accommodating traffic from all four directions, making turns safely while preventing bottlenecks.

---

## III. Rationale of the Evaluation Factors

Five key evaluation factors have been selected for assessing DRL-based decision-making in real-world autonomous driving:

### A. Driving Safety

Driving safety is a fundamental requirement for AVs. It is primarily evaluated based on the **frequency of collisions** with other vehicles. Collision avoidance relies on flexible reactions to hazardous areas, assessing relative speed, distance, and trajectory of surrounding objects. Some systems also use rule-based commands (e.g., stopping at spot-lines during interactions).

### B. Driving Efficiency

Driving efficiency refers to an AV's ability to **maintain a high average speed** while adapting to traffic conditions. Its implications extend to:

- **Road capacity:** Efficient driving minimizes delays and reduces congestion.
- **User experience:** Shorter travel time and smoother rides improve satisfaction.
- **Energy consumption:** Efficient driving reduces energy use.

### C. Training Efficiency

Training efficiency directly impacts the **time and resources** required to deploy a functional AV system. Benefits of improved training efficiency include:

- **Reduced training time:** Allows developers to focus on fine-tuning and testing.
- **Reduced device wear:** Minimizes computational load, reducing maintenance needs.

### D. Unselfishness

Unselfishness refers to an AV's ability to **consider and accommodate the intentions of surrounding HDVs**. An unselfish AV predicts others' intentions and adjusts its behavior to minimize disruptions, contributing to smoother, more harmonious traffic flow. This avoids both overly aggressive and excessively cautious behaviors.

### E. Algorithm Interpretability

Interpretability makes the "black box" of DRL more transparent. Approaches include:

- **Policy visualization** to showcase DRL behaviors
- **Surrogate models** for human-understandable explanations
- **Rule-based methods** (e.g., the FAST criteria: Fairness, Accountability, Sustainability, Transparency)
- **Algorithmic structure adaptation** via standardized benchmarks and neural network architecture changes
- **Human-grounded methods** assessing how easily people understand key computational sections

---

## IV. DRL-Based Decision-Making on Highways

### A. Single-factor Methods

- **Safety:** DDQN integrated with handcrafted and dynamically-learned safety modules [95]; PG method with hard constraints preventing proximity to track edges [101]; DDQN + NMPC for safe highway driving with interpretable constraints [97].
- **Efficiency:** DDPG for overtaking with a reward function incorporating race position [96].
- **Unselfishness:** Cooperative lane-changing via reinforcement learning [102].
- **Interpretability:** DRL combined with imitation learning using expert demonstrations [103].
- **Training efficiency:** Spatial attention module integrated into DQN [104].

### B. Dual-factor Methods

- **Safety + Interpretability:** IDM integrated with DDQN [106].
- **Safety + Efficiency:** Adapted DDQN reward function with TTC threshold and velocity reward [107]; Multi-objective approximate policy iteration (MO-API) [112].
- **Safety + Unselfishness:** Level-k game-based DQN [109].
- **Unselfishness + Training Efficiency:** Cooperative multi-goal credit function PG with MARL curriculum [110].
- **Efficiency + Unselfishness:** MARL for average velocity optimization [111]; velocity and lane-change penalty rewards [113].
- **Safety + Training Efficiency:** Rule-based constraints with multi-head attention [114].

### C. Three-factor Methods

- **Safety + Interpretability + Efficiency:** IDM + collision penalty + velocity difference reward [115].
- **Safety + Efficiency + Unselfishness:** Multi-reward DQN (speed, lane-change limit, overtaking rewards) [116].
- **Safety + Efficiency + Interpretability:** PD controller for car-following + collision/velocity rewards [117]; Risk Potential Field (RPF) visualization [118]; ACC with safe distance and interpretable formulations [119].
- **Safety + Efficiency + Training Efficiency:** LSTM-assisted DDQN [120].

### D. Four-factor Methods

- **Safety + Efficiency + Training Efficiency + Interpretability:** Potential-based reward shaping with safety rules [121]; attention-based safety planner with SVM [126].
- **Safety + Efficiency + Unselfishness + Training Efficiency:** Joint policy MARL with experience reuse [122]; DCG-enhanced cooperative MARL [123]; parameter-sharing MARL [124]; distributional DQN with multi-type input [127].
- **Safety + Efficiency + Unselfishness + Interpretability:** Emergency braking + distributional DRL [125]; rule-based constraints + velocity rewards [128].

### E. Five-factor Methods

All five DDTUI factors addressed via CNN-LSTM with spatiotemporal image representations [129].

---

### Table I: Evaluation of DRL-Based Decision Making in Highway Driving

| Reference | Safety | Efficiency | Training Efficiency | Unselfishness | Interpretability |
|-----------|--------|------------|---------------------|---------------|-----------------|
| [95] | Safety modules | — | — | — | — |
| [96] | — | Overtaking reward | — | — | — |
| [97] | NMPC constraints | — | — | — | — |
| [101] | Hard constraints | — | — | — | — |
| [102] | — | — | — | Local interactions | — |
| [103] | — | — | — | — | Imitation learning |
| [104] | — | — | Attention module | — | — |
| [106] | IDM integration | — | — | — | IDM integration |
| [107] | TTC threshold | Velocity reward | — | — | — |
| [109] | Crash penalty | — | — | Level-k game | — |
| [110] | — | — | MARL curriculum | Cooperative function | — |
| [111] | — | Average velocity | — | MARL | — |
| [112] | Collision monitoring | Velocity comparison | — | — | — |
| [113] | — | Velocity reward | — | Lane change penalty | — |
| [114] | Rule-based | — | Attention mechanism | — | — |
| [115] | IDM & collision | Velocity difference | — | — | IDM integration |
| [116] | Speed-limit reward | Overtaking reward | — | Lane-change limit | — |
| [117] | Lane change penalty | Velocity tracking | — | — | PD controller |
| [118] | Reward function | Reward function | — | — | Risk potential field |
| [119] | Adaptive cruise | High-speed reward | — | — | ACC formulations |
| [120] | Collision penalty | Velocity difference | LSTM-DDQN | — | — |
| [121] | Safety rules | Reward function | Reward shaping | — | Safety rules |
| [122] | Reward function | Reward function | MARL reuse | Joint policy | — |
| [123] | Reaction time | Lane-changing point | DCG efficiency | MARL | — |
| [124] | Collision penalties | Overtaking reward | Parameter sharing | MARL | — |
| [125] | Collision rewards | Velocity ratio | — | Lane change limit | Emergency braking |
| [126] | Safety layer | Velocity ratio | Attention mechanism | — | SVM boundaries |
| [127] | Collision rewards | Velocity ratio | Distributional DQN | MARL | — |
| [128] | Collision rewards | Velocity difference | — | Lane change penalty | Rule-based |
| [129] | Collision reduction | Speed increase | CNN-LSTM | Lane change limit | Representations |

*'—' indicates that the corresponding factor was not explicitly addressed.*

---

## V. DRL-Based Decision-Making in On-ramp Merging

### A. Single-factor Methods

- **Efficiency:** Q-learning with speed and queue-length reward balancing [130]; reduction of total travel time reward [131].
- **Safety:** Safety factor as negative reward for small relative distances [132]; rewards for safe distance and collision penalties [133].

### B. Dual-factor Methods

- **Efficiency + Unselfishness:** Average velocity reward + MARL for general profits [134].
- **Efficiency + Interpretability:** DDPG tuning of traditional controller parameters [135].

### C. Three-factor Methods

- **Efficiency + Interpretability + Training Efficiency:** Teacher-student model with traditional control as teacher [137].
- **Efficiency + Interpretability + Unselfishness:** Ramp metering (RM) with Q-learning for speed, transparency, and cooperation [138].
- **Safety + Efficiency + Unselfishness:** MARL for cooperative ramp merging [136].

### D. Four-factor Methods

- **Efficiency + Training Efficiency + Unselfishness + Interpretability:** DDPG-assisted ramp metering and variable speed limit (VSL) [139].
- **Safety + Efficiency + Training Efficiency + Interpretability:** APF + MPC with DDQN [140]; safety/efficiency rewards + IDM with IPPO [142].
- **Safety + Efficiency + Training Efficiency + Unselfishness:** DIM with DDPG for cooperation intentions [141].

### E. Five-factor Methods

- Safety supervisor filtering + rule-based constraints + MARL for general profits [143].
- Nash-based game + adversarial constraints + velocity and collision rewards [144].
- DRAC (deceleration rate to avoid crash) + multi-state representations + vehicle cooperation [145].

---

### Table II: Evaluation of DRL-Based Decision Making in On-ramp Merging

| Ref. | Safety | Efficiency | Training Efficiency | Unselfishness | Interpretability |
|------|--------|------------|---------------------|---------------|-----------------|
| [130] | — | Reward function | — | — | — |
| [131] | — | Travel time reward | — | — | — |
| [132] | Safety factor | — | — | — | — |
| [133] | Collision-free driving | — | — | — | — |
| [134] | — | Average velocity reward | — | MARL | — |
| [135] | — | Error state reduction | — | — | Traditional controller |
| [136] | Distance penalty | Distance minimization | — | MARL | — |
| [137] | — | Trip time difference | Teacher-student model | — | Traditional control |
| [138] | — | Speed comparison | — | Ramp metering | Ramp metering |
| [139] | — | DDPG-assisted RM | DDPG | RM and VSL | RM and VSL |
| [140] | APF | MPC with DDQN | MPC with DDQN | — | APF |
| [141] | Collision/stop penalty | DIM with DDPG | DIM with DDPG | HDV intentions | — |
| [142] | Safety reward | Efficiency reward | IPPO | — | IDM |
| [143] | Crash evaluation | Stable speed assessment | Safety supervisor | MARL | Rule-based constraints |
| [144] | Collision rewards | Velocity ratio | Adversarial constraints | Nash-based game | Transparent game process |
| [145] | DRAC | Velocity ratio reward | Multi-state rep. | Vehicle coop. | DRAC |

---

## VI. DRL-Based Decision-Making at Roundabouts

### A. Single-factor Methods

- **Efficiency:** Soft Actor-Critic (SAC) with higher peak rewards [146].
- **Training Efficiency:** Action repeat + asynchronous advantage [147]; ODD-embedded DQN [148].

### B. Dual-factor Methods

- **Efficiency + Training Efficiency:** Conditional Representation Model (CRM) [149].
- **Training Efficiency + Interpretability:** Expert-labeled data as guidance [150].
- **Safety + Efficiency:** Desired velocity + allowable relative distance reward [151].
- **Training Efficiency + Interpretability:** Optimization-embedded DRL with transparent model-based optimization [152].
- **Safety + Unselfishness:** Collision penalties + MARL for collective benefits [153].
- **Safety + Interpretability:** Collision penalties + gradual training mode [154].

### C. Three-factor Methods

- **Safety + Efficiency + Training Efficiency:** Safety/efficiency rewards + TRPO for faster convergence [155]; LSTM-embedded actor-critic [156]; reward normalization + multiple parallel environments [157].

### D. Four-factor Methods

- **Safety + Efficiency + Training Efficiency + Unselfishness:** Synthetic representation mechanism + MARL [158].
- **Safety + Efficiency + Training Efficiency + Interpretability:** IDM + interval prediction model + crash/speed rewards [58]; DDPG + DQN + NMPC integration [159].

### E. Five-factor Methods

All five DDTUI factors addressed via KAN-enhanced DQN + rule-based action inspector + route planning [99].

---

### Table III: Evaluation of DRL-Based Decision Making at Roundabouts

| Ref. | Safety | Efficiency | Training Efficiency | Unselfishness | Interpretability |
|------|--------|------------|---------------------|---------------|-----------------|
| [146] | — | SAC peak rewards | — | — | — |
| [147] | — | — | Action repeat, async advantage | — | — |
| [148] | — | — | ODD-embedded DQN | — | — |
| [149] | — | CRM | CRM | — | — |
| [150] | — | — | Expert guidance | — | Expert guidance |
| [151] | Allowable relative distance | vd | — | — | — |
| [152] | — | — | Optimization-embedded DRL | — | Model-based optimization |
| [153] | Collision penalties | — | — | MARL | — |
| [154] | Collision penalties | — | — | — | Gradual training |
| [155] | Safety rewards | Efficiency rewards | TPRO | — | — |
| [156] | Non-collision rewards | Velocity difference rewards | LSTM actor-critic | — | — |
| [157] | Fewer crashes | Higher success rates | Reward normalization | — | — |
| [158] | Safety distance | Velocity ratio | Synthetic representation | MARL | — |
| [58] | Crash penalties | High-speed rewards | Interval prediction | — | IDM |
| [159] | Collision penalties | Vehicle-stop penalties | DDPG, DQN, NMPC | — | NMPC |
| [99] | Rule-based inspector | High-speed rewards | KAN-DQN | Rule-based planning | Rule-based inspector |

---

## VII. DRL-Based Decision-Making at Unsignalized Intersections

### A. Single-factor Methods

- **Efficiency:** Velocity difference reward + penalty for low speed [160]; constant penalty until target exit reached [161].

### B. Dual-factor Methods

- **Efficiency + Training Efficiency:** Total waiting time reward + background removal ResNet [162]; D2-TSP with DDQN [169]; CIM-enhanced DQN [170].
- **Efficiency + Interpretability:** IDM for vehicle following + velocity difference reward [163]; safety-based rule policy [164]; MPC with TD3 [165]; gridded coordination zone [166].
- **Efficiency + Training Efficiency:** DQN with common and specific sub-tasks [167].
- **Training Efficiency + Unselfishness:** Incentive communication-assisted MARL [168].

### C. Three-factor Methods

- **Efficiency + Unselfishness + Training Efficiency:** MARL + multi-agent DQN [171].
- **Efficiency + Training Efficiency + Interpretability:** Deep Q-learning + transfer learning + IDM [172].
- **Safety + Efficiency + Training Efficiency:** SAC with spatial-temporal attention [173]; randomized prior function (RPF) for ensemble Bayesian posterior [174].

### D. Four-factor Methods

- **Safety + Efficiency + Training Efficiency + Unselfishness:** Autonomous Intersection Management (AIM) + LSTM + MARL [175].
- **Safety + Efficiency + Training Efficiency + Interpretability:** Mix-Attention Network + IDM [176].

### E. Five-factor Methods

All five factors addressed via value decomposition-based multi-agent deep Q-learning (VD-MADQL) + IDM + MARL [177].

---

### Table IV: Evaluation of DRL-Based Decision Making at Unsignalized Intersections

| Ref. | Safety | Efficiency | Training Efficiency | Unselfishness | Interpretability |
|------|--------|------------|---------------------|---------------|-----------------|
| [160] | — | Velocity difference reward | — | — | — |
| [161] | — | Time penalty | — | — | — |
| [162] | — | Total waiting time | Background removal ResNet | — | — |
| [163] | — | Velocity difference reward | — | — | IDM |
| [164] | — | Velocity-based reward | — | — | Safety-based rule policy |
| [165] | — | Safe distance reward | — | — | MPC with TD3 |
| [166] | — | Velocity ratio reward | — | — | Gridded coordination zone |
| [167] | — | Goal attainment reward | DQN with sub-tasks | — | — |
| [168] | — | — | Incentive communication | MARL | — |
| [169] | — | D2-TSP DDQN | — | — | — |
| [170] | — | CIM-enhanced DQN | CIM-enhanced DQN | — | — |
| [171] | — | Low-speed penalty | Multi-agent DQN | MARL | — |
| [172] | — | High-velocity reward | DQL with transfer learning | — | IDM |
| [173] | Collision penalties | High-velocity reward | SAC with attention | — | — |
| [174] | Collision penalties | Goal attainment reward | RPF | — | — |
| [175] | AIM | Constant time penalty | AIM and LSTM | MARL | — |
| [176] | Collision penalties | Goal attainment reward | Mix-Attention Network | — | IDM |
| [177] | Collision penalties | Low-velocity penalty | VD-MADQL | MARL | IDM |

---

## VIII. Conclusion and Discussion

This survey presents a comprehensive overview of the current state of the art in DRL-based decision-making for autonomous vehicles, covering highways, on-ramp merging, roundabouts, and unsignalized intersections, evaluated through the DDTUI framework.

### Distribution of Evaluation Factors

**Table V: Occurrence and Ratio of Evaluation Factors Across Different Scenarios**

| Scenario | Safety | Efficiency | Training Efficiency | Unselfishness | Interpretability |
|----------|--------|------------|---------------------|---------------|-----------------|
| Highway | 23 (76.7%) | 20 (66.7%) | 11 (36.7%) | 13 (43.3%) | 11 (36.7%) |
| Ramp | 9 (56.25%) | 14 (87.5%) | 8 (50%) | 8 (50%) | 9 (56.25%) |
| Roundabout | 10 (62.5%) | 11 (68.75%) | 12 (75%) | 3 (18.75%) | 6 (37.5%) |
| Intersection | 5 (27.7%) | 17 (94.4%) | 12 (66.7%) | 4 (22.2%) | 6 (33.3%) |
| **Total** | **47 (58%)** | **63 (77.8%)** | **44 (54.3%)** | **29 (35.8%)** | **32 (39.5%)** |

*Numbers in parentheses indicate the percentage of total studies for each factor.*

### Key Observations

- **Efficiency** is the most frequently addressed factor (77.8% overall), prioritized at intersections (94.4%) and ramps (87.5%).
- **Safety** is most emphasized on highways (76.7%), reflecting the importance of accident prevention at high speeds.
- **Training efficiency** is particularly significant at roundabouts (75%) and intersections (66.7%).
- **Interpretability** is most valued at ramps (56.25%) and highways (36.7%).
- **Unselfishness** receives the least overall attention (35.8%), though highways and ramps give it more focus.

### Future Challenges

1. **Achieving a balance between all five DDTUI factors in a single framework:** Very few studies managed to incorporate all five factors simultaneously — only 3 out of 16 roundabout studies and 1 out of 19 intersection studies addressed all five. Future research should develop integrated frameworks that holistically balance DDTUI concurrently.

2. **Improving the interpretability of DRL models without sacrificing performance:** Less than 40% of reviewed papers explicitly addressed interpretability, and most used only one interpretability method. Future work should explore combining multiple methods (e.g., APF and IDM concurrently) to enhance interpretability.

3. **Enhancing the unselfishness of AVs in complex multi-agent environments:** While ~50% of studies use MARL to promote unselfishness, real-world traffic uncertainties remain challenging. Future research should explore more sophisticated MARL techniques grounded in real-world data, potentially combining game theory with driving style classification from real-world datasets.

---

## References

[1] Department for Transport, "Road accidents and safety statistics," 2023.

[2] D. Omeiza, H. Webb, M. Jirotka, and L. Kunze, "Explanations in autonomous driving: A survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 10142–10162, 2021.

[3] H. A. Ignatious, M. Khan et al., "An overview of sensors in autonomous vehicles," *Procedia Computer Science*, vol. 198, pp. 736–741, 2022.

[4] "Self-driving cars: A survey," *Expert Systems with Applications*, vol. 165, p. 113816, 2021.

[5] J. Perez, V. Milanés et al., "Autonomous driving manoeuvres in urban road traffic environment: a study on roundabouts," *IFAC Proceedings Volumes*, vol. 44, no. 1, pp. 13795–13800, 2011.

[9] H. Vijayakumar, D. Zhao et al., "A holistic safe planner for automated driving considering interaction with human drivers," *IEEE Transaction on Intelligent Vehicle*, vol. 9, no. 1, pp. 2061–2076, 2023.

[14] M. Abdel-Aty and S. Ding, "A matched case-control analysis of autonomous vs human-driven vehicle accidents," *Nature Communications*, vol. 15, no. 1, p. 4931, 2024.

[22] R. Tian, S. Li et al., "Adaptive game-theoretic decision making for autonomous vehicle control at roundabouts," in *Proceedings of the IEEE Conference on Decision and Control*, 2018, pp. 321–326.

[25] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.

[35] Z. Tian, D. Zhao et al., "Efficient and balanced exploration-driven decision making for autonomous racing using local information," *IEEE Transactions on Intelligent Vehicles*, 2024.

[44] W. Yue, X. Wu, C. Li, N. Cheng, P. Duan, and Z. Han, "Navigating the impact of connected and automated vehicles on mixed traffic efficiency," *IEEE Internet of Things Journal*, 2024.

[45] B. Toghi, R. Valiente, D. Sadigh, R. Pedarsani, and Y. P. Fallah, "Social coordination and altruism in autonomous driving," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 12, pp. 24791–24804, 2022.

[47] X. Huang, D. Kroening, W. Ruan et al., "A survey of safety and trustworthiness of deep neural networks," *Computer Science Review*, vol. 37, p. 100270, 2020.

[48] F.-L. Fan, J. Xiong, M. Li, and G. Wang, "On interpretability of artificial neural networks: A survey," *IEEE Transactions on Radiation and Plasma Medical Sciences*, vol. 5, no. 6, pp. 741–760, 2021.

[58] J. Wang, J. Wu, X. Zheng, D. Ni, and K. Li, "Driving safety field theory modeling and its application in pre-collision warning system," *Transportation Research Part C*, vol. 72, pp. 306–324, 2016.

[75] C. M. Martinez, M. Heucke, F.-Y. Wang, B. Gao, and D. Cao, "Driving style recognition for intelligent vehicle control and advanced driver assistance: A survey," *IEEE Transaction on Intelligent Transportation System*, vol. 19, no. 3, pp. 666–676, 2017.

[87] D. Leslie, "Understanding artificial intelligence ethics and safety," *arXiv preprint arXiv:1906.05684*, 2019.

[95] A. Baheri, S. Nageshrao et al., "Deep reinforcement learning with enhanced safety for autonomous highway driving," in *Proc. IEEE Intelligent Vehicles Symposium*, 2020, pp. 1550–1555.

[96] M. Kaushik, V. Prasad et al., "Overtaking maneuvers in simulated highway driving using deep reinforcement learning," in *2018 IEEE Intelligent Vehicles Symposium*, 2018, pp. 1885–1890.

[97] N. Albarella, D. G. Lui et al., "A hybrid deep reinforcement learning and optimal control architecture for autonomous highway driving," *Energies*, vol. 16, no. 8, p. 3490, 2023.

[99] Z. Lin, Z. Tian et al., "A conflicts-free, speed-lossless KAN-based reinforcement learning decision system for interactive driving in roundabouts," *arXiv preprint arXiv:2408.08242*, 2024.

[101] S. Shalev-Shwartz, S. Shammah et al., "Safe, multi-agent, reinforcement learning for autonomous driving," *arXiv preprint arXiv:1610.03295*, 2016.

[105] M. Treiber, A. Hennecke, and D. Helbing, "Congested traffic states in empirical observations and microscopic simulations," *Physical Review E*, vol. 62, no. 2, p. 1805, 2000.

[106] S. Nageshrao, H. E. Tseng, and D. Filev, "Autonomous highway driving using deep reinforcement learning," in *Proc. IEEE International Conference on Systems, Man and Cybernetics*, 2019, pp. 2326–2331.

[129] S. Cheng, B. Yang, Z. Wang, and K. Nakano, "Spatio-temporal image representation and deep-learning-based decision framework for automated vehicles," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 12, pp. 24866–24875, 2022.

[143] D. Chen, M. R. Hajidavalloo et al., "Deep multi-agent reinforcement learning for highway on-ramp merging in mixed traffic," *IEEE Transactions on Intelligent Transportation Systems*, vol. 24, no. 11, pp. 11623–11638, 2023.

[144] X. He, B. Lou et al., "Robust decision making for autonomous vehicles at highway on-ramps," *IEEE Transactions on Intelligent Transportation Systems*, vol. 24, no. 4, pp. 4103–4113, 2023.

[177] Z. Guo, Y. Wu, L. Wang, and J. Zhang, "Coordination for connected and automated vehicles at non-signalized intersections," *IEEE Transactions on Vehicular Technology*, vol. 72, no. 3, pp. 3025–3034, 2023.

---

*Full reference list available in the original paper: arXiv:2501.01886v1*
