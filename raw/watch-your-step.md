Abstract—Although legged robots demonstrate impressive mo-
bility on rough terrain, using them safely in cluttered environ-
ments remains a challenge. A key issue is their inability to avoid
stepping on low-lying objects, such as high-cost small devices or
cables on flat ground. This limitation arises from a disconnection
between high-level semantic understanding and low-level control,
combined with errors in elevation maps during real-world oper-
ation. To address this, we introduce SemLoco, a Reinforcement
Learning (RL) framework designed to avoid obstacles precisely
in densely cluttered environments. SemLoco uses a two-stage
RL approach that combines both soft and hard constraints.
It performs pixel-wise foothold safety inference, which enables
more accurate foot placement. Additionally, SemLoco integrates
semantic map, allowing it to assign traversability costs instead
of relying only on geometric data. SemLoco greatly reduces
collisions and improves safety around sensitive objects, enabling
reliable navigation in situations where traditional controllers
would likely cause damage. Experimental results further show
that SemLoco can be effectively applied to more complex,
unstructured real-world environments. A demo video can be
viewed at https://youtu.be/FSq-RSmIxOM.
I. Introduction
While quadruped robots have achieved exceptional suc-
cess in robust locomotion and agile tracking across complex
outdoor terrains [1]–[3], their performance often degrades in
densely cluttered indoor environments that demand slow, de-
liberate navigation. As their deployment transitions to indoor,
human-centric spaces, a new critical requirement emerges. In
these environments, the primary objective shifts from merely
maintaining the robot’s dynamic balance to ensuring the safety
of the surrounding environment. The core problem is that
robots must execute high-precision foothold selection to avoid
stepping on small, low-lying, or fragile objects (e.g., cables,
small devices) scattered on the ground. It must proactively
protect the human-centric environment through fine-grained
contact point control.
Existing locomotion and navigation frameworks have made
significant progress but fall short in this precise interaction.
Current vision-based reinforcement learning (RL) policies and
geometric planners effectively utilize exteroceptive perception
to build elevation or occupancy maps for obstacle avoidance
[2], [4], [5]. However, they lack explicit semantic prediction.
Because they rely heavily on geometric representations, small
objects (often only a few centimeters high) are frequently
smoothed out as sensor noise or misclassified as traversable
ground. Furthermore, while recent studies incorporate seman-
tic data to enrich perception [6]–[8], they predominantly apply
semantic constraints at the macro-level path or trajectory
planning stage. They rarely pass these semantic preferences
down to the joint-level control or explicit foothold selection.
The core challenge in cluttered, human-centric environ-
arXiv:2603.02657v2 [cs.RO] 4 Apr 2026
ments lies in bridging the gap between high-level semantic
perception and high-frequency, low-level locomotion control.
In dense scenarios, simply altering the global path is insuf-
ficient. Instead, the system must directly translate semantic
properties into explicit foothold constraints, enabling the robot
to precisely step over or around low-lying obstacles without
compromising its dynamic stability.
To address this challenge, we propose the SemLoco
framework, which directly embeds semantic information into
low-level locomotion controller to achieve explicit semantic
foothold prediction. Unlike traditional pipelines that decouple
semantic navigation from geometric locomotion, our approach
translates visual semantics into dynamic traversability costs
that directly penalize unsafe foot placements. Powered by a
two-stage RL training strategy—progressing from soft con-
straints with virtual objects to hard constraints with real rigid
obstacles—SemLoco bypasses the ambiguities of geometric
maps, ensuring precise and safe navigation in highly cluttered
environments. In summary, the main contributions of this work
are as follows:

SemLoco Framework: We propose the SemLoco frame-
work, which solves the exploration trap in the navigation
of dense obstacles. This decouples spatial exploration
from strict dynamics, enabling robust perception-motor
mappings without early-stage kinematic collapse.
Semantic-Geometry Decoupling: We introduce a control
paradigm that explicitly disentangles environmental se-
mantics from geometric elevation. This resolves the inher-
ent vulnerabilities of traditional elevation-based proxies,
making the system highly robust against the depth sensor
noise and specular reflections typical of small, fragile
objects.
Sim-to-Real Deployment and Validation: We rigorously
validate the proposed method across both simulations
and real-world scenarios featuring fragile equipment. Ex-
periments demonstrate that SemLoco drastically reduces
the step collision rate, proving its robust autonomy and
adaptability in cluttered environment.
II. Related Works
Locomotion Control: Deep RL has become the mainstream
paradigm for motion control of legged robots [9]. In re-
cent years, researchers have no longer focused solely on
walking stability but have gradually taken obstacle avoidance
capability in environmental interaction as a core objective.
In precise obstacle avoidance, foothold planning is a key
component. Early optimization-based methods (e.g., model
predictive control) were limited by computational efficiency
and map accuracy, making it difficult to handle environments
with dense obstacles [10], [11]. Recently, Rudin et al. proposed
using RL to learn various gaits in simulation and demonstrated
obstacle avoidance capabilities in environments with scattered
obstacles [12], but their strategy still relies on elevation maps
and has limited recognition ability for low-lying objects. To
address this issue, Miki et al. proposed mapping perceptual
information directly to foot trajectories through end-to-end

learning, achieving real-time obstacle avoidance in complex
terrains [2], but this method still lacks semantic understanding
of obstacles.
For the accuracy of footholds, Jenelten et al. further pro-
posed combining RL with whole-body control and introducing
a differentiable cost function to achieve high-precision avoid-
ance [13]. However, this method decouples foothold search
from gait phase and the robot’s current motion, making it
prone to generating incoherent footholds in dense obstacle en-
vironments and falling into local optima. SemLoco redesigns
the foothold calculation method by combining the Raibert
heuristic with dynamic grid search, directly integrating gait
phase and motion commands. This tightly couples foothold
adjustment with the robot’s dynamic behavior, significantly
improving precise avoidance performance in scenarios with
dense low-lying obstacles.
Semantic Perception in Robot Navigation: Beyond purely
geometry-based navigation, recent works leverage semantic
information to enhance planning. Several modular approaches
employ semantic segmentation models to obtain object masks
and assign heuristic costs to construct dense semantic maps
[14], [15] or semantic graphs [16]. [17] learns a seman-
tic traversability estimator through an automated annotation
pipeline on egocentric videos, simplifying the derivation of
traversal cost maps. Furthermore, semantic-aware planners
are built upon these representations to incorporate object-
level risk into navigation decisions [18], [19]. More recently,
[20] proposes an end-to-end path planning framework that
encodes semantic information in the latent space, improving
planning efficiency. [21] integrates semantic semantics into
quadrupedal navigation, yet its impact is confined to the
tuning of gait parameters, and velocity levels. [22] generates
a passability cost map for local path planning by extracting
rich image semantic features through the vision transformer
DINOv2. These semantic-aware methods primarily operate at
the high-level path planning stage and do not explicitly reason
about foothold selection for legged robots. In cluttered and
narrow environments, even if fragile or undesirable objects are
recognized by the high-level planner, the absence of foothold-
level constraints may still cause the robot to step on them,
limiting the effectiveness of semantic awareness in contact-
rich locomotion. To address the above issues, SemLoco for
the first time, combines semantic information with foothold
planning for legged robots, using it to assign traversability
costs to different types of obstacles, thereby establishing an
effective connection between high-level semantic understand-
ing and low-level motion control.
III. Methodology
A. Reinforcement Learning
1) Observation Space: The Symmetric Actor-Critic (A2C)
algorithm is adopted as the RL framework. The policy obser-
vations, Ot∈R^1513 , are defined as:
Ot=
h
Ccmdt , Bt, Opropriot , Oexterot
i
, (1)
Fig. 2: Framework of SemLoco. Sub-modules have different styles based on their functions. Among them, the red trapezoid represents the neural network,
the blue rectangle represents unprocessed raw data, and the green rectangle represents processed ready-to-use data. (a) Training in the simulator: In stage 1,
we use virtual obstacles (Highly yellow spheres). Although the robot walks on flat ground, it receives a virtual perception map containing height and semantic
information to simulate scenarios with real obstacles. In stage 2, we use rigid obstacles. The robot receives corresponding real perception information, and
fine-tunes the policy to improve task performance. (b) Deployment in the real world: Exteroception information is obtained by Odin1 and fed into the
semantic graph algorithm.

where Ctcmd∈R^3 are the desired velocity commands, including
the base linear velocities vcmdx ∈R^1 in the x-axes, vcmdy ∈R^1
in the y-axes and the base angular velocity ωcmdz ∈R^1 in the
yaw axis.
The Bt∈R^13 specify the behavior parameters used for gen-
erating stable four-beat quadrupedal contact pattern(walking),
which is inspired by [23], [24]. They are defined as:

Bt=

hbase, sfeet,θ 1 − 3 , tfeet, f, d, w, l

, (2)
where hbase ∈R^1 , sfeet ∈R^1 , f ∈R^1 represent the height
of robot’s base, the footswing height and the frequency of
contact. θ 1 − 3 ∈R^3 , tfeet∈R^4 , d ∈R^1 , w ∈R^1 , l ∈R^1 denote
the timing offsets for three pairs of feet, the contact state timer
for each foot, the duty, the stance width and length.
The Opropriot ∈R^57 denote proprioception measured from
IMU, joint encoders and velocity estimator:

Opropriot =

ˆvt,ωt, gt, qt, ̇qt, at− 1 , at− 2

, (3)
where, ˆvt∈R^3 denote the real base linear velocity, which is
estimated by the velocity estimator. The ωt∈R^3 , gt∈R^3 ,
qt∈R^12 and ̇qt∈R^12 denote the real base angular velocity,
gravity vector, joint angle and joint angular velocity, which
are measured from the joint encoder and IMU. The at− 1 ∈R^12
and at− 2 ∈R^12 are the previous actions of the first two steps.
The Oexterot ∈R^1440 are defined as information about ex-
teroception. Environmental information includes the elevation

map Helevt ∈R^720 and the semantic map Hsemt ∈R^720 , both
of which are centered at the center of gravity of the robot.
Both maps are defined over a region that spans 1.5 m in the
x-direction and 1.2 m in the y-direction. At 0.05m resolution,
this results in a 30 × 24 grid of sampled cells.
2) Action Space: The action at ∈R^12 , are defined as
the differences between the nominal position and the target
position for each of the twelve joints of the robot, which
are outputted by the actor network. A proportional derivative
(PD) controller is used to track desired joint positions by
converting positions to torques. The proportional gain kpand
the derivative gain kdare 20 and 0.5 respectively.
3) Network Architecture: The framework is illustrated as
Fig. 2. Our policy consists of two Multilayer Perceptrons
(MLPs) for the actor network and the critic network respec-
tively, a base velocity estimator and a Convolutional Neural
Network (CNN) encoder. The CNN encoder extracts height
features from elevation map and semantic features from se-
mantic map. The feature vector extracted by the CNN module
is concatenated with all other proprioceptive observations, and
the concatenated vector is then fed into an MLP with hidden
layer sizes of [512, 256, 128] and ELU activation function.
The base velocity estimator, which is a MLP with hidden layer
sizes [256, 128] and ELU activations, is trained in a supervised
manner.
B. Semantic-Aware Adaptive Foothold Planning

We proposed a semantic-aware Raibert Heuristic to over-
come the vulnerability of traditional controllers in cluttered
environments. Rather than relying solely on the neural network
for implicit obstacle avoidance, our approach explicitly inte-
grates local obstacle constraints into the kinematic planning
loop. By dynamically evaluating semantic collision costs over
a localized search grid, this method refines the nominal, blind
footholds into intrinsically safe targets before they are tracked
by the RL policy.
The standard Raibert Heuristic calculates the ideal foothold
for leg i based on the robot’s base coordinate system to
maintain velocity tracking and offset the effects of inertia and
angular velocity. [11], [25] The nominal stance position is
defined as Pinom= [xinom, yinom]Tbased on the desired stance
width w and length l. Let Tstancebe the stance duration, which
is derived from the desired gait frequency f and duty factor
d as Tstance=df. The nominal Raibert position Piraibertin the
body frame is calculated by incorporating the desired linear
velocity vcmdx , vcmdy and the yaw angular velocity ωcmdz :

Piraibert= Pinom+ ∆Pilin+ ∆Piyaw, (4)
where the translational offset ∆Pilinand the rotational offset
∆Piyaware defined as:

∆Pilin=
"Tstance
2 v
cmd
x
0
, (5)
∆Piyaw=
"
0
Tstance
2 ω
cmd
z · x
i
nom
, (6)
where the xinom encodes the longitudinal position of the
leg (positive for front legs, negative for rear legs), thereby
automatically assigning the correct lateral stepping direction
induced by the body’s yaw rotation.
Building upon this kinematic prior, our proposed method
performs a localized optimal search around the nominal Raib-
ert position Piraibertleveraging the semantic grid map Hsemt. We
construct a localized M×M search gridGiwith resolution ∆grid
centered around Piraibert. For each candidate point pk∈Gi, we
transform it to the world frame and check for virtual collisions
against known obstacles from the semantic module. Crucially,
rather than treating all spatial deviations equally, quadrupeds
exhibit strong kinematic preferences during obstacle avoid-
ance: shortening the stride (stepping backward relative to the
nominal point) is biomechanically safer than over-stretching
forward, while lateral deviations severely compromise the
support polygon and should be strictly avoided. To capture
this, we design a kinematically informed cost function J( pk)
composed of an asymmetric directional deviation cost and a
collision penalty:

J( pk) = Cdir( pk) + wcol· (^1) col( pk), (7)
where the directional deviation cost Cdir( pk) is defined in the
robot’s local heading frame with longitudinal offset ∆x and
lateral offset ∆y:
Cdir( pk) = w+xmax(∆x, 0) + w−xmax(−∆x, 0) + wy|∆y|. (8)
To enforce the aforementioned kinematic priors, we strictly
set the penalty weights as wy ≫ w+x > w−x. The collision
penalty, (^1) col( pk), is an indicator function that evaluates to 1
if the candidate point pktriggers an Axis-Aligned Bounding
Box (AABB) collision with any semantic obstacle, whose
boundaries are uniformly dilated by the foot radius rfootto
ensure a strict safety margin, and 0 otherwise. We assign
a highly punitive weight wcol ≫ wy to ensure that unsafe
locations are fundamentally vetoed.The final optimized target
foothold Pitargetis derived by minimizing the cost function
across all candidates:
Ptargeti = arg min
pk∈Gi
J( pk). (9)
In this task, the semantic information is defined as
”fragility”(from 0 to 1). To obtain semantic information, we
construct a semantic traversability map by projecting semantic
predictions from RGB images into a global grid map. At
time t, an open-vocabulary semantic segmentation model [26]
predicts a class probability distribution P(St= c|u, v) for each
pixel (u, v) in the RGB image. Each pixel is associated with a
3D point from the aligned point cloud and projected to a grid
cell g in the elevation map [5]. For each cell, we maintain a
probability distribution over semantic classes Pt(c|g), which is
updated using Bayesian fusion:
Pt(c|g)∝ P(St= c|pi)Pt− 1 (c|g), (10)
where piis the observed point projected to cell g. Each class
c is assigned a predefined traversal cost φ(c), and the final
traversability cost of the cell is computed as the expected cost:
M(g) =

X
c∈C
Pt(c|g)φ(c). (11)
C. Reward Functions
To facilitate the learning of highly dynamic yet safe loco-
motion, we adopt a multiplicative reward structure from [27].
The total reward Rtotalis defined as the product of primary
task rewards and an exponential auxiliary penalty multiplier:
Rtotal= rprimary· exp

cpenaltyrpenalty

, (12)
where rtask≥ 0 encourages the completion of primary objec-
tives, raux≤ 0 penalizes undesirable behaviors, and cauxis a
positive scaling factor. This form largely resolves the trade-off
issue between different rewards. The terms in (12) are further
elaborated below.
1) Primary Objective Rewards (rprimary): In highly clut-
tered environments, tracking the velocity command is insuf-
ficient for survival, and precise foot placement is equally
important. Thus, our primary reward integrates both kinematic
velocity tracking and semantic-aware foothold tracking:
rprimary= wvel· rvel+ wsem· rsemantic, (13)
where the velocity tracking term rveland the semantic-aware
foothold tracking term are defined as:
rvel= exp


−
∥vxy− vcmdxy ∥^22
σv


+ exp
−
(ωz− ωcmdz )^2
σω
!
, (14)
rsem=
X^4
i= 1
1 iswing· exp


−
∥ pifoot− Pitarget∥^22
σfoot


, (15)
where 1 iswing∈{ 0 , 1 } is the swing phase indicator. The semantic
tracking term rsemenforces the policy to accurately step on the
optimal, collision-free targets Pitargetgenerated by our explicit
spatial planner. By formulating this as a dense positive tracking
reward during the swing phase, we provide a continuous
gradient for the policy.

State and Dynamic Constraint Penalties (rpenalty): The
penalty term, rpenalty=
P
wkck, aggregates various negative
constraints to bridge the reality gap, maintain body balance,
and optimize energy efficiency. To ensure robust obstacle
traversal, we design a minimum foot clearance penalty rather
than a strict trajectory tracking constraint. This encourages the
policy to lift its feet sufficiently high to avoid toe-stubbing,
without penalizing it for stepping higher than the reference
when navigating severe cluttered obstacles.
During the swing phase, the normalized swing progress
φiswing∈ [0, 1] for leg i is calculated based on the stance duty
factor. To synthesize an asymmetric safety-first swing profile
characterized by rapid liftoff and a prolonged airborne plateau
(Fig. 3), we formulate the time-varying reference height ziref
using a square-root sine function:

ziref= sfeet·
q
sin

πφiswing

+ δz, (16)
where sfeet is the footswing height and δzis a fixed safety
margin (δz= 0 .02 m).
To enforce this as a lower-bound safety threshold, we utilize
a ReLU function, mathematically expressed as max(0, x), to
selectively penalize the foot only when its actual vertical
position zifootswingfalls below the reference curve. The clearance
penalty is defined as:

cclearance=
X^4
i= 1
1 iswing· max(0, ziref− zifootswing)^2 , (17)
where the indicator function 1 iswing∈{ 0 , 1 } ensures the penalty
is solely evaluated during the swing phase.
Furthermore, dynamic constraints are applied to strictly re-
strict large joint torques and velocities. Finally, severe collision
penalties are imposed on non-foot rigid bodies (e.g., thighs and
base) to guarantee physical safety.

D. Learning Semantic-Aware Locomotion via Two-Stage RL

Learning semantic-aware locomotion from scratch in a
highly cluttered environment poses a significant exploration
challenge. Direct training with rigid physical obstacles fre-
quently causes a large number of penalties, severely restricting
the agent’s ability to collect meaningful reward signals and
leading to overly conservative policies. Consequently, the
policy often converges to suboptimal local minima, failing to
master either stable locomotion or reliable obstacle avoidance.
To address this, we design a two-stage RL approach.

1) Virtual Obstacle Stage: In the first stage, we create
a virtual training environment where obstacles are visually
and semantically represented in Htelevand Hsemt , but physical
collisions are disabled, as shown in Fig. 2(a). The robot’s limbs
can pass through obstacles without triggering physical reaction
forces or ending the episode. This means the robot will neither
trip over obstacles nor push them away. During this stage, the
policy relies solely on the Semantic-Aware Raibert Heuristic
Reward (rsem) to adjust its foot placement. Working together
with the curriculum, this safe stage allows the policy to
gradually progress from basic stable locomotion to semantic-
aware foot placement. This helps the robot build a robust
mapping from visual inputs to spatial leg-swing coordination,
without the severe penalties of physical collisions.
Fig. 3: Footswing tracking for each leg.
2) Rigid Obstacle Stage: After the policy converges in the
soft dynamic stage, we transfer it to the second stage, where
full physical interactions are enabled. Obstacles are created
as rigid bodies, and their density and variety are increased
compared to the first stage, further improving the robot’s
obstacle avoidance ability. In this stage, the policy adjusts the
previously learned motion patterns to handle complex physical
interactions, such as friction.
E. Curriculum Learning for Generalization and Robustness
We design two types of curriculum:Velocity Curriculum and
Obstacle Density Curriculum.
1) Velocity Curriculum: We use a grid-based adaptive
curriculum form [28]. The linear velocities and angular veloc-
ities are uniformly discretized into a multi-dimensional grid
of difficulty bins. During training, the sampling probabilities
of these bins are dynamically updated based on the agent’s
historical tracking rewards. In addition to velocity commands,
training can be conditioned on behavior parameters. While this
approach did not significantly benefit our primary obstacle-
avoidance task, expanding the instruction space remains an
effective strategy for versatile terrain adaptation.
2) Obstacle Density Curriculum: If the robot starts train-
ing in an environment with too many obstacles, it may often
fail and receive high penalties, leading to a conservative
strategy where it just stays in place. To avoid this, we use
TABLE I: Ablation experiment under different obstacle densities.
Policy 10 obstacles m
− (^2) 15 obstacles m− (^2) 20 obstacles m− (^2) 25 obstacles m− 2
D↑ S ↑ C ↓ D↑ S ↑ C ↓ D↑ S ↑ C ↓ D↑ S ↑ C ↓
(m) (%) (%) (m) (%) (%) (m) (%) (%) (m) (%) (%)
Ours 9.67 97.00 2.37 9.51 92.60 8.78 9.29 87.40 13.45 8.94 82.20 24.
Blind 5.53 28.40 51.11 5.38 23.20 75.56 5.08 15.60 86.67 4.62 9.40 93.
Ours w/o Virtual Obstacle Stage 6.19 51.60 13.18 5.88 45.80 20.88 5.42 39.40 29.67 5.06 30.80 36.
Ours w/o ReLU Clearance Penalty 8.22 75.60 16.48 7.71 62.80 23.07 7.25 53.60 31.87 6.97 46.20 39.
Ours w/o Semantic Map 9.55 97.00 7.46 9.36 92.20 11.28 8.92 84.20 17.33 8.59 77.80 26.
an obstacle density curriculum to control the difficulty of the
training environment. We use ρobstto represent the density
of obstacles. As the robot gets better at tracking velocity
commands, we gradually increase ρobstuntil it reaches the
maximum level. Fig. 4 provides examples.
Fig. 4: Examples of obstacle curricula.
F. Training Setup in Simulator and Real-world Deployment
Training: We conduct two-stage training environments in
the IsaacLab framework [29] and use Proximal Policy Op-
timization [30] to train policies. Each stage is trained with
4096 parallel environments of Unitree Go2 robot. We train on
a single NVIDIA RTX 5090 GPU, with the soft constraint
stage taking 1.5 hours and the hard 3.5 hours.
Real-World Deployment: We deploy SemLoco on a Uni-
tree Go2 equipped with a head-mounted RGB-ToF module
Odin1 and an AGX Orin. To preserve the robot’s nominal
dynamics, all computations—including semantic segmenta-
tion, map fusion, and policy inference—are executed locally.
The control policy runs at 50 Hz, while the asynchronous
perception front-end updates at 10 Hz.
IV. Experiments
A. Experimental Setup
We compare SemLoco with baselines and ablations:

Blind: A blind policy with standard walking gait.
Ours w/o Virtual Obstacle Stage: An ablation of our
policy that removes the first stage.
Ours w/o ReLU Clearance Penalty: This policy uses a
strict trajectory tracking constraint−(z−sfeet·sin

πφiswing

)^2
to replace the ReLU Clearance Penalty.
Ours w/o Semantic Map: Search using high-level infor-
mation instead of semantic information.
Ours (full): Our policy trained via a two-stage RL ap-
proach with semantic input and ReLU clearance penalty.
Experiments are conducted in Isaac Sim. For the simulation
evaluation, we benchmark each policy on a 10m× 2m straight-
line track populated with varying densities of obstacle. The
robot is commanded to traverse the track with a target forward
velocity of 0.4 m/s.
We evaluate the performance using three primary metrics:
Success Rate (S , %): The percentage of trials in which
the robot successfully completes the obstacle course.
Average Distance to Failure (D, m): The average dis-
tance traveled by the robot before encountering a failure.
(This metric is selected to mitigate evaluation biases
caused by bimodal performance distributions.)
Step Collision Rate (C, %): The percentage of footsteps
that collide with obstacles relative to the total number of
steps taken during the traversal.
B. Simulation Evaluation
1) Quantitative Analyses: The quantitative results of our
experiments are summarized in Table I. To ensure a fair and
statistically robust comparison, all policies are evaluated across
an identical set of 500 randomly generated environments for
each density configuration. The reported quantitative metrics
are averaged over these 500 independent trials. In general,
SemLoco consistently outperforms all baseline methods
across different obstacle densities, particularly in highly dense
environments.
Traversal Capability and Survival: As expected, the
Blind policy finds it almost impossible to navigate entire
obstructed terrains, resulting in the lowest average distance
to failure (D) and the lowest success rate (S ). In contrast,
Ours (full) maintains a near-perfect D of approximately 9 m,
demonstrating robust locomotion and sustained survival even
when faced with heavily cluttered obstacles.
Obstacle Avoidance Performance: The effectiveness of our
two-stage RL approach is most evident in the step collision
rate (C). Ours (full) achieves the lowest C among all methods,
effectively reducing unintended foot-obstacle collisions by
74% to 95% (depending on obstacle density) relative to the
blind baseline. Conversely, the Blind policy possesses almost
no capability for obstacle avoidance; as environmental density
increases, almost every footstep inevitably results in a collision
with the terrain debris.
2) Detailed Ablation Analysis: We perform a detailed
ablation analysis focusing on the two-stage RL curriculum,
the reward formulation, and the exteroceptive modalities, as
detailed in the Table I.
Effectiveness of the Two-Stage Curriculum: Removing
the initial virtual obstacle stage (Ours w/o Virtual Obstacle
Fig. 5: Qualitative comparison of obstacle avoidance in a real cluttered environment.We compare our semantic-aware locomotion policy (Bottom) against an
elevation-only baseline (Top). Elevation maps completely fail to capture small, low-profile objects (Fig 1). Consequently, Ours w/o Semantic Map degrades
into a nearly blind policy, failing to adjust foot placements to avoid obstacles. As highlighted by the colored circles, the robot frequently steps on or kicks
the scattered objects; the resulting environmental disruption is highly evident when comparing the Pre-Traversal and Post-Traversal states (the first and last
columns). In contrast, our full framework leverages explicit semantic cues to achieve precise obstacle avoidance. The environment remains almost entirely
undisturbed, with the exception of a negligible displacement of a power bank caused by an incidental grazing contact during the swing leg’s descent.

Stage) leads to catastrophic performance degradation. For
instance, at a moderate density of 15 obstacles/m^2 , the success
rate (S ) plummets from 92.60% to 45.80%, and the mean
distance to failure (D) drops to 5.88 m. Without the soft
dynamic constraints provided in the first stage, it is difficult
for the agent to learn stable obstacle avoidance and stable gait
from the physical feedback with obstacles during its early
exploration phase. This highlights that our two-stage RL is
indispensable for resolving the fundamental conflict between
spatial exploration and strict temporal gait adherence.

Flexible Foot Clearance vs. Strict Tracking: When sub-
stituting our unilateral ReLU-based clearance penalty with a
strict bilateral trajectory tracking constraint (Ours w/o ReLU
Clearance Penalty), we observe a significant deterioration in
obstacle avoidance. At 15 obstacles/m^2 , the step collision rate
(C) increases by a factor of 2.6, rising from 8.78% to 23.07%.
A strict trajectory constraint forces the foot to rigidly adhere
to a predefined mathematical curve, depriving the agent of the
spatial freedom required to dynamically elevate its feet higher
than the nominal trajectory when negotiating unpredictable
debris. The ReLU-based penalty, on the contrary, enforces only
a minimum safe clearance, granting the policy the necessary
flexibility to execute collision-free swing trajectories.

Semantic Awareness vs. Geometric Proxy: Interestingly,
the policy stripped of semantic inputs (Ours w/o Semantic
Map) maintains a Success Rate highly comparable to our full
pipeline across the physical clutter track (e.g., both achieve
97.00% at 10 obstacles/m^2 ). However, it exhibits a noticeably
higher Step Collision Rate (e.g., 7.46% vs. 2.37% at 10
obstacles/m²). This behavior is fundamentally expected in sim-
ulation. Since the scattered obstacles inherently possess promi-
nent physical heights, the neural network efficiently learns
to exploit extreme elevation gradients from the geometric
map as a proxy for unsafe footholds. However, we explicitly
retain this baseline to emphasize the conceptual ceiling of
elevation-only methods. While elevation serves as an effective
obstacle proxy on flat training terrains, it fails under real-
world complexities. First, on uneven topographies like stairs,
pure geometry becomes ambiguous, as height variation may

indicate a safe foothold. Second, small obstacles are easily
obscured by real sensor noise, underscoring the necessity of
explicit semantic information.
C. Sim-to-Real Experiments
To validate the practical viability of SemLoco, we conduct
qualitative real-world tests in an unstructured indoor environ-
ment. The test track is deliberately populated with fragile and
low-profile everyday clutter, including power cables, smart-
phones, boxes, and small stationery.
Sensor Noise and Map Modality Comparison: These
small, low-profile objects represent the exact edge cases where
traditional geometric proxies fail. In our visualization, the
physical depth sensor’s inherent noise floor completely ob-
scures the millimeter-level height variations of these items,
rendering the elevation map virtually indistinguishable from
the flat ground. In stark contrast, the semantic map distinctly
highlights these objects as high-cost hazards. This visual val-
idation confirms the absolute necessity of semantic-geometry
disentanglement for reliable real-world perception.
Semantic Avoidance: The core intelligence of SemLoco
lies not merely in elevating the swing leg, but in its proactive
foothold judgment. During experiment, if our policy predicts
a foothold landing directly on a semantically hazardous object
(e.g., a smartphone), the policy dynamically intervenes. It
actively shortens the stride or step off to the side to secure
a safe foothold immediately before the object. Subsequently,
on the next gait cycle, it commands a high-clearance swing
trajectory to cleanly step over the hazard. While a traditional
blind controller might occasionally step over an object by pure
luck if its foot happens to land perfectly in front of it, it
fundamentally lacks this spatial judgment and proactive stride
modulation.
Baseline Comparison: We qualitatively compare SemLoco
against a Ours w/o Semantic Map baseline. The baseline
policy exhibits a severe ”bulldozer” behavior: entirely obliv-
ious to the surrounding fragility, it marches forward rigidly.
Consequently, it collides with, kicks away, and aggressively
tramples the scattered debris, posing a critical danger to both
the robot and the objects. Conversely, SemLoco demonstrates
significantly more deliberate navigation, effectively mitigating
the majority of severe collisions with fragile debris. This high-
lights the potential of semantic-aware locomotion to improve
operational safety in unstructured, real-world environments.

V. Conclusion
In this work, we introduced SemLoco, a semantic-aware
locomotion framework that achieves precise foothold selection
in dense, unstructured environments. By explicitly disentan-
gling environmental semantics from geometric heights and
employing a two-stage reinforcement learning curriculum,
SemLoco successfully overcomes the geometric ambiguities
and perceptual failure modes inherent to traditional elevation
proxies. Extensive evaluations across both simulation and real-
world indoor scenarios confirm that our approach significantly
mitigates step collision rates.
Limitation and Future Work: While the proposed Sem-
Loco framework significantly enhances semantic-aware loco-
motion, simulation and deployment reveal limitations in both
rigid-body dynamics and semantic representation. In terms of
dynamics and hardware, extreme asymmetric footholds induce
uncompensated angular momentum, occasionally resulting in
observable yaw drift or pushing the joints near kinematic
singularities. Additionally, it is extremely difficult for the real
robots to replicate the foot trajectory and footswing height as
seen in simulations. A noticeable sim-to-real gap stems from
the semantic simulation environment. To manage computa-
tional overhead during training, complex real-world objects
are approximated using simplified geometric primitives (e.g.,
cubes). This approximation limits the policy’s exposure to
realistic object boundaries. Furthermore, our current pipeline
compresses environmental semantics into a ”fragility” cost.
However, real-world applications encompass open-vocabulary
semantics that dictate diverse interactive behaviors. Future
work will explore integrating high-fidelity semantic simula-
tors and Vision-Language-Action (VLA) models to achieve
comprehensive open-vocabulary semantic comprehension and
generalized locomotion.

References
[1] J. Hwangbo, J. Lee et al., “Learning agile and dynamic motor skills for
legged robots,” Science Robotics, vol. 4, no. 26, p. eaau5872, 2019.
[2] T. Miki, J. Lee et al., “Learning robust perceptive locomotion for
quadrupedal robots in the wild,” Science Robotics, vol. 7, no. 62, p.
eabk2822, 2022.
[3] A. Kumar, Z. Fu et al., “Rma: Rapid motor adaptation for legged robots,”
in Robotics: Science and Systems (RSS), 2021.
[4] N. Rudin, D. Hoeller et al., “Advanced skills by learning locomotion
and local navigation end-to-end,” in Proc. IEEE/RSJ Int. Conf. Intell.
Robots Syst. (IROS), 2022.
[5] P. Fankhauser, M. Bloesch, and M. Hutter, “Probabilistic terrain mapping
for mobile robots with uncertain localization,” IEEE Robot. Autom. Lett.,
2018.
[6] D. D. Fan, S. Dey et al., “Learning risk-aware costmaps for traversability
in challenging environments,” IEEE Robot. Autom. Lett., 2022.
[7] M. J. Miles, H. Biggie, and C. Heckman, “Terrain-aware semantic
mapping for cooperative subterranean exploration,” Frontiers in Robotics
and AI, vol. 10, p. 1249586, 2023.
[8] R. Yue, L. Feng et al., “Safety path planning for quadruped robots
optimized by multi-sensor fusion,” IFAC-PapersOnLine, vol. 59, no. 27,
pp. 55–60, 2025.
[9] S. Ha, J. Lee et al., “Learning-based legged locomotion: State of the
art and future perspectives,” The International Journal of Robotics
Research, vol. 44, no. 8, pp. 1396–1427, 2025.
[10] R. Grandia, F. Jenelten et al., “Perceptive locomotion through nonlinear
model-predictive control,” IEEE Transactions on Robotics, vol. 39, no. 5,
pp. 3402–3421, 2023.
[11] D. Kim, J. Di Carlo et al., “Highly dynamic quadruped locomotion
via whole-body impulse control and model predictive control,” arXiv
preprint arXiv:1909.06586, 2019.
[12] N. Rudin, D. Hoeller et al., “Advanced skills by learning locomotion
and local navigation end-to-end,” in Proc. IEEE/RSJ Int. Conf. Intell.
Robots Syst. (IROS), 2022.
[13] F. Jenelten, T. Miki et al., “Perceptive locomotion in rough terrain–online
foothold optimization,” IEEE Robot. Autom. Lett., 2020.
[14] G. Erni, J. Frey et al., “Mem: Multi-modal elevation mapping for
robotics and learning,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.
(IROS), 2023.
[15] D. Maturana, P.-W. Chou et al., “Real-time semantic mapping for
autonomous off-road navigation,” in Field and Service Robotics: Results
of the 11th International Conference. Springer, 2017, pp. 335–350.
[16] P. Kremer, H. Bavle et al., “S-nav: Semantic-geometric planning for
mobile robots,” arXiv preprint arXiv:2307.01613, 2023.
[17] Y. Kim, J. H. Lee et al., “Learning semantic traversability with ego-
centric video and automated annotation strategy,” IEEE Robot. Autom.
Lett., 2024.
[18] X. Cai, M. Everett et al., “Risk-aware off-road navigation via a learned
speed distribution map,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.
(IROS), 2022.
[19] S. Achat, J. Marzat, and J. Moras, “Path planning incorporating se-
mantic information for autonomous robot navigation,” in 19th Interna-
tional Conference on Informatics in Control, Automation and Robotics
(ICINCO) 2022. SCITEPRESS-Science and Technology Publications,
2022, pp. 285–295.
[20] P. Roth, J. Nubert et al., “Viplanner: Visual semantic imperative learning
for local navigation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA),
2024.
[21] Y. Yang, X. Meng et al., “Learning semantics-aware locomotion
skills from human demonstration,” in Proceedings of The 6th
Conference on Robot Learning (CoRL), ser. Proceedings of Machine
Learning Research, vol. 205, 2023, pp. 2205–2214. [Online]. Available:
https://proceedings.mlr.press/v205/
[22] S. Ægidius, D. Hadjivelichkov et al., “Watch your stepp: Semantic
traversability estimation using pose projected features,” in 2025 IEEE
International Conference on Robotics and Automation (ICRA). IEEE,
2025, pp. 2376–2382.
[23] G. B. Margolis and P. Agrawal, “Walk these ways: Tuning robot control
for generalization with multiplicity of behavior,” in Conf. Robot Learn.
(CoRL), 2023.
[24] J. Siekmann, Y. Godse et al., “Sim-to-real learning of all common
bipedal gaits via periodic reward composition,” in Proc. IEEE Int. Conf.
Robot. Autom. (ICRA), 2021.
[25] M. H. Raibert, Legged robots that balance. MIT press, 1986.
[26] Y. Sun, J. Wang et al., “Yolo-e: a lightweight object detection algorithm
for military targets,” Signal, Image and Video Processing, vol. 19, no. 3,
p. 241, 2025.
[27] G. Ji, J. Mun et al., “Concurrent training of a control policy and a
state estimator for dynamic and robust legged locomotion,” IEEE Robot.
Autom. Lett., 2022.
[28] G. B. Margolis, G. Yang et al., “Rapid locomotion via reinforcement
learning,” Int. J. Robot. Res., 2024.
[29] M. Mittal, P. Roth et al., “Isaac lab: A gpu-accelerated simulation frame-
work for multi-modal robot learning,” arXiv preprint arXiv:2511.04831,
2025.
[30] J. Schulman, F. Wolski et al., “Proximal policy optimization algorithms,”
arXiv preprint arXiv:1707.06347, 2017.