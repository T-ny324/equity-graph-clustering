# Phase 3 — Mathematical Reference

## Graph construction: kNN and information-theoretic edges

Two similarity measures, one graph topology. Equation numbers in parentheses refer to Saha, Gao & Gerlach (2022), *A survey of the application of graph-based approaches in stock market analysis and prediction*.

---

## 0. Setup

$N = 105$ assets, daily log returns $r_{i,s}$. At rebalance date $t$, take the trailing window

$$\mathcal{W}(t,T) = \{t-T+1,\dots,t\}, \qquad T = 252$$

and write $\mathbf{r}_i \in \mathbb{R}^{T}$ for asset $i$'s return vector over that window.

The pipeline is always three stages:

$$\underbrace{\mathbf{r}_i, \mathbf{r}_j}_{\text{returns}} \;\longrightarrow\; \underbrace{S_{ij}}_{\text{similarity}} \;\longrightarrow\; \underbrace{A_{ij}}_{\text{adjacency}}$$

**Stage 2 is the choice you are studying** (Pearson vs mutual information). Stage 3 is held fixed at kNN so the comparison is clean — a difference in results is then attributable to the similarity measure, not to a change in topology. That is the experimental design.

### 0.1 Market-mode removal (applies to both)

Before either similarity is computed, residualise:

$$\tilde{r}_{i,s} = r_{i,s} - \beta_i r_{m,s}, \qquad r_{m,s} = \frac{1}{N}\sum_{i} r_{i,s}$$

$r_m$ approximates the leading eigenvector of the correlation matrix — the common-mode component that connects everything to everything. Without removal, every similarity is inflated by a shared term and the graph is a hairball.

**This matters more for MI than for correlation.** MI is non-negative and unsigned, so it cannot distinguish "moves with the market" from "moves against the market" — the common mode contributes to MI regardless of sign, and there is no cancellation. Residualisation is therefore load-bearing, not optional, in the MI branch.

---

# Part A — The kNN graph

## A.1 Similarity from Pearson correlation

$$\rho_{ij} = \frac{\sum_{s\in\mathcal{W}} (\tilde r_{i,s} - \bar{\tilde r}_i)(\tilde r_{j,s} - \bar{\tilde r}_j)}{\sqrt{\widehat{\text{Var}}[\tilde r_i]\,\widehat{\text{Var}}[\tilde r_j]}\; \cdot T} \tag{4}$$

with $\rho_{ij} \in [-1,1]$. The survey's standard conversion to a distance is

$$d_{ij} = \sqrt{2(1-\rho_{ij})} \tag{5}$$

giving $d_{ij}=0$ at $\rho=1$ and $d_{ij}=2$ at $\rho=-1$. This is a genuine metric: it is the Euclidean distance between the standardised return vectors on the unit sphere, which is why it satisfies the triangle inequality while $1-\rho$ does not.

**The sign decision.** Three options, and you must choose explicitly:

| Similarity | Edge means | Consequence |
|---|---|---|
| $S_{ij} = \rho_{ij}$ | co-movement, same direction | hedges are *far apart* |
| $S_{ij} = \lvert\rho_{ij}\rvert$ | co-movement, either direction | hedges are *neighbours* |
| $S_{ij} = -d_{ij}$ | equivalent to $\rho_{ij}$ | monotone in $\rho$ |

For clustering equities into economically coherent groups, use **signed $\rho$**. Two stocks at $\rho = -0.8$ are not peers; they are opposites, and grouping them destroys exactly the structure you want. This becomes important in Part B, because MI *forces* you into the absolute-value regime.

## A.2 The kNN construction

For each node $i$, let $\mathcal{N}_k(i)$ be the $k$ assets with the largest $S_{ij}$, $j \neq i$. The directed adjacency is

$$\vec{A}_{ij} = \begin{cases} 1 & j \in \mathcal{N}_k(i)\\ 0 & \text{otherwise}\end{cases}$$

**This is not symmetric.** $j$ may be among $i$'s top $k$ while $i$ is not among $j$'s — a hub asset appears in many neighbourhoods; a peripheral one in few. Since GNN message passing and spectral clustering both assume an undirected graph, you must symmetrise:

$$A^{\cup}_{ij} = \max(\vec A_{ij}, \vec A_{ji}) \qquad\text{(union / OR)}$$
$$A^{\cap}_{ij} = \min(\vec A_{ij}, \vec A_{ji}) \qquad\text{(intersection / AND)}$$

**Degree bounds.** Under union, $\deg(i) \ge k$ with mean degree in $[k, 2k)$; hubs exceed $k$. Under intersection, $\deg(i) \le k$ and isolated nodes are possible. Edge counts:

$$|E^{\cap}| \;\le\; \tfrac{1}{2}Nk \;\le\; |E^{\cup}| \;\le\; Nk$$

**Use the union.** Connectivity is a hard requirement — an isolated node receives no messages and its GNN embedding is meaningless. Intersection routinely disconnects peripheral assets.

**Weighted variant.** Retain the similarity on surviving edges:

$$W_{ij} = A_{ij}\cdot S_{ij}$$

Keep both: the binary version for topology diagnostics, the weighted version for the GNN.

## A.3 Why kNN rather than a threshold

A threshold graph sets $A_{ij} = \mathbb{1}\{\rho_{ij} > \tau\}$ (used with $\rho_{\text{thres}}$ throughout the survey). It has a structural flaw for this task: **degree is unbounded and correlation levels drift with the regime.** In a crisis, all correlations rise, so a fixed $\tau$ produces a near-complete graph in March 2020 and a sparse one in calm periods. Your graph then encodes the *volatility regime* rather than the *relationship structure*, and Phase 5 will find time-period clusters.

kNN is **rank-based**, therefore invariant to any monotone transformation of the similarity. Edge count is fixed at $\approx Nk$ by construction, so density is stable across regimes and dynamics reflect changes in *who* is connected rather than *how many*. That is precisely the property a temporal clustering study needs.

## A.4 Choosing $k$

Two bounds, one diagnostic.

**Lower — connectivity.** For an Erdős–Rényi-like graph, connectivity requires roughly $k \gtrsim \log N = \log 105 \approx 4.7$. Take $k \ge 5$ and verify empirically.

**Upper — sparsity.** Density is $\approx 2k/(N-1)$. At $k=10$, that is 19%; at $k=20$, 38%. Beyond ~20% the graph stops being informative — a GNN oversmooths and spectral clustering finds nothing.

**The diagnostic that decides it.** Edge persistence, the Jaccard overlap between consecutive edge sets:

$$J(t, t{+}1) = \frac{|E_t \cap E_{t+1}|}{|E_t \cup E_{t+1}|}$$

- $J \to 0$: the graph is noise; clusters will be unstable
- $J \to 1$: the graph is static; the "dynamic" premise collapses
- $J \approx 0.6\text{–}0.85$: structure that evolves

Sweep $k \in \{5,7,10,15,20\}$, plot $J$ and mean degree, choose from the plot. **Caveat to report honestly:** consecutive 252-day windows share ~92% of their observations, so high persistence is partly mechanical.

---

# Part B — Information-theoretic edges

## B.1 Shannon entropy

For a discrete random variable $X$ with alphabet $\mathcal{X}$:

$$H(X) = -\sum_{x\in\mathcal{X}} P(x)\log P(x) \tag{13}$$

**Intuition.** Expected surprise, in bits if $\log_2$, nats if $\ln$. Uniform over $b$ symbols gives the maximum $H = \log b$; a point mass gives $H=0$. Read it as "how many yes/no questions on average to pin down the value."

Joint entropy over a pair:

$$H(X,Y) = -\sum_{x}\sum_{y} P(x,y)\log P(x,y) \tag{14}$$

and over a triple, needed for Part B.5:

$$H(X,Y,Z) = -\sum_{x}\sum_{y}\sum_{z} P(x,y,z)\log P(x,y,z) \tag{17}$$

## B.2 Mutual information

$$\boxed{\;\text{MI}(\mathbf{r}_i,\mathbf{r}_j) = H(\mathbf{r}_i) + H(\mathbf{r}_j) - H(\mathbf{r}_i,\mathbf{r}_j)\;} \tag{12}$$

Equivalently, the Kullback–Leibler divergence between the joint and the product of marginals:

$$\text{MI}(X,Y) = \sum_{x,y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)} = D_{\text{KL}}\big(P(x,y)\,\|\,P(x)P(y)\big)$$

**Intuition.** How much knowing $\mathbf{r}_j$ reduces uncertainty about $\mathbf{r}_i$. The KL form is the sharper reading: MI measures *how far the joint distribution is from independence*.

**Properties:**

| Property | Statement | Why it matters here |
|---|---|---|
| Non-negativity | $\text{MI} \ge 0$ | no sign — see B.4 |
| Zero iff independent | $\text{MI}=0 \iff P(x,y)=P(x)P(y)$ | detects *any* dependence |
| Symmetry | $\text{MI}(X,Y)=\text{MI}(Y,X)$ | adjacency symmetric by construction |
| Invariance | unchanged under any invertible reparametrisation | robust to monotone transforms |
| Bound | $\text{MI}\le\min(H(X),H(Y))$ | needed for normalisation |

**The motivating claim.** Pearson correlation captures only linear dependence. If $Y = X^2$ with $X$ symmetric, then $\rho = 0$ while $\text{MI} > 0$. The survey's argument for MI in finance is that returns exhibit tail dependence and nonlinear co-movement that $\rho$ misses.

## B.3 The Gaussian benchmark — your sanity check

For a bivariate Gaussian pair with correlation $\rho$:

$$\text{MI} = -\tfrac{1}{2}\ln(1-\rho^2)$$

This is the single most useful identity in Part B. Consequences:

1. **It is a unit test.** Simulate Gaussian pairs at known $\rho$; your estimator must recover this curve. If it doesn't, the estimator is wrong before you ever run it on returns.
2. **It defines "what MI adds."** Compute $\text{MI}^{\text{Gauss}}_{ij} = -\frac12\ln(1-\rho_{ij}^2)$ from your correlation matrix. The excess

$$\Delta_{ij} = \widehat{\text{MI}}_{ij} - \text{MI}^{\text{Gauss}}_{ij}$$

   is the **nonlinear dependence not explained by correlation**. If $\Delta \approx 0$ across your universe, MI is telling you nothing new and the honest finding is that the correlation graph suffices.
3. **It is monotone in $|\rho|$**, and depends on $\rho^2$ — which is the formal statement of the sign problem below.

## B.4 The sign problem

$\text{MI}$ depends on $\rho^2$, so

$$\rho = +0.9 \quad\text{and}\quad \rho = -0.9 \quad\Longrightarrow\quad \text{identical MI} = 0.83\ \text{nats}$$

An MI-kNN graph therefore connects an asset to its strongest *hedges* as readily as to its closest peers. For a clustering task whose output should be "groups of economically similar assets," that is a substantive defect, not a technicality.

Two responses:

- **Accept it**, and interpret clusters as "assets sharing a dependence structure" regardless of direction.
- **Restore the sign** by masking with correlation: $S^{\text{signed}}_{ij} = \text{sign}(\rho_{ij})\cdot\widehat{\text{MI}}_{ij}$, then take positives only.

The second is a hybrid and slightly inelegant, but it preserves the property that made correlation work. Whichever you pick, state it — it is one of the more interesting design decisions in the project.

## B.5 The distance metric

$$d^{\text{MI}}_{ij} = H(\mathbf{r}_i) + H(\mathbf{r}_j) - 2\,\text{MI}(\mathbf{r}_i,\mathbf{r}_j) \tag{15}$$

Equivalently $d^{\text{MI}} = H(X,Y) - \text{MI}(X,Y)$, the **variation of information**. This is a true metric: non-negative, symmetric, zero on the diagonal, and it satisfies the triangle inequality — properties the survey notes explicitly and which the raw MI does not possess.

**Normalised variants**, preferable when entropies differ across assets:

$$\text{NMI}_{ij} = \frac{\text{MI}_{ij}}{\sqrt{H_i H_j}} \in [0,1], \qquad \text{or}\qquad \frac{2\,\text{MI}_{ij}}{H_i + H_j}$$

Use a normalised form for kNN. Raw MI scales with marginal entropy, so a high-entropy (volatile) asset shows large MI with everything — meaning your neighbour rankings would partly encode volatility, which is already a node feature.

## B.6 Estimation — where this actually gets hard

Returns are **continuous**; equations (13)–(14) are for **discrete** variables. The estimator is the entire practical problem, and the survey does not address it.

### Plug-in (binned) estimator

Discretise each return series into $b$ bins, count, and substitute empirical frequencies:

$$\hat{P}(x) = \frac{n_x}{T}, \qquad \widehat{\text{MI}} = \sum_{x,y}\frac{n_{xy}}{T}\log\frac{n_{xy}T}{n_x n_y}$$

**The sample-size problem.** With $T = 252$ and $b$ bins, the joint table has $b^2$ cells:

| $b$ | joint cells | samples/cell |
|---|---|---|
| 5 | 25 | 10.1 |
| 8 | 64 | 3.9 |
| 10 | 100 | 2.5 |
| 16 | 256 | **0.98** |

Below roughly 5 samples per cell the estimate is dominated by noise. This is the MI counterpart of $q = N/T$ for correlation — a hard constraint from your window length, not a tuning preference.

**The bias is systematic and upward.** The Miller–Madow correction:

$$\widehat{\text{MI}}_{\text{MM}} = \widehat{\text{MI}}_{\text{plug-in}} - \frac{(b_x-1)(b_y-1)}{2T}$$

Sampling noise makes finite data look dependent even when it is not, so $\widehat{\text{MI}} > 0$ for *independent* series. Since MI is non-negative, the noise has nowhere to cancel — unlike correlation, whose errors are signed and average out. **This is why an MI graph needs a null-model check that a correlation graph does not:** shuffle one series, recompute, and take the resulting value as your noise floor.

**Binning scheme.** Use quantile (equal-frequency) bins, not equal-width. Returns are heavy-tailed, so equal-width bins put nearly everything in the centre and leave the tails almost empty — the opposite of what you want. Quantile binning guarantees $T/b$ observations per marginal bin by construction. Take $b \in \{5,6,7\}$ at $T=252$.

### KSG estimator (recommended)

Kraskov–Stögbauer–Grassberger avoids binning entirely, using $k$-nearest-neighbour distances in the joint space:

$$\widehat{\text{MI}}_{\text{KSG}} = \psi(\kappa) - \big\langle \psi(n_x+1) + \psi(n_y+1)\big\rangle + \psi(T)$$

where $\psi$ is the digamma function, $\kappa$ the neighbour count (typically 3–5), and $n_x, n_y$ count points within the marginal projections of the joint $\kappa$-neighbour ball. Available as `sklearn.feature_selection.mutual_info_regression`.

**Why it is better here:** it adapts resolution to local density — fine where data is dense, coarse in the tails — so it is far less biased than binning at $T = 252$. The cost is $O(T\log T)$ per pair and no free parameter beyond $\kappa$.

**Cost warning.** $\binom{105}{2} = 5{,}460$ pairs per date $\times$ 128 dates $= 699{,}000$ MI estimates. Correlation is a single matrix multiply; MI is not. Budget for it, vectorise where possible, and cache per date.


---

# Part C — The comparison

## C.1 What differs

| | Pearson kNN | MI kNN |
|---|---|---|
| Captures | linear dependence | any dependence |
| Sign | signed | unsigned ($\rho^2$) |
| Estimation | closed form, one matmul | binning or KSG, ~$10^5$–$10^6$ estimates |
| Bias at $T{=}252$ | eigenvalue spread, MP-bounded | upward, one-sided |
| Shrinkage available | yes (Ledoit–Wolf) | no standard analogue |
| Null value under independence | $\approx 0$, signed | $> 0$, needs a shuffle test |

## C.2 The experiment

Hold topology fixed (kNN, same $k$, union symmetrisation), vary only the similarity. Then all differences are attributable to stage 2. Three things to measure:

1. **Edge overlap.** Jaccard between the correlation-kNN and MI-kNN edge sets at each date. High overlap ⇒ MI is recovering the correlation graph and the added cost buys nothing.
2. **Nonlinear excess.** The distribution of $\Delta_{ij} = \widehat{\text{MI}}_{ij} + \frac12\ln(1-\rho_{ij}^2)$. This directly quantifies what MI adds beyond correlation, in nats.
3. **Persistence.** Compare $J(t,t+1)$ for both. MI's one-sided noise may well make it *less* stable.


## C.3 Parameters for the setup

| Parameter | Value | Reason |
|---|---|---|
| $T$ | 252 | $q = 0.42$; MI needs every observation available |
| $k$ | sweep $\{5,7,10,15,20\}$ | connectivity floor $\log N \approx 4.7$; density ceiling ~20% |
| symmetrisation | union | intersection disconnects peripheral nodes |
| MI estimator | KSG, $\kappa=4$ | binning is untenable at $T=252$ |
| MI normalisation | $\text{MI}/\sqrt{H_iH_j}$ | prevents volatility leaking into neighbour rank |
| null model | shuffle, 100 permutations | establishes the MI noise floor |
| factor removal | 1 (market) | mandatory for MI — no sign cancellation |

---

