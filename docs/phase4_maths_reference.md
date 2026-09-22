# Phase 4 — Mathematical Reference

## Graph convolution, self-supervised objectives, and walk-forward training

---

## 0. What Phase 4 computes

A parameterised map from features and structure to embeddings:

$$f_\theta: \big(X[t],\, A[t]\big) \longmapsto Z[t], \qquad X[t]\in\mathbb{R}^{105\times 13},\;\; A[t]\in\mathbb{R}^{105\times105},\;\; Z[t]\in\mathbb{R}^{105\times 16}$$

Phase 5 clusters $Z$. Phase 6 asks whether those clusters beat clustering $X$ alone — which is the whole reason node and edge information were kept separate through Phases 2 and 3.

**The core claim a GNN makes:** an asset is described not only by its own features but by the features of the assets it co-moves with. Your Phase 3c diagnostics justify this — normalised Dirichlet energy of 0.6–0.8 says connected nodes really are closer in feature space than random pairs, so mixing neighbours concentrates signal rather than blurring it.

---

## 1. The graph convolution layer

### 1.1 The operation

$$\boxed{\;H^{(\ell+1)} = \sigma\!\left(\hat{A}\,H^{(\ell)}\,W^{(\ell)}\right), \qquad \hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\;}$$

with $\tilde{A} = A + I$, $\tilde{D} = \operatorname{diag}(\tilde{A}\mathbf{1})$, and $H^{(0)} = X$.

Read right to left, three separate things happen:

| Term | Shape | What it does |
|---|---|---|
| $H W$ | $N\times d_{\text{out}}$ | projects each node's features independently — an ordinary linear layer |
| $\hat{A}\,(\cdot)$ | $N\times d_{\text{out}}$ | mixes each node with its neighbours |
| $\sigma(\cdot)$ | — | nonlinearity |

Only the middle term uses the graph. Set $\hat A = I$ and you have an MLP — which is exactly your B1 baseline, and why the comparison is clean.

### 1.2 Why the self-loop

Without $+I$, row $i$ of $\hat A H$ contains no contribution from node $i$ itself. An asset's embedding would be built purely from its neighbours, discarding its own volatility, liquidity and beta. The self-loop keeps the node in its own neighbourhood.

### 1.3 Why $\tilde D^{-1/2}(\cdot)\tilde D^{-1/2}$ and not $\tilde D^{-1}$

Both normalise for degree. The difference is symmetry.

Row normalisation $\tilde D^{-1}\tilde A$ makes each row sum to 1 — a genuine average over neighbours — but the result is **not symmetric**, so its eigenvalues may be complex and repeated application has no spectral guarantee.

The symmetric split scales edge $(i,j)$ by $1/\sqrt{\tilde d_i \tilde d_j}$, which keeps $\hat A$ symmetric. Then $\hat A = I - \tilde L_{\text{sym}}$ where $\tilde L_{\text{sym}}$ is the normalised Laplacian, whose spectrum lies in $[0,2)$. Therefore

$$\lambda(\hat A) \in (-1,\, 1]$$

Bounded spectrum means stacking layers cannot blow up activations. That bound is the entire justification.

**Interpretation of the asymmetric scaling.** A message from a high-degree neighbour is damped by its $1/\sqrt{d_j}$: a hub connected to forty assets contributes less per edge than a peripheral node connected to eleven. That is the desired behaviour — a hub's signal is diffuse, a specialist's is specific.

---

## 2. Where $\hat A$ comes from: spectral graph theory

This section is the signal-processing derivation. It is not needed to write the code, but it is what makes the layer explicable rather than magic.

### 2.1 The graph Fourier transform

The combinatorial Laplacian is $L = D - A$; normalised, $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$. It is symmetric positive semi-definite, so

$$L = U\Lambda U^{\top}, \qquad \Lambda = \operatorname{diag}(\lambda_1 \le \dots \le \lambda_N)$$

The eigenvectors $U$ are the **graph Fourier basis**, and the eigenvalues are **graph frequencies**. The analogy is exact: on a ring graph, $U$ *is* the DFT matrix and $\lambda$ is $\omega^2$.

The frequency reading follows from the Dirichlet energy:

$$u^{\top}L\,u = \tfrac12\sum_{(i,j)\in E} A_{ij}\,(u_i - u_j)^2$$

A low-frequency eigenvector varies slowly across edges — connected nodes hold similar values. A high-frequency one oscillates between neighbours. $\lambda_1 = 0$ always, with a constant eigenvector: the graph's DC component.

Graph Fourier transform and its inverse:

$$\hat{x} = U^{\top}x, \qquad x = U\hat{x}$$

### 2.2 Spectral filtering

A filter with frequency response $g_\theta$ acts as

$$g_\theta \star x = U\,g_\theta(\Lambda)\,U^{\top}x$$

— transform, multiply, transform back. Exactly the FFT-multiply-IFFT pattern.

**Two problems.** The eigendecomposition costs $O(N^3)$, and $U$ is dense so each application costs $O(N^2)$. Worse, a filter defined by arbitrary $g_\theta(\Lambda)$ is not localised — one application can move information across the whole graph.

### 2.3 Chebyshev approximation

Approximate $g_\theta$ by a degree-$K$ polynomial in $\Lambda$. Because $L^k$ has support only on $k$-hop neighbourhoods, a degree-$K$ polynomial filter is **exactly $K$-hop localised** and requires no eigendecomposition — only repeated sparse multiplication by $L$.

$$g_\theta \star x \approx \sum_{k=0}^{K}\theta_k\,T_k(\tilde\Lambda)\,x$$

with $T_k$ the Chebyshev polynomials.

### 2.4 The first-order approximation

Kipf & Welling take $K=1$ and $\lambda_{\max}\approx 2$:

$$g_\theta \star x \approx \theta_0 x + \theta_1\big(L_{\text{sym}} - I\big)x = \theta_0 x - \theta_1 D^{-1/2}AD^{-1/2}x$$

Constraining $\theta = \theta_0 = -\theta_1$ to halve the parameters:

$$g_\theta \star x \approx \theta\big(I + D^{-1/2}AD^{-1/2}\big)x$$

**The renormalisation trick.** The operator $I + D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0,2]$, and repeated application amplifies. Replacing it with $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ — folding the self-loop *inside* the normalisation — pulls the spectrum back to $(-1,1]$. That substitution is where the final formula comes from.

Depth then substitutes for filter order: a single layer is a 1-hop filter, and $\ell$ stacked layers give an $\ell$-hop receptive field.

### 2.5 The GCN is a low-pass filter

Since $\hat A = I - \tilde L_{\text{sym}}$, the frequency response is

$$h(\lambda) = 1 - \lambda, \qquad \lambda \in [0,2)$$

At $\lambda = 0$ (DC, constant across the graph) the gain is 1. At $\lambda \to 2$ (maximally oscillatory between neighbours) the gain approaches $-1$, and near $\lambda = 1$ it is zero.

**So a GCN layer is a low-pass graph filter.** It preserves smooth signals and attenuates ones that alternate between neighbours. Your Dirichlet-energy result is the statement that your node features are mostly low-frequency on this graph — which is precisely the condition under which a low-pass filter helps.

---

## 3. Oversmoothing

### 3.1 The statement

Stacking $\ell$ layers without nonlinearity applies $\hat A^{\ell}$. In the spectral domain that is $h(\lambda)^{\ell} = (1-\lambda)^{\ell}$, which for every $\lambda \neq 0$ tends to zero. Only the DC component survives.

For a connected graph the top eigenvector of $\hat A$ is $\tilde D^{1/2}\mathbf{1}$ with eigenvalue exactly 1. Hence

$$\hat A^{\ell}H \;\longrightarrow\; c\,\tilde D^{1/2}\mathbf{1}\,v^{\top}$$

Every row converges to the same vector up to a degree factor. All 105 assets receive the same embedding, and clustering becomes meaningless.

**This is repeated low-pass filtering.** Cascade enough identical low-pass stages and everything but DC is gone.

### 3.2 The consequence

Two layers gives a 2-hop receptive field. On a graph with mean degree ~11, a 2-hop neighbourhood already reaches a substantial fraction of 105 nodes. Depth is not the constraint here; oversmoothing is.

### 3.3 Diagnostics to log every run

**Embedding variance.** $\frac{1}{Nd}\sum_{i,f}(Z_{if} - \bar Z_f)^2$. Collapse toward zero is oversmoothing.

**Effective rank.** With singular values $\sigma_1 \ge \dots \ge \sigma_d$ of $Z$ and $p_k = \sigma_k / \sum_j \sigma_j$:

$$\operatorname{erank}(Z) = \exp\!\left(-\sum_k p_k \log p_k\right)$$

The exponential of the spectral entropy — a continuous count of how many dimensions are genuinely in use. If $d = 16$ but $\operatorname{erank} \approx 2$, fourteen dimensions are decorative.

**Dirichlet energy of the output.** $\operatorname{tr}(Z^{\top}\tilde L_{\text{sym}}Z)$, tracked across layers. Monotone decay toward zero is the direct signature.

---

## 4. Training objectives

Clustering is not a loss. You need a self-supervised signal.

### 4.1 Graph autoencoder (GAE)

Encode, then reconstruct the adjacency from embedding inner products:

$$Z = f_\theta(X, \hat A), \qquad \hat{A}^{\text{rec}}_{ij} = \sigma\!\big(z_i^{\top}z_j\big)$$

$$\mathcal{L}_{\text{GAE}} = -\frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}}\log\sigma(z_i^{\top}z_j) \;-\; \frac{1}{|\mathcal{E}^-|}\sum_{(i,j)\in\mathcal{E}^-}\log\big(1-\sigma(z_i^{\top}z_j)\big)$$

**Intuition.** Connected assets should have embeddings pointing the same way; unconnected ones should not. The inner product is an unnormalised cosine similarity.

**Class imbalance.** At $k=10$ your graph has roughly 1,100 edges out of 5,460 possible pairs, so negatives outnumber positives by about four to one. Either sample $|\mathcal{E}^-| = |\mathcal{E}|$ negatives per epoch, or weight the positive term by $|\mathcal{E}^-|/|\mathcal{E}|$. Without one of these the model learns to predict "no edge" everywhere.

**The known weakness.** GAE optimises exactly the structure that was handed to it, so the embedding can become a lossy re-encoding of $A$ rather than a synthesis of $A$ and $X$. It is the right first objective because it is simple and debuggable, not because it is the best.

### 4.2 Deep Graph Infomax (DGI)

Corrupt the features by shuffling rows — same feature distribution, wrong graph positions:

$$\tilde{X} = P X \quad\text{for a random permutation } P$$

Encode both, form a global summary, and train a discriminator to tell real patches from corrupted ones:

$$Z = f_\theta(X,\hat A),\quad \tilde Z = f_\theta(\tilde X,\hat A),\quad s = \sigma\!\left(\tfrac{1}{N}\sum_i z_i\right)$$

$$\mathcal{D}(z, s) = \sigma\!\big(z^{\top}Ws\big)$$

$$\mathcal{L}_{\text{DGI}} = -\frac{1}{2N}\left[\sum_i \log \mathcal{D}(z_i,s) + \sum_i \log\big(1 - \mathcal{D}(\tilde z_i, s)\big)\right]$$

**Intuition.** For a node embedding to be distinguishable from a corrupted one, it must encode information about *where in the graph the node sits*, not merely what its features are. Shuffling preserves the marginal feature distribution exactly, so the only usable signal is feature–position correspondence.

**The connection back to Phase 3.** DGI maximises a Jensen–Shannon bound on the mutual information between local patch representations and the global summary — hence "infomax". Note what changed: in Phase 3 you had to *estimate* MI, which forced you into KSG, permutation nulls and one-sided bias. Here you only need to *maximise* a lower bound, and the bound is optimised by a discriminator. The estimation problem disappears entirely because a bound that moves in the right direction is enough.

Empirically DGI produces better-separated embeddings than GAE, because it is not tied to reconstructing a specific adjacency.

### 4.3 Auxiliary volatility head

Add a linear head predicting next-period realised volatility:

$$\hat{y}_i = w^{\top}z_i + b, \qquad \mathcal{L}_{\text{vol}} = \frac{1}{N}\sum_i \big(\hat y_i - y_i^{(t+1)}\big)^2$$

$$\mathcal{L} = \mathcal{L}_{\text{ssl}} + \lambda\,\mathcal{L}_{\text{vol}}$$

with $y^{(t+1)}$ the cross-sectionally standardised realised volatility over $(t, t+1]$.

**Why volatility and not returns.** Volatility is persistent and predictable; monthly returns are close to unpredictable. Using returns would fit noise and let you claim alpha you have not found. Volatility gives a genuine learnable signal, and it produces a hard out-of-sample number — rank IC or $R^2$ — which is the honest quantitative anchor for the whole project.

**The leakage rule.** $y^{(t+1)}$ is realised *after* $t$. It may appear only in the training loss, never in the encoder input. This is the one place in the project where future data legitimately enters, and it must be confined to the loss term.

---

## 5. Walk-forward training

### 5.1 The protocol

$$\underbrace{[t-W+1,\;t]}_{\text{train}} \;\;\underbrace{(t,\;t+g]}_{\text{embargo}} \;\;\underbrace{(t+g,\;t+g+R]}_{\text{embed}}$$

Fit on $W$ rebalance dates, hold out $g$ months, then embed the next $R$ months with frozen weights. Advance and repeat.

**Why refit at all.** A single fit over all 128 dates leaks: the encoder would have seen 2026 while embedding 2017. And unlike a supervised task, nothing warns you — the loss looks fine and the clusters look coherent.

**Why the embargo.** Your features use trailing windows up to 252 days and your graphs use 252-day correlation windows. Two adjacent rebalance dates share about 92% of their underlying data. Without a gap, train and embed windows overlap through the features themselves even when the dates do not.

### 5.2 The cost

With $W = 36$ and $R = 12$, you get roughly 8 refits over 128 dates. Each is a small model on a 105-node graph — seconds on CPU. Cheap enough that the pooled shortcut has no justification.

---

## 6. Implementation details worth knowing

### 6.1 Initialisation

Glorot: $W \sim \mathcal{U}\big(-a, a\big)$ with $a = \sqrt{6/(d_{\text{in}} + d_{\text{out}})}$, giving $\operatorname{Var}(W) = 2/(d_{\text{in}}+d_{\text{out}})$. This keeps activation variance roughly constant across layers. Since $\hat A$ already has spectrum bounded by 1, it does not compound the scale.

### 6.2 Permutation equivariance

For any permutation matrix $P$:

$$f_\theta(PX, PAP^{\top}) = P\,f_\theta(X, A)$$

Relabelling assets permutes the embeddings identically — the model has no notion of node order. **This is a testable property and belongs in your unit tests:** permute the inputs, run the layer, and check the output matches the permuted original.

### 6.3 Dropout

Standard dropout on the hidden features. DropEdge — randomly removing edges during training — is a graph-specific regulariser that also mitigates oversmoothing, but is unnecessary at two layers.

### 6.4 Dense beats sparse here

At $N = 105$, $\hat A H$ is a $105\times105$ by $105\times64$ dense matmul: about 700k FLOPs, microseconds. Sparse formats and message-passing frameworks pay off in the thousands of nodes. Dense `torch` is faster *and* simpler to debug.

---

## 7. Evaluating the embeddings

Before Phase 5, four checks:

**Not collapsed.** Effective rank comfortably above 1; embedding variance not decaying toward zero.

**Better than the features alone.** Compare $Z$ against raw $X$ under the same downstream clusterer. If $Z$ is not better, the graph contributed nothing and that is the finding.

**Stable over time.** Embeddings at $t$ and $t+1$ should be similar up to rotation — the encoder is refit periodically, so absolute coordinates are not comparable across refits, only relative geometry. Compare pairwise distance matrices, not raw vectors.

**Structured.** Do embedding neighbours recover sector? Compute the same adjusted homophily from Phase 3c on a kNN graph built in $Z$-space.

---

## Checkpoint questions

- Why $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ rather than $\tilde D^{-1}\tilde A$?
- What is the frequency response of a GCN layer, and why does that make oversmoothing inevitable with depth?
- Your effective rank is 2 with $d = 16$. What has gone wrong?
- Why does DGI shuffle features rather than perturb the adjacency?
- DGI maximises mutual information. Why does it avoid every estimation problem you hit in Phase 3?
- Why is next-month volatility a defensible auxiliary target when next-month return is not?
- Two adjacent rebalance dates share 92% of their data. What does that imply for the embargo?