# Phase 2 — Mathematical Reference

Node features for dynamic equity clustering. Every quantity below is a **marginal** property of a single asset, computed on a trailing window.

---

## 1. Notation and the two clocks

$N = 105$ assets indexed $i$; daily sessions indexed $s$; rebalance dates indexed $t$.

$P_{i,s}$ is the adjusted close, $V_{i,s}$ the share volume, and

$$D_{i,s} = P_{i,s} V_{i,s}$$

the **dollar volume**. Daily log return:

$$r_{i,s} = \log P_{i,s} - \log P_{i,s-1}$$

A trailing window of length $W$ ending at $s$ is

$$\mathcal{W}(s, W) = \{s-W+1,\; \dots,\; s\}$$

**The invariant.** A feature is a map $f$ from a trailing window to a scalar, and

$$X_{t,i,f} = f\big(\{r_{i,s}, D_{i,s}\}_{s \in \mathcal{W}(t,W)}\big)$$

depends only on data in $(-\infty, t]$. The window is right-aligned — this is the whole of the leakage discipline, stated in one line.

The output is a tensor $X \in \mathbb{R}^{T \times N \times F}$ with $F = 15$.

---

## 2. Risk features

### 2.1 Realised volatility

$$\hat{\sigma}_{i,t}^{(W)} = \sqrt{\frac{252}{W-1} \sum_{s \in \mathcal{W}(t,W)} \left(r_{i,s} - \bar{r}_{i,t}\right)^2}$$

**Intuition.** The RMS of returns over the window — a short-time energy estimate of the return signal. $W$ is a bias–variance dial: short windows track regime change but are noisy; long windows are stable but smear across transitions. Exactly the STFT window-length tradeoff. Including both $W=21$ and $W=63$ is a crude two-scale decomposition.

**On the $\sqrt{252}$.** Under i.i.d. returns, variance is additive in time, so $\text{Var}(252\text{ days}) = 252\,\text{Var}(1\text{ day})$ and standard deviation scales as $\sqrt{252}$. Real returns exhibit volatility clustering and are *not* i.i.d., so this is a **reporting convention**, not a fact about the data. It cancels in any cross-sectional comparison — which is all we use it for.

### 2.2 Semi-volatility

$$\hat{\sigma}^{-}_{i,t} = \sqrt{252} \cdot \text{std}\big(\{r_{i,s} : s \in \mathcal{W}(t,63),\; r_{i,s} < 0\}\big)$$

**Intuition.** Volatility computed on losses only. Ordinary volatility treats a $+5\%$ day and a $-5\%$ day identically; investors do not. For a symmetric distribution $\hat{\sigma}^{-} \approx \hat{\sigma}$, so the *ratio* $\hat{\sigma}^{-}/\hat{\sigma}$ is a pure asymmetry measure. Note the denominator is the count of negative days, which varies — this is a conditional standard deviation, not a masked-and-rescaled one.

### 2.3 Volatility of volatility

$$\text{volvol}_{i,t} = \text{std}\big(\{\hat{\sigma}^{(21)}_{i,s}\}_{s \in \mathcal{W}(t,63)}\big)$$

**Intuition.** How unstable is the risk level itself? Two assets can share an average volatility of 25% while one sits steadily at 25% and the other alternates between 10% and 40%. A second-order statistic — the "acceleration" of risk.

### 2.4 Skewness and excess kurtosis

With central moments $m_k = \frac{1}{W}\sum (r_{i,s} - \bar{r})^k$:

$$\text{skew} = \frac{m_3}{m_2^{3/2}}, \qquad \text{kurt}_{\text{excess}} = \frac{m_4}{m_2^{2}} - 3$$

**Intuition.** Skew measures asymmetry: negative means the left tail is longer, the crash-prone profile typical of equities. Excess kurtosis measures tail weight relative to a Gaussian; the $-3$ makes the Gaussian the zero point. Both need long windows ($W = 252$) because higher moments are estimated from the tails, where observations are by definition scarce — a 63-day window has perhaps three observations doing all the work in $m_4$.

---

## 3. Liquidity features

### 3.1 Log average dollar volume

$$\text{ladv}_{i,t} = \log\left(\frac{1}{W}\sum_{s \in \mathcal{W}(t,63)} D_{i,s}\right)$$

**Why the log is mandatory.** Dollar volume spans several orders of magnitude across a large-cap universe. Untransformed, the cross-sectional distribution is dominated by a handful of names, and after z-scoring almost every asset sits within a hair of zero while two sit at $+8$. That is an outlier detector, not a feature. The log makes the distribution roughly symmetric and makes *ratios* the unit of comparison — a name twice as liquid sits a constant distance away regardless of absolute size.

### 3.2 Amihud illiquidity

$$\text{ILLIQ}_{i,t} = \log\left(\frac{1}{W}\sum_{s \in \mathcal{W}(t,63)} \frac{|r_{i,s}|}{D_{i,s}}\right)$$

**Intuition.** The inner ratio has units of *return per dollar traded* — how far the price moves per unit of volume. It is a direct price-impact estimate: a liquid name absorbs large flow with little movement (small ratio); an illiquid one lurches (large ratio). Higher means *less* liquid.

The log is again mandatory: the raw measure is severely right-skewed, being a ratio with a small, positive, right-skewed denominator.

**The signal-processing reading.** $|r|$ is the response, $D$ the excitation. The ratio is an inverse gain — a measure of how much the market absorbs before the price responds.

### 3.3 Zero-return fraction

$$z_{i,t} = \frac{1}{W}\sum_{s \in \mathcal{W}(t,63)} \mathbb{1}\{|r_{i,s}| < \epsilon\}, \qquad \epsilon = 10^{-8}$$

**Intuition.** A price that does not move often did not trade meaningfully. Crude, free, and empirically a good proxy for the bid–ask spread — it is the basis of the Lesmond–Ogden–Trzcinka measure. On a large-cap universe this should be near zero for almost everything; names where it is not are the ones to look at.

---

## 4. Factor features

Define the equal-weighted market return:

$$r_{m,s} = \frac{1}{N}\sum_{i=1}^{N} r_{i,s}$$

### 4.1 Beta

$$\beta_{i,t} = \frac{\widehat{\text{Cov}}\big(r_i, r_m\big)_{\mathcal{W}(t,252)}}{\widehat{\text{Var}}\big(r_m\big)_{\mathcal{W}(t,252)}}$$

**Intuition.** The OLS slope of the asset on the market — the sensitivity of $r_i$ to a unit move in $r_m$. $\beta > 1$ amplifies the market; $\beta < 1$ damps it.

**Why cov/var and not a regression loop.** They are algebraically identical: the OLS slope of $y$ on $x$ *is* $\text{Cov}(x,y)/\text{Var}(x)$. But `rolling().cov()` is vectorised across all 105 columns at once, while a loop fits $105 \times 2929$ separate regressions. Roughly two orders of magnitude in runtime for the same number.

**Connection to Phase 3.** $r_m$ is a proxy for the first principal component of the return covariance matrix — the market mode, the common-mode component that connects every asset to every other. In Phase 3 you will remove it before building the graph, for exactly the reason it is useful here: it is the single largest source of shared variation, and it drowns out everything else.

### 4.2 Idiosyncratic volatility

$$\varepsilon_{i,s} = r_{i,s} - \beta_{i,t}\, r_{m,s}, \qquad \text{idio}_{i,t} = \sqrt{252}\cdot\text{std}\big(\{\varepsilon_{i,s}\}_{\mathcal{W}(t,252)}\big)$$

**Intuition.** Volatility remaining after the market is projected out — the asset's own noise. This decomposes total risk:

$$\sigma_i^2 \approx \beta_i^2 \sigma_m^2 + \sigma_{\varepsilon,i}^2$$

systematic plus idiosyncratic. Two assets with identical total volatility can be entirely different animals: one a high-beta index proxy, the other a low-beta name with company-specific turbulence.

**One approximation worth naming.** The standard market model is $r_i = \alpha_i + \beta_i r_m + \varepsilon_i$. We omit $\alpha$, so the residual absorbs any mean drift difference. Since daily $\alpha$ is on the order of $10^{-4}$ while daily $\sigma$ is around $10^{-2}$, the effect on the standard deviation is negligible — but it is an approximation, not an identity.

---

## 5. Trend features

### 5.1 Momentum

$$M^{(W)}_{i,t} = \sum_{s \in \mathcal{W}(t,W)} r_{i,s} = \log\frac{P_{i,t}}{P_{i,t-W}}$$

The **telescoping identity**: summing log returns gives the log cumulative return directly, with no compounding arithmetic. This is the property that makes log returns the right unit for anything time-additive.

### 5.2 Skip-month momentum

$$M^{(252,21)}_{i,t} = \sum_{s=t-251}^{t-21} r_{i,s} = M^{(252)}_{i,t} - M^{(21)}_{i,t}$$

spanning $252 - 21 = 231$ days.

**Why skip the recent month.** Short-horizon returns tend to *reverse* rather than continue — from bid–ask bounce, liquidity provision, and one-month reversal effects. Including the most recent month contaminates a momentum signal with a reversal signal of opposite sign, partially cancelling it. Excluding it is standard practice in the literature (Jegadeesh–Titman).

This is also the identity your unit test exploits: on a series with constant daily return $r$, the feature equals exactly $231r$. Any off-by-one in the window boundaries fails immediately.

### 5.3 Distance from 52-week high

$$d_{i,t} = \log\frac{P_{i,t}}{\max_{s \in \mathcal{W}(t,252)} P_{i,s}} \;\le\; 0$$

**Intuition.** Zero exactly when the asset makes a new one-year high; increasingly negative as it sits further below. A drawdown measure, and empirically a distinct signal from momentum — an asset can have strong 12-month momentum while being 20% off its high, and the two describe different situations.

---

## 6. Cross-sectional standardisation

Applied per feature, per date, **across tickers** — never across time.

### 6.1 Median absolute deviation

$$\text{MAD}_t = \text{median}_i\big(|x_{i,t} - \text{median}_j(x_{j,t})|\big)$$

For Gaussian data, $\text{MAD} \approx 0.6745\,\sigma$, so the **consistency constant**

$$1.4826 = \frac{1}{\Phi^{-1}(0.75)} \approx \frac{1}{0.6745}$$

rescales MAD onto the same footing as a standard deviation. This is why $\text{MAD}\times 1.4826$ can be substituted for $\sigma$ without changing the interpretation of a "3-sigma" threshold.

### 6.2 The robust z-score

$$z_{i,t} = \text{clip}\left(\frac{x_{i,t} - \text{median}_j(x_{j,t})}{1.4826\,\text{MAD}_t},\; -c,\; +c\right), \qquad c = 3$$

**Why median/MAD rather than mean/std.** The **breakdown point** — the fraction of arbitrarily corrupted observations a statistic tolerates before becoming unbounded — is $0\%$ for the mean and standard deviation, and $50\%$ for the median and MAD. Concretely: a single extreme observation inflates $\hat{\sigma}$, so the threshold $3\hat{\sigma}$ expands to accommodate the very outlier it was meant to clip. The outlier survives, and every other observation is compressed toward zero. With MAD, one outlier moves the scale essentially not at all, and the clip does its job.

**Why standardise cross-sectionally at all.** In March 2020 every volatility rose together. Raw values would encode "it was March 2020" as the dominant axis of variation, and your clusters would become time-period clusters. Standardising within each date removes the common level and leaves *relative position in the cross-section* — which is the thing that is comparable across regimes. Same reasoning as the cross-sectional z-scores in the momentum project.

---

## 7. Feature redundancy

After assembly, examine

$$C = \text{Corr}\big(\tilde{X}\big), \qquad \tilde{X} \in \mathbb{R}^{TN \times F}$$

the $F\times F$ correlation matrix pooled over dates and assets.

**Why this matters more here than in regression.** A clustering algorithm sees only *distance in feature space*:

$$d(i,j)^2 = \sum_{f=1}^{F} (x_{i,f} - x_{j,f})^2$$

Every feature enters with equal weight. If three features are near-copies of each other, that concept enters with weight $3$ — silently. Ridge or Lasso would handle collinearity by shrinking coefficients; $k$-means has no such mechanism. Redundancy is therefore an *implicit weighting decision*, and if $\text{corr}(\text{vol}_{21}, \text{vol}_{63}) \approx 0.95$ your clusters are volatility clusters whether or not you intended that.

---

## 8. Summary table

| Feature | Window | Definition | Measures |
|---|---|---|---|
| `vol_21`, `vol_63` | 21, 63 | $\sqrt{252}\,\text{std}(r)$ | risk level, two scales |
| `semivol_63` | 63 | $\sqrt{252}\,\text{std}(r \mid r<0)$ | downside risk |
| `volofvol_63` | 63 | $\text{std}(\hat\sigma^{(21)})$ | risk instability |
| `skew_252` | 252 | $m_3/m_2^{3/2}$ | asymmetry |
| `kurt_252` | 252 | $m_4/m_2^2 - 3$ | tail weight |
| `log_adv_63` | 63 | $\log \overline{D}$ | trading size |
| `amihud_63` | 63 | $\log \overline{|r|/D}$ | price impact |
| `zero_ret_frac_63` | 63 | $\overline{\mathbb{1}\{r \approx 0\}}$ | non-trading |
| `beta_252` | 252 | $\text{Cov}(r_i,r_m)/\text{Var}(r_m)$ | market sensitivity |
| `idio_vol_252` | 252 | $\sqrt{252}\,\text{std}(r_i - \beta r_m)$ | own risk |
| `mom_21`, `mom_63` | 21, 63 | $\sum r$ | short trend |
| `mom_252_21` | 231 | $M^{(252)} - M^{(21)}$ | long trend, reversal-stripped |
| `dist_52w_high` | 252 | $\log(P/\max P)$ | drawdown |