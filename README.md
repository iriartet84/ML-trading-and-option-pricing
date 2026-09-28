# Machine Learning for Trading and Option Pricing

## Overview

This project applies machine learning to two financial-market problems:

1. **News sentiment-driven trading strategies** for HSBC and JPMorgan.
2. **Machine-learning surrogate models for option pricing**, using Gaussian Process Regression (GPR) and neural networks to approximate computationally expensive pricing engines.

The project combines financial data, sentiment analysis, quantitative trading, derivatives pricing, Monte Carlo simulation, and machine learning. The common objective is to investigate whether statistical learning can either extract information from financial text or substantially reduce the computational cost of numerical pricing.

## Part I — Sentiment-Driven Trading

### Research Question

> **Can news sentiment indicators generate profitable trading signals after transaction costs?**

The first part develops and backtests trading strategies based on daily news sentiment for **HSBC and JPMorgan**.

The dataset contains **602 trading days from January 2018 to May 2020**, including the COVID-19 market crash.

From the underlying news data, six sentiment indicators are constructed:

* `positiveP`
* `negativeP`
* `BULL`
* `BEAR`
* `BBr` — bullish sentiment share
* `PNlog` — log-transformed positive-to-negative sentiment

Relative news volume (`RVT`) is also used to identify periods of unusually intense news activity.

### Trading Strategies

Two distinct strategy families are evaluated.

#### 1. Single-stock sentiment strategy

Sentiment indicators are standardised using rolling z-scores:

$$
z_t = \frac{e_t-\mu_t}{\sigma_t}
$$

Positions are opened when sentiment moves sufficiently far from its recent mean.

Both **momentum** and **contrarian** specifications are tested.

A news-volume regime filter can temporarily move the strategy to cash during periods of unusually high information flow.

Signals are shifted by one day so that positions are taken only after the signal becomes observable.

#### 2. Relative-sentiment pairs strategy

The second strategy compares sentiment between JPMorgan and HSBC:

$$
g_t=e_t^{JPM}-e_t^{HSBC}
$$

The standardised sentiment spread generates a dollar-neutral long-short position between the two banks.

This allows the model to trade **relative sentiment**, rather than predicting the direction of a single stock.

### Backtesting Design

The strategies are evaluated using the performance framework from the course, including:

* Cumulative return
* Annual return
* Annualised Sharpe ratio
* Win rate
* Annualised volatility
* Maximum drawdown
* Maximum drawdown duration
* Number of trades

A **1% transaction cost is charged on every position change**, making turnover a central consideration.

To reduce look-ahead bias, the sample is divided chronologically:

* **70% training period**
* **30% held-out test period**

Hyperparameters are selected exclusively using the training sample.

The search covers:

* Rolling window: \(n \in \{10,20,40,60,80\}\)
* Z-score threshold: \(k \in \{0.5,1.0,1.5,2.0,2.5\}\)
* News-volume filter: off, 85th percentile, 90th percentile

### Trading Results

The strongest single-stock configuration is an **HSBC contrarian strategy based on positive news sentiment**, with:

* 20-day rolling window
* Z-score threshold of 2.0
* 85th-percentile news-volume filter
* Long-short positioning
* 12 trades over the sample

Its additive cumulative return is **0.508**, compared with **−0.633** for HSBC buy-and-hold under the same performance convention.

On the held-out test period, the strategy's excess return relative to buy-and-hold is **0.749**.

The strongest pairs strategy uses the **relative positive-sentiment spread between JPMorgan and HSBC** in momentum mode. It produces a cumulative return of **0.663**, compared with **−0.313** for the equal-weight long-both benchmark.

These results should be interpreted within the small sample and the specific transaction-cost and performance conventions used in the exercise.

### Robustness

Several checks are used to assess whether the headline results depend on a single parameter configuration:

* Held-out test performance
* Alternative rolling windows
* Alternative sentiment thresholds
* Pre- and post-COVID performance
* Rolling-window evaluation
* Transaction-cost sensitivity

The strongest HSBC configuration remains positive relative to buy-and-hold across a broad range of parameter values rather than appearing only at one isolated combination.

---

# Part II — Machine Learning for Option Pricing

## Research Question

> **Can machine-learning models approximate computationally expensive option-pricing functions accurately enough to provide substantial computational speedups?**

The second part treats option pricing as a supervised learning problem.

Numerical pricing engines are first used to generate synthetic datasets of:

$$
(\text{option/model parameters}) \rightarrow (\text{option price})
$$

Machine-learning models are then trained to approximate this pricing function.

The trained models can subsequently produce prices much faster than repeatedly running the underlying numerical engine.

## Surrogate Pricing Framework

The project studies two pricing problems.

### A. American Options

American put and call options are priced using a **Cox-Ross-Rubinstein (CRR) binomial tree**.

The tree uses:

* 1,000 time steps
* Variable strike
* Maturity
* Interest rate
* Dividend yield
* Volatility

The spot price is normalised to:

$$
S_0=100
$$

A total of **1,500 Latin-hypercube parameter combinations** are generated.

Each combination is priced separately as an American put and call, producing the training datasets.

### B. Barrier Options under Heston

The second pricing problem is a **down-and-out European call under the Heston stochastic-volatility model**.

The model is:

$$
dS_t=(r-q)S_tdt+\sqrt{\nu_t}S_tdW_t^S
$$

$$
d\nu_t=\kappa(\theta-\nu_t)dt+\eta\sqrt{\nu_t}dW_t^\nu
$$

with:

$$
dW_t^S dW_t^\nu=\rho dt
$$

The numerical engine uses:

* Full-truncation Euler discretisation
* Daily monitoring
* 20,000 Monte Carlo paths
* Variable strike and barrier
* Heston variance parameters
* Interest rate and dividend yield

A dataset of **1,000 Latin-hypercube parameter combinations** is generated.

The resulting labels contain Monte Carlo sampling noise, which is explicitly tracked through the standard error of each simulated price.

## Machine-Learning Models

Two supervised learning approaches are compared.

### Gaussian Process Regression

The GPR models use an anisotropic **Matérn 5/2 kernel** with automatic relevance determination.

This allows the model to learn a separate length scale for each input variable.

For the Heston dataset, a learned noise component is included to account for Monte Carlo uncertainty in the training labels.

GPR is particularly useful in this setting because it provides both:

* Point predictions
* Predictive uncertainty

### Neural Network

A feed-forward multilayer perceptron is trained using:

* ReLU activation functions
* Adam optimisation
* Standardised inputs and targets
* Early stopping
* L2 regularisation
* Cross-validated architecture selection

The selected architecture is a two-layer network with **128 neurons per hidden layer**.

## Validation

The numerical pricing engines are independently validated before being used to generate training data.

For the American options, the CRR implementation is checked against Black-Scholes-Merton and known early-exercise properties.

For the Heston barrier option, the simulation is validated against:

* Semi-analytic Heston pricing
* Black-Scholes limiting cases
* Reiner-Rubinstein barrier pricing
* Broadie-Glasserman-Kou monitoring corrections

This ensures that the machine-learning models are trained against a validated numerical benchmark rather than an untested implementation.

## Machine-Learning Results

Both machine-learning approaches achieve high predictive accuracy, but GPR consistently performs better on the held-out data.

| Pricing problem           | Model          |   RMSE |    MAE |     R² | NRMSE |
| ------------------------- | -------------- | -----: | -----: | -----: | ----: |
| American put              | GPR            | $0.088 | $0.040 | 0.9999 | 0.008 |
| American put              | Neural network | $0.428 | $0.300 | 0.9985 | 0.039 |
| American call             | GPR            | $0.069 | $0.037 | 1.0000 | 0.007 |
| American call             | Neural network | $0.320 | $0.233 | 0.9990 | 0.032 |
| Barrier down-and-out call | GPR            | $0.320 | $0.232 | 0.9985 | 0.039 |
| Barrier down-and-out call | Neural network | $0.979 | $0.733 | 0.9859 | 0.119 |

The American-option problem is easier to approximate because it produces deterministic, relatively smooth labels in a five-dimensional parameter space.

The barrier-option problem is more difficult because it combines:

* Ten input dimensions
* A knock-out boundary
* Monte Carlo sampling noise
* Euler discretisation

## Computational Speedup

The machine-learning surrogates dramatically reduce the cost of repeated pricing.

| Pricing engine         | Surrogate      | Time per option |  Speedup |
| ---------------------- | -------------- | --------------: | -------: |
| American binomial tree | GPR            |        0.068 ms |    ~190× |
| American binomial tree | Neural network |        0.005 ms |  ~2,454× |
| Heston Monte Carlo     | GPR            |        0.070 ms |  ~3,391× |
| Heston Monte Carlo     | Neural network |        0.010 ms | ~23,358× |

The results illustrate the fundamental trade-off:

**GPR** provides greater predictive accuracy and uncertainty estimates, while the **neural network** provides substantially faster inference and better scalability.

For repeated pricing applications such as calibration, scenario analysis, or portfolio risk calculations, the one-off cost of generating the training data can therefore be amortised over a large number of subsequent valuations.

---

# Part III — Monte Carlo Improvements

The final part improves the Heston Monte Carlo engine used to generate the barrier-option dataset.

The project distinguishes between two types of improvement:

### Engineering improvements

* NumPy path vectorisation
* Numba just-in-time compilation
* Multithreaded execution

### Statistical improvements

* Antithetic variates
* Control variates

The improvements are evaluated using **work-normalised variance**, combining computational cost and estimator variance into a common efficiency measure.

## Control Variate

The strongest statistical improvement uses the semi-analytic Heston European call price as a control variate.

The simulated European payoff is highly correlated with the barrier payoff because the two payoffs coincide whenever the barrier is not breached.

On the benchmark contract:

* Correlation: approximately **0.96**
* Variance reduction: approximately **11.7×**

Combining the control variate with antithetic sampling and parallel computation produces approximately **38× the efficiency of the original Q2 production engine**.

The improved estimator is also tested by regenerating a subset of the original barrier dataset at matched accuracy.

## Key Findings

The project produces several broader findings about machine learning and computational finance:

1. **Financial text can be transformed into systematic trading signals**, although transaction costs make low-turnover strategies particularly important.

2. **Machine learning can approximate complex pricing functions extremely accurately** within a well-defined parameter domain.

3. **Gaussian Process Regression provides a strong combination of accuracy and uncertainty quantification** for relatively small synthetic datasets.

4. **Neural networks provide substantially faster inference**, making them attractive when pricing throughput is the primary constraint.

5. **Monte Carlo efficiency can be improved through both engineering and statistical methods**, with the largest gains coming from combining variance reduction with parallel computation.

6. **Surrogate models should not be treated as unrestricted pricing engines**: their accuracy is established within the parameter region used to generate the training data, and extrapolation remains uncertain.
