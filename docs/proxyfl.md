# ProxyFL for object detection

`proxyfl` trains the general detection head with Indecisive-Categories Proxy Learning (ICPL) locally and Global Proxy Tuning (GPT) after aggregation.

```bash
python -m tefed train --config local.toml --method proxyfl --seed 0 --output runs/proxyfl_0
```

## Configuration

```toml
[baselines.proxyfl]
pseudo_threshold = 0.95
proxy_weight = 1.0
proposal_floor = 0.05
server_steps = 10
server_learning_rate = 0.005
prior_momentum = 0.99
```

The fields control the class-confidence threshold, ICPL weight, foreground score floor, full-matrix GPT steps, GPT learning rate and prior EMA momentum.

## Local learning

Proxies are the class-convolution weights. ICPL uses matching one-to-many input features in separate per-scale pools. Labeled objects supply true-class singleton sets and negatives at their box-center grid cells. Weak and photometric strong views share proposal indices.

The frozen round-start detector supplies class distribution `q` using calibration temperature `T_C`; current local probabilities use ordinary softmax. The native detector score controls `proposal_floor`. For ICPL, local or global class confidence reaching the threshold gives a global top-1 singleton. Otherwise, the possible-class set is `S={c:q[c]>prior[c]}`. Pseudo box supervision uses high-confidence global proposals passing the foreground floor.

With normalized feature `z` and normalized class row `p`, the positive logit is `z·p[argmax(q)]` for high confidence and `z·sum(q[c]p[c],c in S)` for low confidence. The low-confidence weighted sum is unnormalized. If `S` is empty, omit both weak and strong features from ICPL anchors and negatives. `omitted_empty_count` counts these proposals once, while `low_count` counts the retained nonempty low-confidence proposals. High-confidence singletons and labeled negatives retain their existing roles.

Negatives have nonempty disjoint class sets and feature cosine at least `1e-6`; self-pairs are excluded. ICPL averages `log(exp(positive)+sum(exp(negative)))-positive` over the retained weak/strong unlabeled anchors across scales. Global probabilities and the prior are detached; student features and proxy weights receive gradients.

## Global update

Sample-weighted aggregation initializes the detector. GPT then updates both native classification branches, preserving the aggregated biases and remaining parameters. Each uploaded local class row `W_m[c]` is a fixed anchor, with global-class logits:

```text
delta = min(max_c min_(j != c) ||W_avg[c] - W_avg[j]||_2, 100)
logit_j = -||W_m[c] - W_G[j]||_2 - delta * 1[j=c]
```

Plain SGD minimizes mean cross-entropy; `server_steps` counts full-matrix updates. Client counts from the final local epoch include labeled classes and ICPL-accepted global pseudo classes. Weak/strong copies count once. The server normalizes summed counts and updates `prior_next = momentum*prior + (1-momentum)*round_distribution`. The prior starts uniformly and is retained in recovery snapshots.

Algorithm sources: [ProxyFL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_ProxyFL_A_Proxy-Guided_Framework_for_Federated_Semi-Supervised_Learning_CVPR_2026_paper.pdf) and [reference implementation f732a279](https://github.com/DuowenC/FSSLlib/tree/f732a279d0ea4f9e9c7d9022164dc96ef6c0ab78).
