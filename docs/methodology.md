# Methodology and interpretation

## Normalized growth simulation

The environment state is a five-value vector in `[0, 1]`:

1. light
2. nutrient level
3. temperature proxy
4. ultrasound exposure
5. trace elements

The action sets every value except temperature. Default growth is:

```text
growth = 0.2 × (
    0.4 × light
  + 0.3 × nutrient
  + 0.2 × ultrasound
  + 0.1 × trace_elements
)

reward = growth - 0.05
```

Temperature performs a bounded random drift but does not affect the default
growth equation. This is retained prototype behavior and an explicit limitation,
not a biological claim.

## Reinforcement learning

PPO observes the normalized five-value state and selects four normalized
controls. Because the coefficients are monotonic and costs/constraints are not
modeled, the learned policy is expected to favor large controllable values.
Useful RL research extensions should introduce measured calibration, resource
costs, toxicity bounds, delayed effects, and uncertainty.

## Sound exposure

The automation opens one hash-routed frequency page per configured treatment,
starts playback, waits, stops playback, and appends a session row. A `success`
row records automation completion; it does not prove acoustic amplitude,
frequency accuracy, sample exposure, or biological response. Those require
independent instrumentation and protocol records.

## Turbidity/lux analysis

The retained table has four groups of three samples on seven measured days.
The analysis reports arithmetic means and standard errors using sample standard
deviation. Missing days are not interpolated. Confirm sample independence,
instrument direction, measurement units, and treatment allocation before using
inferential statistics.

