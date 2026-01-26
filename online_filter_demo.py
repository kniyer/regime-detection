"""
Online HMM Filtering: Real-Time Regime Detection Demo
Math & Markets - Post 71

This script demonstrates the OnlineHMMFilter class for real-time
regime detection. It visualizes how the filter accumulates evidence
and the unavoidable detection lag when regimes change.

Requirements:
    pip install numpy matplotlib

All code is open source—use it, modify it, build on it. 
No guarantees about correctness or performance. 
Test everything yourself before deploying with real capital.

Author: K. Iyer
Substack: https://kniyer.substack.com
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# Color scheme
BLUE = '#2E86AB'
RED = '#E94F37'
GRAY = '#6C757D'
GREEN = '#28A745'
ORANGE = '#F77F00'
PURPLE = '#7B2CBF'
TEAL = '#20C997'


class OnlineHMMFilter:
    """
    Real-time HMM filtering for regime detection.
    Updates state probabilities one observation at a time.
    """
    
    def __init__(self, transition_matrix, emission_params):
        self.A = transition_matrix
        self.params = emission_params
        self.belief = np.array([0.8, 0.2])  # Initial: 80% Bull
        self.history = [0.2]  # Track P(Bear) history
        
    def _emission_prob(self, r, state):
        """Probability of observing return r in given state."""
        if state == 0:  # Bull
            mu, sigma = self.params['bull_mu'], self.params['bull_sigma']
        else:  # Bear
            mu, sigma = self.params['bear_mu'], self.params['bear_sigma']
        
        return (1 / (sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((r - mu) / sigma) ** 2)
    
    def update(self, r):
        """Update belief given new observation. Returns P(Bear)."""
        # Predict: apply transition matrix
        prior = self.belief @ self.A
        
        # Update: incorporate observation
        likelihood = np.array([
            self._emission_prob(r, 0),
            self._emission_prob(r, 1)
        ])
        
        posterior = prior * likelihood
        posterior /= posterior.sum()
        
        self.belief = posterior
        self.history.append(posterior[1])
        return posterior[1]
    
    def reset(self):
        self.belief = np.array([0.8, 0.2])
        self.history = [0.2]


# Setup filter with typical market parameters
A = np.array([[0.98, 0.02],
              [0.05, 0.95]])

params = {
    'bull_mu': 0.05,
    'bull_sigma': 1.0,
    'bear_mu': -0.10,
    'bear_sigma': 2.2
}

# Generate a realistic scenario with regime change
np.random.seed(123)  # Different seed for clearer crash pattern

# Scenario: Bull market, then crash, then recovery
n_days = 60
true_states = np.zeros(n_days, dtype=int)
true_states[0:20] = 0    # Bull
true_states[20:40] = 1   # Bear (crash)
true_states[40:60] = 0   # Recovery

returns = []
for t in range(n_days):
    if true_states[t] == 0:
        returns.append(np.random.randn() * 1.0 + 0.05)
    else:
        returns.append(np.random.randn() * 2.2 - 0.15)

returns = np.array(returns)

# Run the filter
hmm_filter = OnlineHMMFilter(A, params)
p_bear_history = [0.2]  # Initial

for r in returns:
    p_bear = hmm_filter.update(r)
    
# Get full history (includes initial)
p_bear_history = hmm_filter.history

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: Daily Returns with regime coloring
ax1 = axes[0]
for t in range(n_days):
    color = GREEN if true_states[t] == 0 else RED
    ax1.bar(t+1, returns[t], color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.axvline(x=20.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=40.5, color='black', linestyle='--', linewidth=2, alpha=0.7)

ax1.set_ylabel('Daily Return (%)')
ax1.set_title('Online HMM Filtering: Real-Time Regime Detection', fontsize=14, fontweight='bold')

# Add regime labels
ax1.text(10, 4, 'BULL', fontsize=12, fontweight='bold', color=GREEN, ha='center')
ax1.text(30, 4, 'BEAR', fontsize=12, fontweight='bold', color=RED, ha='center')
ax1.text(50, 4, 'BULL', fontsize=12, fontweight='bold', color=GREEN, ha='center')

# Legend
bull_patch = mpatches.Patch(color=GREEN, alpha=0.7, label='Bull regime')
bear_patch = mpatches.Patch(color=RED, alpha=0.7, label='Bear regime')
ax1.legend(handles=[bull_patch, bear_patch], loc='upper right')

ax1.set_ylim(-7, 5)

# Panel 2: P(Bear) Evolution
ax2 = axes[1]
times = np.arange(0, n_days + 1)
ax2.fill_between(times, 0, p_bear_history, alpha=0.4, color=RED, label='P(Bear)')
ax2.fill_between(times, p_bear_history, 1, alpha=0.4, color=GREEN, label='P(Bull)')
ax2.plot(times, p_bear_history, color=PURPLE, linewidth=2)

ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision threshold')
ax2.axvline(x=20.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=40.5, color='black', linestyle='--', linewidth=2, alpha=0.7)

# Find detection points
detection_bear = None
detection_bull = None
for t in range(21, 40):
    if p_bear_history[t] > 0.5 and detection_bear is None:
        detection_bear = t
        break
        
for t in range(41, 60):
    if p_bear_history[t] < 0.5 and detection_bull is None:
        detection_bull = t
        break

# Mark detection points
if detection_bear:
    ax2.axvline(x=detection_bear, color=ORANGE, linestyle=':', linewidth=2)
    ax2.annotate(f'Bear detected\n(day {detection_bear}, lag={detection_bear-20}d)', 
                xy=(detection_bear, 0.6), xytext=(detection_bear+5, 0.75),
                fontsize=9, color=ORANGE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ORANGE))

if detection_bull:
    ax2.axvline(x=detection_bull, color=TEAL, linestyle=':', linewidth=2)
    ax2.annotate(f'Bull detected\n(day {detection_bull}, lag={detection_bull-40}d)', 
                xy=(detection_bull, 0.4), xytext=(detection_bull+5, 0.25),
                fontsize=9, color=TEAL, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=TEAL))

ax2.set_ylabel('P(Bear)')
ax2.set_ylim(0, 1)
ax2.set_title('Filter State: Evidence Accumulates Over Time', fontweight='bold')
ax2.legend(loc='upper right')

# Panel 3: Cumulative P&L with detection-based strategy
ax3 = axes[2]

# Strategy 1: Buy and hold
bh_cumret = (1 + returns/100).cumprod() * 100

# Strategy 2: HMM-based (exit when P(Bear) > 0.5)
hmm_returns = []
for t in range(n_days):
    if p_bear_history[t] > 0.5:
        hmm_returns.append(0)  # In cash
    else:
        hmm_returns.append(returns[t])  # In market
hmm_cumret = (1 + np.array(hmm_returns)/100).cumprod() * 100

# Strategy 3: Oracle (perfect knowledge)
oracle_returns = np.where(true_states == 0, returns, 0)
oracle_cumret = (1 + oracle_returns/100).cumprod() * 100

ax3.plot(range(1, n_days+1), bh_cumret, color=GRAY, linewidth=2, label='Buy & Hold')
ax3.plot(range(1, n_days+1), oracle_cumret, color=GREEN, linewidth=2, linestyle='--', label='Oracle (perfect)')
ax3.plot(range(1, n_days+1), hmm_cumret, color=BLUE, linewidth=2.5, label='HMM Filter')

ax3.axhline(y=100, color='black', linewidth=0.5)
ax3.axvline(x=20.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(x=40.5, color='black', linestyle='--', linewidth=2, alpha=0.7)

# Shade the detection lag period (exposed to bear but not detected yet)
if detection_bear:
    ax3.axvspan(20, detection_bear, alpha=0.2, color=ORANGE, label='Detection lag (exposed)')

ax3.set_xlabel('Day')
ax3.set_ylabel('Portfolio Value')
ax3.set_title('Strategy Performance: HMM vs Buy & Hold vs Oracle', fontweight='bold')
ax3.legend(loc='lower left')

# Add final stats
final_bh = bh_cumret[-1]
final_hmm = hmm_cumret[-1]
final_oracle = oracle_cumret[-1]

stats_text = (f'Final Values:\n'
              f'  Buy & Hold: {final_bh:.1f}\n'
              f'  HMM Filter: {final_hmm:.1f}\n'
              f'  Oracle: {final_oracle:.1f}')
ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('/home/claude/chart_online_filter_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print("Online HMM Filter Demo saved!")
print(f"\nDetection lag Bull→Bear: {detection_bear - 20} days")
print(f"Detection lag Bear→Bull: {detection_bull - 40} days")
print(f"\nFinal portfolio values:")
print(f"  Buy & Hold: {final_bh:.1f}")
print(f"  HMM Filter: {final_hmm:.1f}")
print(f"  Oracle:     {final_oracle:.1f}")

# Also create a step-by-step trace showing the filter updates
print("\n" + "="*70)
print("STEP-BY-STEP FILTER TRACE (first 25 days + regime change period)")
print("="*70)
print(f"{'Day':>4} {'Return':>8} {'P(Bear)':>10} {'Signal':>10} {'True State':>12}")
print("-"*70)

for t in range(min(35, n_days)):
    ret = returns[t]
    p_bear = p_bear_history[t+1]
    signal = 'BEAR' if p_bear > 0.5 else 'BULL'
    true_st = 'Bear' if true_states[t] == 1 else 'Bull'
    
    # Highlight regime changes
    marker = ""
    if t == 20:
        marker = " ← CRASH STARTS"
    elif t == detection_bear:
        marker = " ← DETECTED!"
    
    print(f"{t+1:>4} {ret:>+8.2f}% {p_bear:>10.1%} {signal:>10} {true_st:>12}{marker}")
