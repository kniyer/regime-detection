"""
Ensemble Regime Detector for V6
Math & Markets - Post 71

This module implements an ensemble approach to regime detection,
combining HMM, VIX-based, and volatility breakout signals into
a unified P(Bear) estimate with confirmation logic.

Requirements:
    pip install numpy

All code is open source—use it, modify it, build on it. 
No guarantees about correctness or performance. 
Test everything yourself before deploying with real capital.

Author: K. Iyer
Substack: https://kniyer.substack.com
"""

import numpy as np


class EnsembleRegimeDetector:
    """
    Combines multiple regime signals into ensemble P(Bear).
    
    Signals:
        - HMM: Hidden Markov Model filtered probability
        - VIX: Threshold-based signal from VIX level
        - Vol: Volatility breakout vs baseline
    
    Features:
        - Weighted averaging of signals
        - Confirmation layer (require agreement for extreme calls)
        - Hysteresis to prevent thrashing
    """
    
    def __init__(self, weights=None):
        """
        Initialize detector with signal weights.
        
        Parameters:
            weights: dict with keys 'hmm', 'vix', 'vol'
                     Default: {'hmm': 0.40, 'vix': 0.35, 'vol': 0.25}
        """
        self.weights = weights or {'hmm': 0.40, 'vix': 0.35, 'vol': 0.25}
        self.last_p_bear = 0.3  # For hysteresis tracking
        self.last_allocation = 'neutral'
        
    def vix_signal(self, vix):
        """
        Convert VIX level to P(Bear) using threshold logic.
        
        Parameters:
            vix: Current VIX level
            
        Returns:
            P(Bear) between 0 and 1
        """
        if vix < 18:
            return 0.1
        elif vix < 25:
            return 0.3
        elif vix < 35:
            return 0.6
        else:
            return 0.9
    
    def vol_signal(self, realized_vol, baseline_vol):
        """
        Compare current realized vol to baseline for breakout detection.
        
        Parameters:
            realized_vol: Current 20-day realized volatility (annualized)
            baseline_vol: Baseline 60-day realized volatility (annualized)
            
        Returns:
            P(Bear) between 0 and 1
        """
        if baseline_vol <= 0:
            return 0.3  # Default to neutral if no baseline
        
        ratio = realized_vol / baseline_vol
        if ratio > 1.5:
            return 0.7
        elif ratio > 1.2:
            return 0.4
        else:
            return 0.2
    
    def compute_ensemble(self, hmm_p_bear, vix, realized_vol, baseline_vol):
        """
        Compute weighted ensemble P(Bear) from all signals.
        
        Parameters:
            hmm_p_bear: P(Bear) from HMM filter (0-1)
            vix: Current VIX level
            realized_vol: 20-day realized vol (annualized)
            baseline_vol: 60-day baseline vol (annualized)
            
        Returns:
            tuple: (p_bear, signals_dict, agreement_ratio)
        """
        # Compute individual signals
        p_hmm = hmm_p_bear
        p_vix = self.vix_signal(vix)
        p_vol = self.vol_signal(realized_vol, baseline_vol)
        
        signals = {'hmm': p_hmm, 'vix': p_vix, 'vol': p_vol}
        
        # Weighted average
        p_bear = (
            self.weights['hmm'] * p_hmm +
            self.weights['vix'] * p_vix +
            self.weights['vol'] * p_vol
        )
        
        # Check agreement (how many signals indicate Bear > 0.5)
        bear_votes = sum(1 for p in [p_hmm, p_vix, p_vol] if p > 0.5)
        agreement = bear_votes / 3
        
        return p_bear, signals, agreement
    
    def get_allocation(self, p_bear, agreement, hysteresis=0.1):
        """
        Convert P(Bear) to allocation decision with hysteresis and confirmation.
        
        Parameters:
            p_bear: Ensemble P(Bear)
            agreement: Fraction of signals agreeing (0-1)
            hysteresis: Buffer to prevent thrashing (default 0.1)
            
        Returns:
            tuple: (allocation_string, confidence)
            allocation_string: 'risk_on', 'neutral', or 'risk_off'
            confidence: Strength of signal (0-1)
        """
        # Apply hysteresis - ignore small changes
        if abs(p_bear - self.last_p_bear) < hysteresis:
            effective_p = self.last_p_bear
        else:
            effective_p = p_bear
            self.last_p_bear = p_bear
        
        # Confirmation check for extreme calls
        # If signals disagree, stay neutral
        if effective_p > 0.6 and agreement < 0.67:
            # Want to go risk-off but signals don't agree strongly
            effective_p = 0.5
        
        if effective_p < 0.3 and agreement > 0.33:
            # Want to go risk-on but some signals say Bear
            effective_p = 0.4
        
        # Map to allocation zones
        if effective_p < 0.3:
            allocation = 'risk_on'
            confidence = 1 - effective_p
        elif effective_p > 0.6:
            allocation = 'risk_off'
            confidence = effective_p
        else:
            allocation = 'neutral'
            confidence = 0.5
        
        self.last_allocation = allocation
        return allocation, confidence
    
    def update(self, hmm_p_bear, vix, realized_vol, baseline_vol):
        """
        Full update cycle: compute ensemble and get allocation.
        
        Parameters:
            hmm_p_bear: P(Bear) from HMM filter
            vix: Current VIX level
            realized_vol: 20-day realized vol
            baseline_vol: 60-day baseline vol
            
        Returns:
            dict with all outputs
        """
        p_bear, signals, agreement = self.compute_ensemble(
            hmm_p_bear, vix, realized_vol, baseline_vol
        )
        allocation, confidence = self.get_allocation(p_bear, agreement)
        
        return {
            'p_bear': p_bear,
            'signals': signals,
            'agreement': agreement,
            'allocation': allocation,
            'confidence': confidence
        }
    
    def reset(self):
        """Reset internal state."""
        self.last_p_bear = 0.3
        self.last_allocation = 'neutral'


def compute_realized_vol(returns, window=20):
    """
    Compute annualized realized volatility.
    
    Parameters:
        returns: Array of daily returns (in %)
        window: Lookback window in days
        
    Returns:
        Annualized volatility (in %)
    """
    if len(returns) < window:
        return np.std(returns) * np.sqrt(252)
    return np.std(returns[-window:]) * np.sqrt(252)


# =============================================================================
# Demo / Example Usage
# =============================================================================

if __name__ == "__main__":
    detector = EnsembleRegimeDetector()
    
    print("=" * 70)
    print("ENSEMBLE REGIME DETECTOR - Demo")
    print("=" * 70)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Calm Bull Market',
            'hmm': 0.15, 'vix': 14, 'rvol': 12, 'baseline': 15
        },
        {
            'name': 'Elevated Uncertainty',
            'hmm': 0.45, 'vix': 22, 'rvol': 18, 'baseline': 15
        },
        {
            'name': 'Crisis Mode',
            'hmm': 0.75, 'vix': 32, 'rvol': 28, 'baseline': 15
        },
        {
            'name': 'Signal Disagreement',
            'hmm': 0.80, 'vix': 18, 'rvol': 14, 'baseline': 15
        },
        {
            'name': 'VIX Spike, Vol Lag',
            'hmm': 0.50, 'vix': 38, 'rvol': 20, 'baseline': 15
        },
    ]
    
    for scenario in scenarios:
        result = detector.update(
            scenario['hmm'],
            scenario['vix'],
            scenario['rvol'],
            scenario['baseline']
        )
        
        print(f"\n{scenario['name']}")
        print(f"  Inputs: HMM={scenario['hmm']:.2f}, VIX={scenario['vix']}, "
              f"RVol={scenario['rvol']}, Baseline={scenario['baseline']}")
        print(f"  Signals: HMM={result['signals']['hmm']:.2f}, "
              f"VIX={result['signals']['vix']:.2f}, "
              f"Vol={result['signals']['vol']:.2f}")
        print(f"  Ensemble P(Bear): {result['p_bear']:.2f}")
        print(f"  Agreement: {result['agreement']:.0%}")
        print(f"  → Allocation: {result['allocation'].upper()} "
              f"(confidence: {result['confidence']:.2f})")
    
    print("\n" + "=" * 70)
    print("Note: In live use, feed actual HMM output, VIX, and computed")
    print("realized volatility. This demo uses illustrative values.")
    print("=" * 70)
