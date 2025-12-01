#!/usr/bin/env python3
"""
PQC Scheduler - Statistical Analysis Script
Copyright (c) 2025 Dyber, Inc. All Rights Reserved.

Performs statistical analysis on benchmark results including:
- Welch's t-test with Bonferroni correction
- Cohen's d effect size
- Bootstrap confidence intervals (BCa method)
- Generates publication-quality figures
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results(results_dir: Path) -> List[Dict]:
    """Load all run_*.json files from a results directory."""
    results = []
    for f in sorted(results_dir.glob("run_*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def extract_metric(results: List[Dict], metric: str) -> np.ndarray:
    """Extract a specific metric from all results."""
    return np.array([r[metric] for r in results if metric in r])


def welch_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Perform Welch's t-test for unequal variances."""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (group1.mean() - group2.mean()) / pooled_std


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, 
                 ci: float = 0.95, method: str = 'bca') -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval using BCa method.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        method: 'percentile' or 'bca' (bias-corrected and accelerated)
    
    Returns:
        (lower, upper) confidence interval bounds
    """
    n = len(data)
    alpha = 1 - ci
    
    # Generate bootstrap samples
    boot_means = np.array([
        np.random.choice(data, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    
    if method == 'percentile':
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    elif method == 'bca':
        # Bias correction
        theta_hat = data.mean()
        z0 = stats.norm.ppf(np.mean(boot_means < theta_hat))
        
        # Acceleration (jackknife)
        jackknife_means = np.array([
            np.delete(data, i).mean() for i in range(n)
        ])
        jack_mean = jackknife_means.mean()
        num = np.sum((jack_mean - jackknife_means) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
        a = num / den if den != 0 else 0
        
        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)
        
        p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))
        
        lower = np.percentile(boot_means, 100 * p_low)
        upper = np.percentile(boot_means, 100 * p_high)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return lower, upper


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction to multiple p-values."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return [(p, p < corrected_alpha) for p in p_values]


def analyze_comparison(baseline: np.ndarray, treatment: np.ndarray, 
                       metric_name: str, alpha: float = 0.01) -> Dict:
    """Perform complete statistical comparison between two groups."""
    t_stat, p_value = welch_t_test(treatment, baseline)
    effect_size = cohens_d(treatment, baseline)
    
    diff = treatment.mean() - baseline.mean()
    ci_low, ci_high = bootstrap_ci(treatment - baseline.mean(), method='bca')
    
    pct_improvement = (diff / baseline.mean()) * 100 if baseline.mean() != 0 else 0
    
    return {
        'metric': metric_name,
        'baseline_mean': baseline.mean(),
        'baseline_std': baseline.std(),
        'treatment_mean': treatment.mean(),
        'treatment_std': treatment.std(),
        'difference': diff,
        'pct_improvement': pct_improvement,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': effect_size,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'significant': p_value < alpha,
        'n_baseline': len(baseline),
        'n_treatment': len(treatment),
    }


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def print_analysis_report(analyses: List[Dict], output_file: Optional[str] = None):
    """Print formatted analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("PQC Scheduler Benchmark Results")
    lines.append("=" * 80)
    lines.append("")
    
    for analysis in analyses:
        lines.append(f"Metric: {analysis['metric']}")
        lines.append("-" * 40)
        lines.append(f"  Baseline:  {analysis['baseline_mean']:.4f} ± {analysis['baseline_std']:.4f} (n={analysis['n_baseline']})")
        lines.append(f"  Treatment: {analysis['treatment_mean']:.4f} ± {analysis['treatment_std']:.4f} (n={analysis['n_treatment']})")
        lines.append(f"  Difference: {analysis['difference']:.4f} ({analysis['pct_improvement']:+.2f}%)")
        lines.append(f"  95% CI: [{analysis['ci_95_low']:.4f}, {analysis['ci_95_high']:.4f}]")
        lines.append("")
        lines.append(f"  t-statistic: {analysis['t_statistic']:.4f}")
        lines.append(f"  p-value: {analysis['p_value']:.2e}")
        lines.append(f"  Cohen's d: {analysis['cohens_d']:.4f} ({effect_size_interpretation(analysis['cohens_d'])})")
        lines.append(f"  Significant: {'Yes' if analysis['significant'] else 'No'}")
        lines.append("")
    
    report = "\n".join(lines)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)


def plot_comparison_bars(analyses: List[Dict], output_path: str):
    """Create bar chart comparing baseline vs treatment."""
    fig, axes = plt.subplots(1, len(analyses), figsize=(4 * len(analyses), 6))
    if len(analyses) == 1:
        axes = [axes]
    
    for ax, analysis in zip(axes, analyses):
        means = [analysis['baseline_mean'], analysis['treatment_mean']]
        stds = [analysis['baseline_std'], analysis['treatment_std']]
        
        bars = ax.bar(['Baseline', 'MDP-Optimal'], means, yerr=stds, 
                      capsize=5, color=['#3498db', '#2ecc71'], alpha=0.8)
        
        ax.set_ylabel(analysis['metric'])
        ax.set_title(f"{analysis['pct_improvement']:+.1f}% improvement")
        
        # Add significance marker
        if analysis['significant']:
            y_max = max(means) + max(stds)
            ax.annotate('***' if analysis['p_value'] < 0.001 else '**' if analysis['p_value'] < 0.01 else '*',
                       xy=(0.5, y_max * 1.05), ha='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def plot_latency_distribution(results: List[Dict], output_path: str):
    """Plot latency distribution with percentiles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latencies = extract_metric(results, 'latency_mean_us')
    p50 = extract_metric(results, 'latency_p50_us')
    p99 = extract_metric(results, 'latency_p99_us')
    
    x = np.arange(len(results))
    width = 0.25
    
    ax.bar(x - width, latencies, width, label='Mean', color='#3498db', alpha=0.8)
    ax.bar(x, p50, width, label='P50', color='#2ecc71', alpha=0.8)
    ax.bar(x + width, p99, width, label='P99', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Run')
    ax.set_ylabel('Latency (μs)')
    ax.set_title('Latency Distribution Across Runs')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([f'Run {i}' for i in range(len(results))])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved latency distribution to {output_path}")


def plot_throughput_timeline(results: List[Dict], output_path: str):
    """Plot throughput across runs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    throughputs = extract_metric(results, 'throughput_mean')
    
    ax.plot(range(len(throughputs)), throughputs, 'o-', 
            color='#3498db', linewidth=2, markersize=8)
    ax.axhline(y=throughputs.mean(), color='#e74c3c', linestyle='--', 
               label=f'Mean: {throughputs.mean():.0f} ops/s')
    
    ax.fill_between(range(len(throughputs)), 
                    throughputs.mean() - throughputs.std(),
                    throughputs.mean() + throughputs.std(),
                    alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Run')
    ax.set_ylabel('Throughput (ops/s)')
    ax.set_title('Throughput Consistency Across Runs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved throughput timeline to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Statistical analysis of PQC Scheduler benchmark results'
    )
    parser.add_argument('results_dir', type=Path, 
                        help='Directory containing run_*.json files')
    parser.add_argument('--baseline', type=Path, 
                        help='Baseline results directory for comparison')
    parser.add_argument('--output', type=Path, default=Path('.'),
                        help='Output directory for reports and figures')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Significance level (default: 0.01)')
    parser.add_argument('--bootstrap', type=int, default=10000,
                        help='Number of bootstrap samples (default: 10000)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} runs")
    
    if len(results) == 0:
        print("No results found!")
        sys.exit(1)
    
    # Generate plots
    plot_latency_distribution(results, args.output / 'latency_distribution.png')
    plot_throughput_timeline(results, args.output / 'throughput_timeline.png')
    
    # Comparison analysis if baseline provided
    if args.baseline:
        print(f"\nLoading baseline from {args.baseline}")
        baseline_results = load_results(args.baseline)
        print(f"Loaded {len(baseline_results)} baseline runs")
        
        metrics = ['throughput_mean', 'latency_mean_us', 'sla_compliance', 'security_score']
        analyses = []
        
        for metric in metrics:
            baseline_data = extract_metric(baseline_results, metric)
            treatment_data = extract_metric(results, metric)
            
            if len(baseline_data) > 0 and len(treatment_data) > 0:
                analysis = analyze_comparison(
                    baseline_data, treatment_data, metric, args.alpha
                )
                analyses.append(analysis)
        
        # Apply Bonferroni correction
        p_values = [a['p_value'] for a in analyses]
        corrected = bonferroni_correction(p_values, args.alpha)
        for i, (p, sig) in enumerate(corrected):
            analyses[i]['bonferroni_significant'] = sig
        
        # Print and save report
        print_analysis_report(analyses, args.output / 'statistical_report.txt')
        
        # Save JSON
        with open(args.output / 'statistical_analysis.json', 'w') as f:
            json.dump(analyses, f, indent=2)
        
        # Generate comparison plots
        plot_comparison_bars(analyses, args.output / 'comparison_bars.png')
    
    print(f"\nAnalysis complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()
