"""
KubeShield Model Evaluation Script
Evaluates anomaly detection models on test data with comprehensive metrics
"""

import numpy as np
import argparse
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, auc, accuracy_score
)
import matplotlib.pyplot as plt
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates anomaly detection models"""
    
    def __init__(self, threshold=0.85):
        """
        Initialize evaluator
        
        Args:
            threshold: Anomaly detection threshold
        """
        self.threshold = threshold
        logger.info(f"Initialized ModelEvaluator with threshold={threshold}")
    
    def compute_metrics(self, y_true: np.ndarray, 
                       y_scores: np.ndarray,
                       threshold: float = None) -> dict:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: Ground truth labels (0 = normal, 1 = anomaly)
            y_scores: Anomaly scores from model [0, 1]
            threshold: Decision threshold
        
        Returns:
            Dictionary of metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Binary predictions
        y_pred = (y_scores >= threshold).astype(int)
        
        logger.info(f"Computing metrics (threshold={threshold})...")
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except:
            roc_auc = 0.0
        
        # PR-AUC
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall_vals, precision_vals)
        except:
            pr_auc = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold': float(threshold)
        }
        
        return metrics
    
    def threshold_optimization(self, y_true: np.ndarray, 
                              y_scores: np.ndarray,
                              metric='f1') -> Tuple[float, dict]:
        """
        Find optimal threshold by maximizing metric
        
        Args:
            y_true: Ground truth labels
            y_scores: Anomaly scores
            metric: Metric to optimize ('f1', 'roc_auc', 'precision', 'recall')
        
        Returns:
            Tuple of (optimal_threshold, metrics)
        """
        logger.info(f"Optimizing threshold for maximum {metric}...")
        
        best_threshold = 0.5
        best_score = 0.0
        best_metrics = None
        
        thresholds = np.linspace(0.0, 1.0, 101)
        
        for threshold in thresholds:
            metrics = self.compute_metrics(y_true, y_scores, threshold)
            
            if metric == 'f1':
                score = metrics['f1_score']
            elif metric == 'roc_auc':
                score = metrics['roc_auc']
            elif metric == 'precision':
                score = metrics['precision']
            elif metric == 'recall':
                score = metrics['recall']
            else:
                score = metrics['f1_score']
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} "
                   f"({metric}={best_score:.4f})")
        
        return best_threshold, best_metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_scores: np.ndarray,
                      output_path: str = None):
        """
        Plot ROC curve
        
        Args:
            y_true: Ground truth labels
            y_scores: Anomaly scores
            output_path: Path to save plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - KubeShield Anomaly Detection')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {output_path}")
        
        plt.close()
    
    def plot_pr_curve(self, y_true: np.ndarray, 
                     y_scores: np.ndarray,
                     output_path: str = None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: Ground truth labels
            y_scores: Anomaly scores
            output_path: Path to save plot
        """
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_vals, precision_vals)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - KubeShield Anomaly Detection')
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {output_path}")
        
        plt.close()
    
    def generate_report(self, y_true: np.ndarray,
                       y_scores: np.ndarray,
                       output_path: str = None) -> dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: Ground truth labels
            y_scores: Anomaly scores
            output_path: Path to save report
        
        Returns:
            Report dictionary
        """
        logger.info("="*70)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("="*70)
        
        # Compute metrics at default threshold
        metrics = self.compute_metrics(y_true, y_scores)
        
        # Find optimal threshold
        optimal_threshold, optimal_metrics = self.threshold_optimization(y_true, y_scores)
        
        # Classification report
        y_pred = (y_scores >= self.threshold).astype(int)
        class_report = classification_report(y_true, y_pred, 
                                            target_names=['Normal', 'Anomaly'],
                                            output_dict=True)
        
        report = {
            'dataset_statistics': {
                'total_samples': len(y_true),
                'normal_samples': int(np.sum(y_true == 0)),
                'anomaly_samples': int(np.sum(y_true == 1)),
                'anomaly_ratio': float(np.mean(y_true))
            },
            'metrics_at_threshold': metrics,
            'optimal_metrics': optimal_metrics,
            'classification_report': class_report,
            'score_statistics': {
                'min': float(y_scores.min()),
                'max': float(y_scores.max()),
                'mean': float(y_scores.mean()),
                'std': float(y_scores.std())
            }
        }
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def print_report(self, report: dict):
        """Print formatted evaluation report"""
        
        logger.info("\n" + "="*70)
        logger.info("DATASET STATISTICS")
        logger.info("="*70)
        
        stats = report['dataset_statistics']
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Normal samples: {stats['normal_samples']}")
        logger.info(f"Anomaly samples: {stats['anomaly_samples']}")
        logger.info(f"Anomaly ratio: {stats['anomaly_ratio']:.4f}")
        
        logger.info("\n" + "="*70)
        logger.info(f"METRICS AT THRESHOLD {report['metrics_at_threshold']['threshold']:.2f}")
        logger.info("="*70)
        
        metrics = report['metrics_at_threshold']
        logger.info(f"Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"Precision:  {metrics['precision']:.4f}")
        logger.info(f"Recall:     {metrics['recall']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"F1-Score:   {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:     {metrics.get('pr_auc', 0):.4f}")
        
        logger.info("\n" + "="*70)
        logger.info("OPTIMAL METRICS")
        logger.info("="*70)
        
        opt_metrics = report['optimal_metrics']
        logger.info(f"Optimal Threshold: {opt_metrics['threshold']:.4f}")
        logger.info(f"Accuracy:   {opt_metrics['accuracy']:.4f}")
        logger.info(f"Precision:  {opt_metrics['precision']:.4f}")
        logger.info(f"Recall:     {opt_metrics['recall']:.4f}")
        logger.info(f"F1-Score:   {opt_metrics['f1_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate KubeShield anomaly detection')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to prediction scores (CSV)')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth labels (CSV)')
    parser.add_argument('--threshold', type=float, default=0.85,
                       help='Anomaly threshold (default: 0.85)')
    parser.add_argument('--output', type=str, help='Output report path')
    parser.add_argument('--plots', action='store_true',
                       help='Generate ROC and PR curves')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("KubeShield Model Evaluation")
    logger.info("="*70)
    
    # Load data
    logger.info(f"Loading predictions from {args.predictions}...")
    y_scores = np.loadtxt(args.predictions, delimiter=',')
    
    logger.info(f"Loading ground truth from {args.ground_truth}...")
    y_true = np.loadtxt(args.ground_truth, delimiter=',').astype(int)
    
    # Evaluate
    evaluator = ModelEvaluator(threshold=args.threshold)
    report = evaluator.generate_report(y_true, y_scores, output_path=args.output)
    
    # Print report
    evaluator.print_report(report)
    
    # Generate plots
    if args.plots:
        plots_dir = Path('evaluation_plots')
        plots_dir.mkdir(exist_ok=True)
        
        evaluator.plot_roc_curve(y_true, y_scores, 
                                str(plots_dir / 'roc_curve.png'))
        evaluator.plot_pr_curve(y_true, y_scores,
                               str(plots_dir / 'pr_curve.png'))
    
    logger.info("\n✓ Evaluation completed!")


if __name__ == '__main__':
    main()
