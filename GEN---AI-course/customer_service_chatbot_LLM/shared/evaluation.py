"""
Evaluation System
Provides comprehensive model evaluation metrics for all tasks.
Generates accuracy, precision, recall, F1, and confusion matrices.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import json
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationSystem:
    """
    Comprehensive evaluation system for model performance.
    
    Key Metrics:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP) - "When I predict positive, how often am I right?"
    - Recall: TP / (TP + FN) - "Of all actual positives, how many did I find?"
    - F1 Score: Harmonic mean of precision and recall
    - Confusion Matrix: Visual breakdown of predictions
    
    Where:
    - TP = True Positives (correctly predicted positive)
    - TN = True Negatives (correctly predicted negative)
    - FP = False Positives (incorrectly predicted positive)
    - FN = False Negatives (incorrectly predicted negative)
    """
    
    def __init__(self, results_dir: str = "./evaluation_results"):
        """
        Initialize the evaluation system.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Store evaluation history
        self.evaluation_history = []
        
        logger.info(f"✅ Evaluation system initialized")
        logger.info(f"   Results directory: {results_dir}")
    
    def calculate_accuracy(
        self,
        predictions: List[Any],
        ground_truth: List[Any]
    ) -> float:
        """
        Calculate accuracy score.
        
        Accuracy = (Correct Predictions) / (Total Predictions)
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        try:
            if len(predictions) != len(ground_truth):
                raise ValueError("Predictions and ground truth must have same length")
            
            accuracy = accuracy_score(ground_truth, predictions)
            logger.info(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"❌ Accuracy calculation failed: {e}")
            raise
    
    def generate_confusion_matrix(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Confusion Matrix shows:
        - Rows: Actual classes
        - Columns: Predicted classes
        - Diagonal: Correct predictions
        - Off-diagonal: Mistakes
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            labels: Optional label names for display
            
        Returns:
            Confusion matrix as numpy array
        """
        try:
            cm = confusion_matrix(ground_truth, predictions, labels=labels)
            logger.info(f"📊 Confusion matrix generated ({cm.shape[0]}x{cm.shape[1]})")
            
            return cm
            
        except Exception as e:
            logger.error(f"❌ Confusion matrix generation failed: {e}")
            raise
    
    def calculate_precision_recall(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            average: Averaging method ('binary', 'micro', 'macro', 'weighted')
                    - binary: For binary classification
                    - weighted: Accounts for class imbalance (recommended)
                    - macro: Simple average across classes
                    - micro: Global average
            
        Returns:
            Dictionary with precision, recall, and f1_score
        """
        try:
            precision = precision_score(
                ground_truth,
                predictions,
                average=average,
                zero_division=0
            )
            
            recall = recall_score(
                ground_truth,
                predictions,
                average=average,
                zero_division=0
            )
            
            f1 = f1_score(
                ground_truth,
                predictions,
                average=average,
                zero_division=0
            )
            
            results = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"📊 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"📊 F1 Score: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Precision/Recall calculation failed: {e}")
            raise
    
    def evaluate_model(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        task_name: str,
        labels: Optional[List[str]] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        This is the main evaluation function that computes all metrics!
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            task_name: Name of the task being evaluated
            labels: Optional label names
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Evaluating: {task_name}")
            logger.info(f"{'='*60}")
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(predictions, ground_truth)
            
            # Generate confusion matrix
            cm = self.generate_confusion_matrix(predictions, ground_truth, labels)
            
            # Calculate precision, recall, F1
            pr_metrics = self.calculate_precision_recall(
                predictions,
                ground_truth,
                average=average
            )
            
            # Generate classification report
            report = classification_report(
                ground_truth,
                predictions,
                target_names=labels,
                output_dict=True,
                zero_division=0
            )
            
            # Compile results
            results = {
                'task_name': task_name,
                'accuracy': accuracy,
                'precision': pr_metrics['precision'],
                'recall': pr_metrics['recall'],
                'f1_score': pr_metrics['f1_score'],
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'num_samples': len(predictions),
                'timestamp': datetime.now().isoformat(),
                'meets_threshold': accuracy >= 0.70  # 70% minimum requirement
            }
            
            # Check if meets internship requirements
            if results['meets_threshold']:
                logger.info(f"✅ PASSED: Accuracy {accuracy:.2%} >= 70% threshold")
            else:
                logger.warning(f"⚠️ FAILED: Accuracy {accuracy:.2%} < 70% threshold")
            
            # Store in history
            self.evaluation_history.append(results)
            
            logger.info(f"{'='*60}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            raise
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot and optionally save confusion matrix visualization.
        
        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels or range(len(confusion_matrix)),
                yticklabels=labels or range(len(confusion_matrix)),
                cbar_kws={'label': 'Count'}
            )
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Confusion matrix saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Plotting confusion matrix failed: {e}")
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, float],
        title: str = "Model Metrics",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot bar chart comparing different metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(10, 6))
            
            metrics = list(metrics_dict.keys())
            values = list(metrics_dict.values())
            
            # Create bar chart
            bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
            
            # Add 70% threshold line
            plt.axhline(y=0.70, color='red', linestyle='--', label='70% Threshold')
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"💾 Metrics plot saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Plotting metrics failed: {e}")
    
    def log_metrics(
        self,
        task_name: str,
        metrics: Dict[str, Any],
        save_to_file: bool = True
    ) -> None:
        """
        Log metrics to console and optionally save to file.
        
        Args:
            task_name: Name of the task
            metrics: Dictionary of metrics
            save_to_file: Whether to save to JSON file
        """
        try:
            logger.info(f"\n📊 Metrics for {task_name}:")
            logger.info(f"   Accuracy:  {metrics.get('accuracy', 0):.4f}")
            logger.info(f"   Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"   Recall:    {metrics.get('recall', 0):.4f}")
            logger.info(f"   F1 Score:  {metrics.get('f1_score', 0):.4f}")
            
            if save_to_file:
                filename = f"{task_name.replace(' ', '_').lower()}_metrics.json"
                filepath = os.path.join(self.results_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                
                logger.info(f"💾 Metrics saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Logging metrics failed: {e}")
    
    def generate_evaluation_report(
        self,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for all tasks.
        
        Returns:
            Dictionary with summary of all evaluations
        """
        try:
            if not self.evaluation_history:
                logger.warning("⚠️ No evaluations in history")
                return {}
            
            report = {
                'total_evaluations': len(self.evaluation_history),
                'evaluations': self.evaluation_history,
                'summary': {
                    'tasks_passed': sum(1 for e in self.evaluation_history if e['meets_threshold']),
                    'tasks_failed': sum(1 for e in self.evaluation_history if not e['meets_threshold']),
                    'average_accuracy': np.mean([e['accuracy'] for e in self.evaluation_history]),
                    'average_precision': np.mean([e['precision'] for e in self.evaluation_history]),
                    'average_recall': np.mean([e['recall'] for e in self.evaluation_history]),
                    'average_f1': np.mean([e['f1_score'] for e in self.evaluation_history])
                },
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 EVALUATION REPORT SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total Evaluations: {report['total_evaluations']}")
            logger.info(f"Tasks Passed (≥70%): {report['summary']['tasks_passed']}")
            logger.info(f"Tasks Failed (<70%): {report['summary']['tasks_failed']}")
            logger.info(f"Average Accuracy: {report['summary']['average_accuracy']:.4f}")
            logger.info(f"{'='*60}\n")
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"💾 Report saved to: {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Evaluation System\n")
    
    # Initialize evaluation system
    eval_system = EvaluationSystem()
    
    # Example 1: Binary Classification
    print("1️⃣ Testing Binary Classification Evaluation...")
    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred_binary = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    
    results_binary = eval_system.evaluate_model(
        predictions=y_pred_binary,
        ground_truth=y_true_binary,
        task_name="Binary Classification Test",
        labels=['Negative', 'Positive'],
        average='binary'
    )
    
    # Example 2: Multi-class Classification
    print("\n2️⃣ Testing Multi-class Classification Evaluation...")
    y_true_multi = ['positive', 'negative', 'neutral', 'positive', 'negative',
                    'neutral', 'positive', 'negative', 'neutral', 'positive']
    y_pred_multi = ['positive', 'negative', 'neutral', 'positive', 'neutral',
                    'neutral', 'positive', 'negative', 'positive', 'positive']
    
    results_multi = eval_system.evaluate_model(
        predictions=y_pred_multi,
        ground_truth=y_true_multi,
        task_name="Sentiment Analysis Test",
        labels=['negative', 'neutral', 'positive'],
        average='weighted'
    )
    
    # Plot confusion matrix
    print("\n3️⃣ Generating visualizations...")
    eval_system.plot_confusion_matrix(
        confusion_matrix=np.array(results_multi['confusion_matrix']),
        labels=['negative', 'neutral', 'positive'],
        title="Sentiment Analysis Confusion Matrix",
        save_path=os.path.join(eval_system.results_dir, "confusion_matrix_test.png")
    )
    
    # Plot metrics
    metrics_to_plot = {
        'Accuracy': results_multi['accuracy'],
        'Precision': results_multi['precision'],
        'Recall': results_multi['recall'],
        'F1 Score': results_multi['f1_score']
    }
    
    eval_system.plot_metrics_comparison(
        metrics_dict=metrics_to_plot,
        title="Model Performance Metrics",
        save_path=os.path.join(eval_system.results_dir, "metrics_comparison_test.png")
    )
    
    # Generate final report
    print("\n4️⃣ Generating evaluation report...")
    report = eval_system.generate_evaluation_report(
        save_path=os.path.join(eval_system.results_dir, "evaluation_report.json")
    )
    
    print("\n✅ All tests completed!")
    print(f"\n💡 Results saved to: {eval_system.results_dir}")
