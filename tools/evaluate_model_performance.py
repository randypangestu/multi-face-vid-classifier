#!/usr/bin/env python3
"""
Simple Face Anti-Spoofing Model Evaluation Script

Calculates key metrics: Accuracy, Precision, Recall, FAR, and FRR.
"""

import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def load_ground_truth_labels(labels_file):
    """Load ground truth labels from TSV file."""
    labels_dict = {}
    with open(labels_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if line:
                video, label = line.split('\t')
                labels_dict[video] = int(label)
    return labels_dict


def load_predictions(predictions_dir):
    """Load model predictions from JSON files."""
    predictions_dict = {}
    for filename in os.listdir(predictions_dir):
        if filename.endswith('_out.json'):
            video_name = filename.replace('_out.json', '')
            filepath = os.path.join(predictions_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                predictions_dict[video_name] = data['classification_binary_label']
    return predictions_dict


def calculate_metrics(y_true, y_pred):
    """Calculate key performance metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # FAR (False Acceptance Rate) = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FRR (False Rejection Rate) = FN / (FN + TP)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'far': far,
        'frr': frr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def get_subclass_config():
    """Define subclass categories for videos."""
    return {
        # Label 0 (Real/Live)
        'Person w/o IDs': ['veriff3', 'veriff7'],
        'ID only': ['veriff5'],
        'No face': ['veriff6', 'veriff14'],
        '1 Person w/ IDs': ['veriff1', 'veriff8', 'veriff11', 'veriff17', 'veriff18'],
        
        # Label 1 (Fake/Spoof)
        'Multiple Person w/o IDs': ['veriff2', 'veriff19','veriff16'],
        'Multiple Person Non-selfie, LQ': ['veriff4'],
        'Multiple Person w/ IDs': ['veriff9', 'veriff10', 'veriff12']
    }


def analyze_subclass_performance(ground_truth, predictions, common_videos):
    """Analyze performance for each subclass."""
    subclass_config = get_subclass_config()
    
    print("\n" + "=" * 50)
    print("SUBCLASS PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    for subclass, videos in subclass_config.items():
        # Filter videos that exist in our data
        available_videos = [v for v in videos if v in common_videos]
        
        if not available_videos:
            print(f"\n{subclass}: No videos available")
            continue
            
        # Get predictions and ground truth for this subclass
        y_true_sub = [ground_truth[v] for v in available_videos]
        y_pred_sub = [predictions[v] for v in available_videos]
        
        # Calculate accuracy for this subclass
        correct = sum(1 for gt, pred in zip(y_true_sub, y_pred_sub) if gt == pred)
        total = len(y_true_sub)
        accuracy = correct / total if total > 0 else 0
        
        # Show results
        print(f"\n{subclass}:")
        print(f"  Videos: {available_videos}")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) [{correct}/{total}]")
        
        # Show individual results
        for video, gt, pred in zip(available_videos, y_true_sub, y_pred_sub):
            status = "✓" if gt == pred else "✗"
            print(f"    {video}: GT={gt}, Pred={pred} {status}")


def print_results(metrics):
    """Print key evaluation metrics."""
    print("=" * 50)
    print("OVERALL PERFORMANCE RESULTS")
    print("=" * 50)
    
    print(f"Accuracy:   {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision:  {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:     {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"FAR:        {metrics['far']:.3f} ({metrics['far']*100:.1f}%)")
    print(f"FRR:        {metrics['frr']:.3f} ({metrics['frr']*100:.1f}%)")
    
    print(f"\nConfusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")


def main():
    """Main evaluation function."""
    labels_file = "/media/ranpang/Personal/Codes/Veriff/test_details/videos/labels2.txt"
    predictions_dir = "/media/ranpang/Personal/Codes/Veriff/repo/multi-face-video-classifier/data/output_test/rc6"
    
    # Load data
    ground_truth = load_ground_truth_labels(labels_file)
    predictions = load_predictions(predictions_dir)
    # Find common videos
    common_videos = set(ground_truth.keys()) & set(predictions.keys())
    
    if not common_videos:
        print("ERROR: No matching videos found!")
        return
    
    # Prepare labels
    y_true = [ground_truth[video] for video in sorted(common_videos)]
    y_pred = [predictions[video] for video in sorted(common_videos)]
    print('y_true:', y_true)
    print('y_pred:', y_pred)
    # Calculate and print overall metrics
    metrics = calculate_metrics(y_true, y_pred)
    print_results(metrics)
    
    # Analyze subclass performance
    analyze_subclass_performance(ground_truth, predictions, common_videos)


if __name__ == "__main__":
    main()
