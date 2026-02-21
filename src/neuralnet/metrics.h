#pragma once
#include "core.h"
#include "neuralnet.h"

namespace torch {

namespace metrics {

    // =======================================================================
    //                        MULTICLASS CLASSIFICATION
    // =======================================================================

    // Returns raw count of correct predictions (useful for accumulating over batches)
    inline size_t accuracy(const Tensor& pred_logits, const Tensor& target) {
        Tensor p = pred_logits.argmax(1); // [B]
        Tensor t = target;
        // Ensure target is flat for accuracy to avoid broadcasting bugs
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        
        Tensor match = (p == t).astype(DType::Float32);
        return (size_t)::sum(match).item<float>();
    }

    // Returns accuracy as a ratio [0.0, 1.0]
    inline float accuracy_score(const Tensor& pred_logits, const Tensor& target) {
        size_t correct = accuracy(pred_logits, target);
        return (float)correct / (float)pred_logits.shape()[0];
    }

    // =======================================================================
    //                          BINARY CLASSIFICATION
    // =======================================================================

    // Returns binary accuracy as a ratio [0.0, 1.0]
    inline float binary_accuracy(const Tensor& preds, const Tensor& target, float threshold = 0.5f) {
        Tensor p = (preds >= threshold).astype(DType::Float32);
        Tensor t = target.astype(DType::Float32);
        
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        if (p.shape().size() > 1 && p.shape().back() == 1) p = p.flatten();
        
        Tensor match = (p == t).astype(DType::Float32);
        return ::sum(match).item<float>() / (float)p.shape()[0];
    }

    inline float precision_score(const Tensor& preds, const Tensor& target, float threshold = 0.5f) {
        Tensor p = (preds >= threshold).astype(DType::Float32);
        Tensor t = target.astype(DType::Float32);
        
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        if (p.shape().size() > 1 && p.shape().back() == 1) p = p.flatten();
        
        // True Positives: pred == 1 and target == 1
        float tp = ::sum(p * t).item<float>();
        // False Positives: pred == 1 and target == 0
        float fp = ::sum(p * (1.0 - t)).item<float>();
        
        if (tp + fp == 0.0f) return 0.0f;
        return tp / (tp + fp);
    }

    inline float recall_score(const Tensor& preds, const Tensor& target, float threshold = 0.5f) {
        Tensor p = (preds >= threshold).astype(DType::Float32);
        Tensor t = target.astype(DType::Float32);
        
        if (t.shape().size() > 1 && t.shape().back() == 1) t = t.flatten();
        if (p.shape().size() > 1 && p.shape().back() == 1) p = p.flatten();
        
        // True Positives: pred == 1 and target == 1
        float tp = ::sum(p * t).item<float>();
        // False Negatives: pred == 0 and target == 1
        float fn = ::sum((1.0 - p) * t).item<float>();
        
        if (tp + fn == 0.0f) return 0.0f;
        return tp / (tp + fn);
    }

    inline float f1_score(const Tensor& preds, const Tensor& target, float threshold = 0.5f) {
        float precision = precision_score(preds, target, threshold);
        float recall = recall_score(preds, target, threshold);
        
        if (precision + recall == 0.0f) return 0.0f;
        return 2.0f * (precision * recall) / (precision + recall);
    }

    // =======================================================================
    //                                REGRESSION
    // =======================================================================

    inline float mean_squared_error(const Tensor& preds, const Tensor& target) {
        Tensor diff = preds - target;
        return ::mean(diff * diff).item<float>();
    }

    inline float mean_absolute_error(const Tensor& preds, const Tensor& target) {
        Tensor diff = preds - target;
        return ::mean(::abs(diff)).item<float>();
    }

    // Coefficient of Determination (R^2 Score)
    inline float r2_score(const Tensor& preds, const Tensor& target) {
        float t_mean = ::mean(target).item<float>();
        
        // Total Sum of Squares
        Tensor diff_tot = target - t_mean;
        Tensor ss_tot = ::sum(diff_tot * diff_tot);
        
        // Residual Sum of Squares
        Tensor diff_res = target - preds;
        Tensor ss_res = ::sum(diff_res * diff_res);
        
        float tot = ss_tot.item<float>();
        if (tot == 0.0f) return 0.0f; // Avoid division by zero
        
        return 1.0f - (ss_res.item<float>() / tot);
    }

} 

} 

