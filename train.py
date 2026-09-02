import os
import sys
import json
import random
import torch
import numpy as np
import logging
import argparse
import math
import time
from collections import Counter
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader.dataset import (
    LAVABatchScaleBatchSampler, collate_fn, create_dataset)
from utils.train_utils import count_parameters
from models.model_runner import ModelFactory, VLAWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


class LossLogger:
    METRICS = (
        # Base objectives and global LAVA correspondence
        ("Loss_Flow", "loss_mse", ".6f"),
        ("Loss_Future", "loss_future_feat", ".6f"),
        ("Loss_LAVA", "loss_lava", ".6f"),
        ("Lambda_LAVA", "lambda_lava", ".6f"),
        ("Lambda_LAVA_Ratio", "lambda_lava_ratio", ".6f"),
        ("LAVA_Weight_Phase", "lava_weight_phase", "d"),
        ("Loss_Base", "loss_base", ".6f"),
        ("Weighted_LAVA", "weighted_lava", ".6f"),
        ("LAVA_Base_Ratio", "lava_base_ratio", ".6f"),
        ("Pos_Sim", "pos_sim", ".6f"),
        ("Negative_Sim", "negative_sim", ".6f"),
        ("Shuffle_Sim", "shuffle_sim", ".6f"),
        ("Order_Margin", "order_margin", ".6f"),
        ("Retrieval_Acc", "retrieval_acc", ".6f"),
        ("Action_Pair_Sim", "action_pair_sim", ".6f"),
        ("World_Pair_Sim", "world_pair_sim", ".6f"),
        ("Same_Task_Neg_Sim", "same_task_negative_sim", ".6f"),
        ("Cross_Task_Neg_Sim", "cross_task_negative_sim", ".6f"),
        ("Task_Shortcut_Gap", "task_shortcut_gap", ".6f"),
        ("Temporal_Neg_Sim", "temporal_negative_sim", ".6f"),
        ("Temporal_Margin", "temporal_margin", ".6f"),
        ("Temporal_Acc", "temporal_acc", ".6f"),
        ("Order_Acc", "order_acc", ".6f"),
        ("Candidate_Acc", "candidate_acc", ".6f"),
        ("Positive_Temporal_World_Sim", "positive_temporal_world_sim", ".6f"),
        ("Positive_Far_World_Sim", "positive_far_world_sim", ".6f"),
        ("Cross_Task_Family_Sim", "cross_task_family_sim", ".6f"),
        ("Cross_Task_Margin", "cross_task_margin", ".6f"),
        ("Cross_Task_Acc", "cross_task_acc", ".6f"),
        ("Far_Neg_Sim", "far_negative_sim", ".6f"),
        ("Far_Margin", "far_margin", ".6f"),
        ("Far_Acc", "far_acc", ".6f"),
        ("Block_Swap_Sim", "block_swap_sim", ".6f"),
        ("Block_Swap_Margin", "block_swap_margin", ".6f"),
        ("Block_Swap_Acc", "block_swap_acc", ".6f"),
        ("Derangement_Sim", "derangement_sim", ".6f"),
        ("Derangement_Margin", "derangement_margin", ".6f"),
        ("Derangement_Acc", "derangement_acc", ".6f"),
        ("Order_Family_Sim", "order_family_sim", ".6f"),
        ("Average_Negative_Family_Count", "average_negative_family_count", ".4f"),
        ("Hardest_Cross_Task_Fraction", "hardest_cross_task_fraction", ".6f"),
        ("Hardest_Local_Fraction", "hardest_local_fraction", ".6f"),
        ("Hardest_Far_Fraction", "hardest_far_fraction", ".6f"),
        ("Hardest_Order_Fraction", "hardest_order_fraction", ".6f"),
        ("Cross_Task_Candidate_Count", "cross_task_candidate_count", ".4f"),
        ("Order_Candidate_Count", "order_candidate_count", ".4f"),
        # V5 detached action-similarity weighting
        ("Arm_Action_Distance", "arm_action_distance", ".6f"),
        ("Gripper_State_Distance", "gripper_state_distance", ".6f"),
        ("Gripper_Change_Distance", "gripper_change_distance", ".6f"),
        ("Combined_Action_Distance", "combined_action_distance", ".6f"),
        *((f"Action_Distance_{family}_Median_S{scale}",
           f"action_distance_{family.lower()}_median_s{scale}", ".6f")
          for family in ("Cross", "Local", "Far")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Action_Distance_Aggregate_S{scale}",
           f"action_distance_aggregate_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Action_Distance_Beta_S{scale}",
           f"action_distance_beta_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"{family}_Neg_Weight_Mean", f"{key}_neg_weight_mean", ".6f")
          for family, key in (("CrossTask", "cross_task"),
                              ("Local", "local"), ("Far", "far"))),
        *((f"{family}_Neg_Weight_Below_0_5", f"{key}_neg_weight_below_0_5", ".6f")
          for family, key in (("CrossTask", "cross_task"),
                              ("Local", "local"), ("Far", "far"))),
        ("Effective_Negative_Mass", "effective_negative_mass", ".6f"),
        *((f"{kind}_{family}_Margin", f"{kind.lower()}_{key}_margin", ".6f")
          for kind in ("Raw", "Weighted")
          for family, key in (("CrossTask", "cross_task"),
                              ("Local", "local"), ("Far", "far"))),
        *((f"{kind}_{family}_Acc", f"{kind.lower()}_{key}_acc", ".6f")
          for kind in ("Raw", "Weighted")
          for family, key in (("CrossTask", "cross_task"),
                              ("Local", "local"), ("Far", "far"))),
        ("Raw_Candidate_Acc", "raw_candidate_acc", ".6f"),
        ("Weighted_Candidate_Acc", "weighted_candidate_acc", ".6f"),
        *((f"{kind}_Hardest_{family}_Fraction",
           f"{kind.lower()}_hardest_{key}_fraction", ".6f")
          for kind in ("Raw", "Weighted")
          for family, key in (("CrossTask", "cross_task"), ("Local", "local"),
                              ("Far", "far"), ("Order", "order"))),
        # Action execution and alignment-tap health (existing-forward reductions)
        ("Loss_Flow_Pos_0_7", "loss_flow_pos_0_7", ".6f"),
        ("Loss_Flow_Pos_8_15", "loss_flow_pos_8_15", ".6f"),
        ("Loss_Flow_Pos_16_31", "loss_flow_pos_16_31", ".6f"),
        ("Loss_Flow_Executed", "loss_flow_executed", ".6f"),
        ("Tap_Final_Cos", "tap_final_cos", ".6f"),
        ("Tap_Final_L2", "tap_final_l2", ".6f"),
        ("Tap_Hidden_Norm", "tap_hidden_norm", ".6f"),
        ("Tap_Hidden_Std", "tap_hidden_std", ".6f"),
        ("Final_Hidden_Norm", "final_hidden_norm", ".6f"),
        ("Final_Hidden_Std", "final_hidden_std", ".6f"),
        # Representation health
        ("Raw_Change_Norm", "raw_change_norm", ".6f"),
        ("Raw_Change_Std", "raw_change_std", ".6f"),
        ("Raw_Change_Norm_CV", "raw_change_norm_cv", ".6f"),
        ("InputNorm_Change_Norm", "input_norm_change_norm", ".6f"),
        ("InputNorm_Change_Norm_CV", "input_norm_change_norm_cv", ".6f"),
        ("Raw_InputNorm_Norm_Corr", "raw_input_norm_norm_corr", ".6f"),
        ("Raw_WorldResidual_Norm_Corr", "raw_world_residual_norm_corr", ".6f"),
        ("World_Residual_Norm", "world_residual_norm", ".6f"),
        ("World_Residual_Std", "world_residual_std", ".6f"),
        ("Action_Residual_Norm", "action_residual_norm", ".6f"),
        ("Action_Residual_Std", "action_residual_std", ".6f"),
        # Raw and EMA-calibrated LogSig geometry
        ("Action_LogSig_L1_Raw_Norm", "action_logsig_l1_raw_norm", ".6f"),
        ("Action_LogSig_L2_Raw_Norm", "action_logsig_l2_raw_norm", ".6f"),
        ("Action_LogSig_L2_L1_Ratio", "action_logsig_l2_l1_ratio", ".6f"),
        ("World_LogSig_L1_Raw_Norm", "world_logsig_l1_raw_norm", ".6f"),
        ("World_LogSig_L2_Raw_Norm", "world_logsig_l2_raw_norm", ".6f"),
        ("World_LogSig_L2_L1_Ratio", "world_logsig_l2_l1_ratio", ".6f"),
        ("Action_LogSig_L1_Calibrated_Norm", "action_logsig_l1_calibrated_norm", ".6f"),
        ("Action_LogSig_L2_Calibrated_Norm", "action_logsig_l2_calibrated_norm", ".6f"),
        ("Action_LogSig_L2_Energy_Fraction", "action_logsig_l2_energy_fraction", ".6f"),
        ("World_LogSig_L1_Calibrated_Norm", "world_logsig_l1_calibrated_norm", ".6f"),
        ("World_LogSig_L2_Calibrated_Norm", "world_logsig_l2_calibrated_norm", ".6f"),
        ("World_LogSig_L2_Energy_Fraction", "world_logsig_l2_energy_fraction", ".6f"),
        *((f"Action_LogSig_L2_L1_Ratio_S{scale}",
           f"action_logsig_l2_l1_ratio_s{scale}", ".6f")
          for scale in (2, 4, 8, 16)),
        *((f"World_LogSig_L2_L1_Ratio_S{scale}",
           f"world_logsig_l2_l1_ratio_s{scale}", ".6f")
          for scale in (2, 4, 8, 16)),
        # Per-scale LAVA behavior
        *((f"Loss_S{scale}", f"loss_s{scale}", ".6f") for scale in (1, 2, 4, 8, 16)),
        *((f"Pos_Sim_S{scale}", f"pos_sim_s{scale}", ".6f") for scale in (1, 2, 4, 8, 16)),
        *((f"Order_Margin_S{scale}", f"order_margin_s{scale}", ".6f") for scale in (1, 2, 4, 8, 16)),
        *((f"Temporal_Neg_Sim_S{scale}", f"temporal_negative_sim_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Temporal_Margin_S{scale}", f"temporal_margin_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Temporal_Acc_S{scale}", f"temporal_acc_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Order_Acc_S{scale}", f"order_acc_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Candidate_Acc_S{scale}", f"candidate_acc_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Far_Margin_S{scale}", f"far_margin_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Cross_Task_Margin_S{scale}", f"cross_task_margin_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Block_Swap_Margin_S{scale}", f"block_swap_margin_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Derangement_Margin_S{scale}", f"derangement_margin_s{scale}", ".6f")
          for scale in (1, 2, 4, 8, 16)),
        *((f"Action_LogSig_L2_Energy_Fraction_S{scale}",
           f"action_logsig_l2_energy_fraction_s{scale}", ".6f")
          for scale in (2, 4, 8, 16)),
        *((f"World_LogSig_L2_Energy_Fraction_S{scale}",
           f"world_logsig_l2_energy_fraction_s{scale}", ".6f")
          for scale in (2, 4, 8, 16)),
        *((f"{side}_LogSig_L{level}_EMA_RMS_S{scale}",
           f"{side.lower()}_logsig_l{level}_ema_rms_s{scale}", ".6f")
          for side in ("Action", "World") for level in (1, 2)
          for scale in (2, 4, 8, 16)),
        # Flow-timestep-conditioned LAVA behavior
        ("LAVA_T_Mean", "lava_t_mean", ".6f"),
        *((f"Loss_{label}", f"loss_{key}", ".6f") for label, key in (
            ("T0_025", "t0_025"), ("T025_050", "t025_050"),
            ("T050_075", "t050_075"), ("T075_100", "t075_100"))),
        *((f"Pos_Sim_{label}", f"pos_sim_{key}", ".6f") for label, key in (
            ("T0_025", "t0_025"), ("T025_050", "t025_050"),
            ("T050_075", "t050_075"), ("T075_100", "t075_100"))),
        *((f"Count_{label}", f"count_{key}", "d") for label, key in (
            ("T0_025", "t0_025"), ("T025_050", "t025_050"),
            ("T050_075", "t050_075"), ("T075_100", "t075_100"))),
        # Sampling, optimization, and system health
        ("LAVA_Samples", "lava_sample_count", "d"),
        ("Order_Negatives", "lava_order_negative_count", "d"),
        ("LAVA_Coverage_Pos_1_8", "lava_coverage_pos_1_8", ".6f"),
        ("LAVA_Coverage_Pos_9_16", "lava_coverage_pos_9_16", ".6f"),
        ("LAVA_Coverage_Pos_17_24", "lava_coverage_pos_17_24", ".6f"),
        ("LAVA_Coverage_Pos_25_31", "lava_coverage_pos_25_31", ".6f"),
        ("LAVA_Position_Mean", "lava_position_mean", ".4f"),
        ("LAVA_Position_Min", "lava_position_min", ".0f"),
        ("LAVA_Position_Max", "lava_position_max", ".0f"),
        ("LAVA_Executed_Horizon_Ratio", "lava_executed_horizon_ratio", ".6f"),
        ("LAVA_Executed_Path_Ratio", "lava_executed_path_ratio", ".6f"),
        ("Loss_LAVA_Executed", "loss_lava_executed", ".6f"),
        ("Loss_LAVA_Tail", "loss_lava_tail", ".6f"),
        ("Pos_Sim_Executed", "pos_sim_executed", ".6f"),
        ("Pos_Sim_Tail", "pos_sim_tail", ".6f"),
        ("Candidate_Acc_Executed", "candidate_acc_executed", ".6f"),
        ("Candidate_Acc_Tail", "candidate_acc_tail", ".6f"),
        ("Local_Margin_Executed", "local_margin_executed", ".6f"),
        ("Local_Margin_Tail", "local_margin_tail", ".6f"),
        ("Order_Margin_Executed", "order_margin_executed", ".6f"),
        ("Order_Margin_Tail", "order_margin_tail", ".6f"),
        ("Negative_Distance_Over_L", "negative_distance_over_l", ".6f"),
        ("Far_Negative_Distance_Over_L", "far_negative_distance_over_l", ".6f"),
        ("Local_Negative_Fallback_Rate", "local_negative_fallback_rate", ".6f"),
        ("Dropped_Pair_Rate", "dropped_pair_rate", ".6f"),
        ("Scale_Mean", "lava_scale_mean", ".4f"),
        ("Scale_Min", "lava_scale_min", ".0f"),
        ("Scale_Max", "lava_scale_max", ".0f"),
        *((f"Scale_{scale}_Count", f"lava_scale_{scale}_count", "d")
          for scale in (1, 2, 4, 8, 16)),
        ("LR", "learning_rate", ".8e"),
        ("Grad_Norm", "grad_norm", ".6f"),
        ("Grad_Norm_LAVA_Branch", "lava_branch_grad_norm", ".6f"),
        ("Grad_Cos_Shared", "grad_cos_shared", ".6f"),
        ("Grad_Norm_Base_Shared", "grad_norm_base_shared", ".6f"),
        ("Grad_Norm_LAVA_Shared", "grad_norm_lava_shared", ".6f"),
        ("Weighted_Grad_Ratio", "weighted_grad_ratio", ".6f"),
        *((f"Grad_Cos_{label}", f"grad_cos_{key}", ".6f")
          for label, key in (("Input", "input"), ("B1_4", "b1_4"),
                             ("B5_8", "b5_8"), ("B9_10", "b9_10"),
                             ("B11_12", "b11_12"))),
        *((f"Grad_Norm_Base_{label}", f"grad_norm_base_{key}", ".6f")
          for label, key in (("Input", "input"), ("B1_4", "b1_4"),
                             ("B5_8", "b5_8"), ("B9_10", "b9_10"),
                             ("B11_12", "b11_12"))),
        *((f"Grad_Norm_LAVA_{label}", f"grad_norm_lava_{key}", ".6f")
          for label, key in (("Input", "input"), ("B1_4", "b1_4"),
                             ("B5_8", "b5_8"), ("B9_10", "b9_10"),
                             ("B11_12", "b11_12"))),
        *((f"Weighted_Grad_Ratio_{label}", f"weighted_grad_ratio_{key}", ".6f")
          for label, key in (("Input", "input"), ("B1_4", "b1_4"),
                             ("B5_8", "b5_8"), ("B9_10", "b9_10"),
                             ("B11_12", "b11_12"))),
        ("Update_Time_s", "update_time_s", ".4f"),
        ("Data_Time_s", "data_time_s", ".4f"),
        ("GPU_Peak_Allocated_GB", "gpu_peak_allocated_gb", ".4f"),
        ("GPU_Peak_Reserved_GB", "gpu_peak_reserved_gb", ".4f"),
    )

    NAN_DEFAULT_KEYS = {
        "same_task_negative_sim", "cross_task_negative_sim", "task_shortcut_gap",
        "temporal_negative_sim", "temporal_margin", "temporal_acc",
        "order_acc", "candidate_acc", "positive_temporal_world_sim",
        "positive_far_world_sim", "cross_task_family_sim", "cross_task_margin",
        "cross_task_acc", "far_negative_sim", "far_margin", "far_acc",
        "block_swap_sim", "block_swap_margin", "block_swap_acc",
        "derangement_sim", "derangement_margin", "derangement_acc",
        "order_family_sim", "average_negative_family_count",
        "hardest_cross_task_fraction", "hardest_local_fraction",
        "hardest_far_fraction", "hardest_order_fraction",
        "negative_distance_over_l", "local_negative_fallback_rate",
        "far_negative_distance_over_l", "dropped_pair_rate",
        "loss_lava_executed", "loss_lava_tail", "pos_sim_executed",
        "pos_sim_tail", "candidate_acc_executed", "candidate_acc_tail",
        "local_margin_executed", "local_margin_tail",
        "order_margin_executed", "order_margin_tail",
        "arm_action_distance", "gripper_state_distance",
        "gripper_change_distance", "combined_action_distance",
        "cross_task_neg_weight_mean", "local_neg_weight_mean",
        "far_neg_weight_mean", "cross_task_neg_weight_below_0_5",
        "local_neg_weight_below_0_5", "far_neg_weight_below_0_5",
        "effective_negative_mass", "raw_candidate_acc", "weighted_candidate_acc",
        *(f"action_distance_{family}_median_s{scale}"
          for family in ("cross", "local", "far") for scale in (1, 2, 4, 8, 16)),
        *(f"action_distance_aggregate_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"action_distance_beta_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"{kind}_{family}_{metric}"
          for kind in ("raw", "weighted")
          for family in ("cross_task", "local", "far") for metric in ("margin", "acc")),
        *(f"{kind}_hardest_{family}_fraction"
          for kind in ("raw", "weighted")
          for family in ("cross_task", "local", "far", "order")),
        "raw_input_norm_norm_corr", "raw_world_residual_norm_corr",
        "grad_cos_shared", "grad_norm_base_shared", "grad_norm_lava_shared",
        "weighted_grad_ratio",
        *(f"grad_cos_{key}" for key in ("input", "b1_4", "b5_8", "b9_10", "b11_12")),
        *(f"grad_norm_base_{key}" for key in ("input", "b1_4", "b5_8", "b9_10", "b11_12")),
        *(f"grad_norm_lava_{key}" for key in ("input", "b1_4", "b5_8", "b9_10", "b11_12")),
        *(f"weighted_grad_ratio_{key}" for key in ("input", "b1_4", "b5_8", "b9_10", "b11_12")),
        *(f"action_logsig_l2_l1_ratio_s{scale}" for scale in (2, 4, 8, 16)),
        *(f"world_logsig_l2_l1_ratio_s{scale}" for scale in (2, 4, 8, 16)),
        *(f"temporal_negative_sim_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"temporal_margin_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"temporal_acc_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"order_acc_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"candidate_acc_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"far_margin_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"cross_task_margin_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"block_swap_margin_s{scale}" for scale in (1, 2, 4, 8, 16)),
        *(f"derangement_margin_s{scale}" for scale in (1, 2, 4, 8, 16)),
    }

    def __init__(self, log_dir="log/loss"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"train_loss_{timestamp}.csv")
        with open(self.log_file, 'w') as f:
            headers = ["Epoch", "Step", "Global_Step", "Loss"]
            headers.extend(header for header, _, _ in self.METRICS)
            f.write(",".join(headers) + "\n")

    def log(self, epoch, step, global_step, loss, info):
        values = [str(epoch), str(step), str(global_step), f"{loss:.6f}"]
        for _, key, format_spec in self.METRICS:
            default = float("nan") if key in self.NAN_DEFAULT_KEYS else 0
            value = info.get(key, default)
            values.append(format(int(value), format_spec) if format_spec == "d"
                          else format(float(value), format_spec))
        with open(self.log_file, 'a') as f:
            f.write(",".join(values) + "\n")


def build_train_config_from_yaml(cfg):
    """Extract the training-parameter dict needed by VLAWrapper from OmegaConf cfg.training"""
    t = cfg.training
    ff = cfg.model.get('future_feat', {})
    lava = cfg.model.get('lava', {})
    return {
        'time_mu': t.time_mu,
        'time_sigma': t.time_sigma,
        'use_vel_weight': t.use_vel_weight,
        'vel_weight_alpha': t.vel_weight_alpha,
        'vel_weight_sigma': t.vel_weight_sigma,
        'use_future_feat': ff.get('enabled', False) if ff else False,
        'lambda_future_feat': t.get('lambda_future_feat', 0.0),
        'use_lava': lava.get('enabled', False) if lava else False,
        'lambda_lava': t.get('lambda_lava', 0.0),
        'lava_temperature': t.get('lava_temperature', 0.07),
        'lava_order_negative': t.get('lava_order_negative', True),
        'lava_negative_mode': t.get('lava_negative_mode', 'batch'),
        'lava_action_similarity_weighting': t.get(
            'lava_action_similarity_weighting', False),
        'action_execution_horizon': cfg.common.get('action_execution_horizon', 16),
    }


def parameter_grad_norm(parameters):
    """L2 norm over existing parameter gradients, computed before clipping."""
    squared_norms = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in parameters if parameter.grad is not None
    ]
    if not squared_norms:
        return 0.0
    return torch.stack(squared_norms).sum().sqrt().item()


def should_run_lava_grad_diagnostics(next_global_step, is_update_boundary,
                                     interval):
    return (is_update_boundary and interval > 0
            and next_global_step % interval == 0)


def compute_lava_weight(next_global_step, peak_weight, warmup_steps,
                        schedule="constant", min_weight=0.0,
                        decay_start_step=None, total_steps=None):
    """Return the resume-safe LAVA weight and its numeric schedule phase.

    Phases are 0=warmup, 1=peak/constant, and 2=reduced.  The function uses
    optimizer-step indices, so restoring ``global_step`` also restores the
    exact weight schedule without storing additional mutable state.
    """
    if next_global_step <= 0:
        raise ValueError(
            f"next_global_step must be positive, got {next_global_step}")
    if peak_weight < 0.0:
        raise ValueError(f"peak_weight must be non-negative, got {peak_weight}")
    if schedule not in {"constant", "step", "cosine"}:
        raise ValueError(
            "lava_weight_schedule must be constant, step, or cosine, "
            f"got {schedule}")

    if warmup_steps > 0 and next_global_step <= warmup_steps:
        return (
            peak_weight * float(next_global_step) / float(warmup_steps),
            0,
        )

    if schedule == "step":
        if decay_start_step is None or decay_start_step <= 0:
            raise ValueError(
                "decay_start_step must be positive for the step schedule")
        if next_global_step > decay_start_step:
            return min_weight, 2

    if schedule == "cosine":
        if decay_start_step is None or decay_start_step < 0:
            raise ValueError(
                "decay_start_step must be non-negative for the cosine schedule")
        if total_steps is None or total_steps <= decay_start_step:
            raise ValueError(
                "total_steps must be greater than decay_start_step for the "
                "cosine schedule")
        if next_global_step > total_steps:
            return min_weight, 2
        if next_global_step > decay_start_step:
            progress = (
                float(next_global_step - decay_start_step)
                / float(total_steps - decay_start_step)
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                min_weight + (peak_weight - min_weight) * cosine_factor,
                2,
            )

    return peak_weight, 1


def build_lava_gradient_parameter_groups(model):
    """Select fixed architectural bands for localized conflict diagnostics."""
    named_parameters = list(model.named_parameters())

    def matching(prefixes):
        return [parameter for name, parameter in named_parameters
                if parameter.requires_grad and name.startswith(prefixes)]

    return {
        "input": matching((
            "time_mlp.", "action_proj.", "proprio_proj.",
            "dino_adapters.", "concat_fusion.", "task_cond_proj.",
            "type_emb_", "register_tokens")),
        "b1_4": matching(tuple(f"blocks.{index}." for index in range(0, 4))),
        "b5_8": matching(tuple(f"blocks.{index}." for index in range(4, 8))),
        "b9_10": matching(tuple(f"blocks.{index}." for index in range(8, 10))),
        "b11_12": matching(tuple(f"blocks.{index}." for index in range(10, 12))),
    }


def compute_shared_gradient_diagnostics(loss_base, loss_lava, parameters,
                                        lava_weight, eps=1e-12,
                                        parameter_groups=None):
    """Compare base/LAVA gradients with one pair of autograd.grad calls.

    The global result keeps the original shared-parameter definition. Optional
    groups reuse those same gradient tensors, so localized diagnostics add no
    backward traversals.
    """
    nan_result = {
        "grad_cos_shared": float("nan"),
        "grad_norm_base_shared": float("nan"),
        "grad_norm_lava_shared": float("nan"),
        "weighted_grad_ratio": float("nan"),
    }
    if (loss_base is None or loss_lava is None
            or not loss_base.requires_grad or not loss_lava.requires_grad):
        return nan_result

    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    base_grads = torch.autograd.grad(
        loss_base, parameters, retain_graph=True, allow_unused=True)
    lava_grads = torch.autograd.grad(
        loss_lava, parameters, retain_graph=True, allow_unused=True)

    dot = None
    base_squared = None
    lava_squared = None
    for base_grad, lava_grad in zip(base_grads, lava_grads):
        if base_grad is None or lava_grad is None:
            continue
        base_float = base_grad.detach().float()
        lava_float = lava_grad.detach().float()
        current_dot = (base_float * lava_float).sum()
        current_base_squared = base_float.square().sum()
        current_lava_squared = lava_float.square().sum()
        dot = current_dot if dot is None else dot + current_dot
        base_squared = (current_base_squared if base_squared is None
                        else base_squared + current_base_squared)
        lava_squared = (current_lava_squared if lava_squared is None
                        else lava_squared + current_lava_squared)

    if dot is None:
        result = dict(nan_result)
    else:
        base_norm = base_squared.sqrt()
        lava_norm = lava_squared.sqrt()
        denominator = base_norm * lava_norm
        grad_cos = (dot / denominator).item() if denominator > eps else float("nan")
        weighted_ratio = (
            abs(float(lava_weight)) * lava_norm / base_norm).item() \
            if base_norm > eps else float("nan")
        result = {
            "grad_cos_shared": grad_cos,
            "grad_norm_base_shared": base_norm.item(),
            "grad_norm_lava_shared": lava_norm.item(),
            "weighted_grad_ratio": weighted_ratio,
        }

    if not parameter_groups:
        return result

    parameter_to_index = {id(parameter): index
                          for index, parameter in enumerate(parameters)}
    for group_name, group_parameters in parameter_groups.items():
        indices = [parameter_to_index[id(parameter)] for parameter in group_parameters
                   if id(parameter) in parameter_to_index]
        group_base_squared = None
        group_lava_squared = None
        group_dot = None
        for index in indices:
            base_grad, lava_grad = base_grads[index], lava_grads[index]
            if base_grad is not None:
                value = base_grad.detach().float().square().sum()
                group_base_squared = value if group_base_squared is None else group_base_squared + value
            if lava_grad is not None:
                value = lava_grad.detach().float().square().sum()
                group_lava_squared = value if group_lava_squared is None else group_lava_squared + value
            if base_grad is not None and lava_grad is not None:
                value = (base_grad.detach().float() * lava_grad.detach().float()).sum()
                group_dot = value if group_dot is None else group_dot + value

        base_norm = (group_base_squared.sqrt() if group_base_squared is not None
                     else loss_base.new_tensor(0.0))
        lava_norm = (group_lava_squared.sqrt() if group_lava_squared is not None
                     else loss_base.new_tensor(0.0))
        denominator = base_norm * lava_norm
        result[f"grad_cos_{group_name}"] = (
            (group_dot / denominator).item()
            if group_dot is not None and denominator > eps else float("nan"))
        result[f"grad_norm_base_{group_name}"] = base_norm.item()
        result[f"grad_norm_lava_{group_name}"] = lava_norm.item()
        result[f"weighted_grad_ratio_{group_name}"] = (
            (abs(float(lava_weight)) * lava_norm / base_norm).item()
            if base_norm > eps else float("nan"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Training Script (DINOv3) - two-stage LR version")
    parser.add_argument("--config", type=str, default="./configs/robotwin_all.yaml",
                        help="Path to config file")
    parser.add_argument("--norm_stats_path", type=str, 
                        default="./utils/stat-500-all.json",
                        help="Path to normalization stats")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from after an interruption "
                             "(restores model + optimizer + scheduler + epoch, continues the "
                             "lr schedule saved in the checkpoint)")
    parser.add_argument("--init_from", type=str,
                        default=None,
                        help="Path to checkpoint to initialize model weights from "
                             "Only model weights "
                             "are loaded; optimizer and lr scheduler start fresh from --config. "
                             "Mutually exclusive with --resume.")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
        help="OmegaConf dot-list overrides, e.g. --set training.batch_size=8 training.epochs=1",
    )
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional optimizer-step limit, useful for smoke tests")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not write checkpoints (useful for smoke tests)")
    args = parser.parse_args()

    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive: "
                         "--resume continues an interrupted run (inherits its lr schedule), "
                         "--init_from starts a new stage with a fresh lr schedule.")

    # =========================================================================
    # 1. Environment / Config
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device: {device}, Precision: {dtype}")

    config = load_config(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # Training hyperparameters
    seed = int(config.training.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    epochs = config.training.epochs
    grad_accum_steps = config.training.grad_accum_steps
    save_interval_epoch = config.training.save_interval_epoch
    checkpoint_tag = str(config.training.get('checkpoint_tag', '')).strip()
    batch_size = config.training.batch_size
    grad_clip_norm = config.training.grad_clip_norm
    lr = config.training.learning_rate
    lr_min = config.training.lr_min

    if lr_min > lr:
        logger.warning(f"lr_min ({lr_min}) > learning_rate ({lr}): "
                       f"CosineAnnealingLR will RAISE the lr towards lr_min instead of decaying. "
                       f"Check your config.")

    logger.info(f"Seed: {seed}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | "
                f"grad_accum: {grad_accum_steps} | save_every: {save_interval_epoch} ep")
    logger.info(f"LR schedule: CosineAnnealingLR {lr:.2e} -> {lr_min:.2e}")
    logger.info(f"DINOv3 feat_layers: {list(config.model.vision_encoder.feat_layers)} | "
                f"include_cls_register: {config.model.vision_encoder.include_cls_register}")
    logger.info(f"Vel-Weight: {'ENABLED' if config.training.use_vel_weight else 'DISABLED'}")
    lava_cfg = config.model.get('lava', {})
    use_lava = bool(lava_cfg.get('enabled', False)) if lava_cfg else False
    lambda_lava_max = float(config.training.get('lambda_lava', 0.0))
    lava_warmup_ratio = float(config.training.get('lava_warmup_ratio', 0.0))
    lava_weight_schedule = str(
        config.training.get('lava_weight_schedule', 'constant')).lower()
    lava_min_weight = float(config.training.get('lava_min_weight', 0.0))
    lava_decay_start_ratio = float(
        config.training.get('lava_decay_start_ratio', 1.0))
    lava_grad_diagnostics_interval = int(
        config.training.get('lava_grad_diagnostics_interval', 0))
    if not 0.0 <= lava_warmup_ratio <= 1.0:
        raise ValueError(f"lava_warmup_ratio must be in [0,1], got {lava_warmup_ratio}")
    if lava_grad_diagnostics_interval < 0:
        raise ValueError(
            "lava_grad_diagnostics_interval must be non-negative, got "
            f"{lava_grad_diagnostics_interval}")
    if lava_weight_schedule not in {'constant', 'step', 'cosine'}:
        raise ValueError(
            "lava_weight_schedule must be constant, step, or cosine, got "
            f"{lava_weight_schedule}")
    if not 0.0 <= lava_min_weight <= lambda_lava_max:
        raise ValueError(
            "lava_min_weight must be in [0, lambda_lava], got "
            f"{lava_min_weight} for lambda_lava={lambda_lava_max}")
    if not 0.0 <= lava_decay_start_ratio <= 1.0:
        raise ValueError(
            "lava_decay_start_ratio must be in [0,1], got "
            f"{lava_decay_start_ratio}")
    if (lava_weight_schedule == 'step'
            and lava_decay_start_ratio <= lava_warmup_ratio):
        raise ValueError(
            "step schedule must begin after warmup: "
            f"decay_start_ratio={lava_decay_start_ratio}, "
            f"warmup_ratio={lava_warmup_ratio}")
    if (lava_weight_schedule == 'cosine'
            and lava_decay_start_ratio < lava_warmup_ratio):
        raise ValueError(
            "cosine decay must begin at or after warmup: "
            f"decay_start_ratio={lava_decay_start_ratio}, "
            f"warmup_ratio={lava_warmup_ratio}")
    if use_lava:
        logger.info(
            f"LAVA: lambda={lambda_lava_max}, warmup_ratio={lava_warmup_ratio}, "
            f"weight_schedule={lava_weight_schedule}, "
            f"min_weight={lava_min_weight}, "
            f"decay_start_ratio={lava_decay_start_ratio}, "
            f"action_tap={lava_cfg.get('action_target_layer', 'final')}, "
            f"temperature={config.training.lava_temperature}, "
            f"scales={list(config.training.lava_scales)}, "
            f"sample_ratio={config.training.lava_sample_ratio}, "
            f"sampling_balance={config.training.get('lava_sampling_balance', 'none')}, "
            f"negative_mode={config.training.get('lava_negative_mode', 'batch')}, "
            f"negative_radius=min("
            f"{config.training.get('lava_negative_window_multiplier', 4)}L, "
            f"{config.training.get('lava_negative_window_max', 32)}), "
            f"order_negative={config.training.lava_order_negative}, "
            f"grad_diagnostics_every={lava_grad_diagnostics_interval} steps")

    # =========================================================================
    # 2. Dataset / DataLoader
    # =========================================================================
    train_dataset = create_dataset(config, val=False)

    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    lava_batch_sampler = None
    if (use_lava
            and str(config.training.get('lava_scale_sampling', 'uniform'))
            == 'batch_uniform'):
        lava_batch_sampler = LAVABatchScaleBatchSampler(
            len(train_dataset), batch_size,
            tuple(config.training.lava_scales), seed=seed, drop_last=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=lava_batch_sampler,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory,
            collate_fn=collate_fn,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
            generator=data_generator,
        )
    logger.info(f"Dataset Size: {len(train_dataset)} | Batches per Epoch: {len(train_dataloader)}")

    # =========================================================================
    # 3. Model
    # =========================================================================
    logger.info(">>> Initializing VLA (DINOv3)")

    vision_encoder, dino_hidden_size, num_register_tokens, patch_size = ModelFactory.create_vision_encoder(
        config.model.vision_encoder.checkpoint_path,
        dtype, device,
    )

    feat_layers = list(config.model.vision_encoder.feat_layers)
    num_dino_layers = len(feat_layers)

    # Task condition vector dim (uses DINOv3 hidden_size; controlled by use_task_cond)
    use_task_cond = config.model.get('use_task_cond', False)
    task_cond_dim = dino_hidden_size if use_task_cond else None
    logger.info(f"Task Condition: {'ENABLED' if use_task_cond else 'DISABLED'}"
                + (f" (dim={task_cond_dim}, dir={config.dataset.get('task_cond_dir', None)})" if use_task_cond else ""))

    # Future feature prediction config (logging)
    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    logger.info(f"Future-Feat Pred: {'ENABLED' if use_future_feat else 'DISABLED'}"
                + (f" (target_layer={ff_cfg.get('target_layer', -1)}, "
                   f"lambda={config.training.get('lambda_future_feat', 0.0)})" if use_future_feat else ""))

    action_model = ModelFactory.create_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=num_dino_layers,
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
    )

    action_model.to(device, dtype=dtype)
    action_model.train()

    count_parameters(action_model, model_name="Action Model (Trainable)")

    train_config_dict = build_train_config_from_yaml(config)

    model = VLAWrapper(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=num_register_tokens,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        norm_stats_key=config.dataset.get('norm_stats_key', 'robotwin2'),
        train_config=train_config_dict,
        future_feat_target_layer=ff_cfg.get('target_layer', -1) if ff_cfg else -1,
        lava_target_layer=lava_cfg.get('dino_target_layer', -4) if lava_cfg else -4,
        vision_encode_batch_size=config.model.vision_encoder.get('encode_batch_size', None),
    )

    # =========================================================================
    # 4. Optimizer / Scheduler
    # =========================================================================
    optimizer = AdamW(
        action_model.parameters(),
        lr=lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )
    lava_branch_parameters = [
        parameter for name, parameter in action_model.named_parameters()
        if name.startswith(("lava_world_encoder", "lava_action_projector"))
    ]
    if use_lava and not lava_branch_parameters:
        raise RuntimeError("LAVA is enabled but no lava_* trainable parameters were found")
    lava_gradient_parameter_groups = build_lava_gradient_parameter_groups(action_model)

    optimizer_steps_per_epoch = max(1, len(train_dataloader) // grad_accum_steps)
    total_optimizer_steps = max(1, epochs * optimizer_steps_per_epoch)
    lava_warmup_steps = int(round(total_optimizer_steps * lava_warmup_ratio))
    lava_decay_start_step = int(round(
        total_optimizer_steps * lava_decay_start_ratio))

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_optimizer_steps,
        eta_min=lr_min,
    )
    if use_lava:
        logger.info(
            f"LAVA lambda schedule: warmup={lava_warmup_steps}, "
            f"decay_start={lava_decay_start_step}, "
            f"total={total_optimizer_steps} optimizer steps")

    # =========================================================================
    # 5. Load checkpoint: --init_from (new stage) or --resume (interrupted run)
    # =========================================================================
    start_epoch = 0
    global_step = 0
    previous_lava_weight_phase = None

    if args.init_from:
        # New training stage (e.g. stage-2 low-lr phase): load ONLY model weights.
        # Optimizer/scheduler stay fresh so the lr schedule restarts from --config.
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_from}")

        logger.info(f">>> Initializing weights from {args.init_from} "
                    f"(fresh optimizer + lr schedule from config)")
        ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)

        msg = action_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.info(f"Model loaded (init_from epoch={ckpt.get('epoch', '?')}). "
                    f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        logger.info(f"Starting new stage: lr={lr:.2e} -> {lr_min:.2e}, epochs={epochs}")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info(f">>> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

        msg = action_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.info(f"Model loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            steps_done = ckpt['epoch'] * len(train_dataloader)
            for _ in range(steps_done):
                scheduler.step()
            logger.warning("Old checkpoint without scheduler_state_dict; "
                           "scheduler advanced manually (lr may drift slightly).")

        start_epoch = ckpt['epoch']
        global_step = ckpt.get('global_step',
                               start_epoch * len(train_dataloader) // grad_accum_steps)
        if lava_batch_sampler is not None:
            sampler_state = ckpt.get('lava_batch_sampler_state')
            if sampler_state is None:
                raise KeyError(
                    "V5 batch_uniform resume requires lava_batch_sampler_state")
            lava_batch_sampler.load_state_dict(sampler_state)
        logger.info(f"Resumed at epoch={start_epoch}, global_step={global_step}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_save_dir = os.path.join(args.save_dir, f"sft_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)

    OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    loss_logger = LossLogger(log_dir=os.path.join(run_save_dir, "log"))

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        batches_seen = 0
        optimizer.zero_grad()
        start_time = time.time()
        last_iteration_end = start_time
        update_window_start = start_time
        update_window_data_time = 0.0
        update_window_scales = []
        epoch_lava_task_counts = Counter()
        epoch_action_similarity_audit = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_dataloader):
            batch_ready_time = time.time()
            update_window_data_time += batch_ready_time - last_iteration_end
            if batch.get('evolution_scales') is not None:
                update_window_scales.extend(batch['evolution_scales'].tolist())
            if (batch.get('evolution_batch_indices') is not None
                    and batch.get('task_name') is not None):
                for batch_index in batch['evolution_batch_indices'].tolist():
                    epoch_lava_task_counts[str(batch['task_name'][batch_index])] += 1
            lava_weight, lava_weight_phase = compute_lava_weight(
                global_step + 1,
                lambda_lava_max,
                lava_warmup_steps,
                schedule=lava_weight_schedule,
                min_weight=lava_min_weight,
                decay_start_step=lava_decay_start_step,
                total_steps=total_optimizer_steps,
            )
            if lava_weight_phase != previous_lava_weight_phase:
                logger.info(
                    "LAVA weight phase=%d at optimizer step %d: lambda=%.6f",
                    lava_weight_phase, global_step + 1, lava_weight)
                previous_lava_weight_phase = lava_weight_phase
            with torch.amp.autocast('cuda', dtype=dtype):
                loss, info_dic = model(batch, lava_weight=lava_weight)
                loss = loss / grad_accum_steps

            for record in info_dic.pop('_action_similarity_audit', []):
                task_name = str(record['task'])
                scale = int(record['scale'])
                for family in ('cross', 'local', 'far'):
                    value = record.get(family)
                    if value is None:
                        continue
                    key = (task_name, scale, family)
                    aggregate = epoch_action_similarity_audit.setdefault(
                        key, {'count': 0, 'distance_sum': 0.0,
                              'weight_sum': 0.0, 'below_half': 0,
                              'arm_sum': 0.0, 'gripper_state_sum': 0.0,
                              'gripper_change_sum': 0.0})
                    aggregate['count'] += 1
                    aggregate['distance_sum'] += float(value['distance'])
                    aggregate['weight_sum'] += float(value['weight'])
                    aggregate['below_half'] += int(value['weight'] < 0.5)
                    for component in ('arm', 'gripper_state', 'gripper_change'):
                        aggregate[f'{component}_sum'] += float(value[component])

            loss_base_tensor = info_dic.pop('_loss_base_tensor', None)
            loss_lava_tensor = info_dic.pop('_loss_lava_tensor', None)
            is_update_boundary = (step + 1) % grad_accum_steps == 0
            run_grad_diagnostics = (
                use_lava
                and info_dic.get('lava_sample_count', 0) > 0
                and should_run_lava_grad_diagnostics(
                    global_step + 1, is_update_boundary,
                    lava_grad_diagnostics_interval))
            grad_diagnostics = {
                'grad_cos_shared': float('nan'),
                'grad_norm_base_shared': float('nan'),
                'grad_norm_lava_shared': float('nan'),
                'weighted_grad_ratio': float('nan'),
            }
            if run_grad_diagnostics:
                grad_diagnostics = compute_shared_gradient_diagnostics(
                    loss_base_tensor, loss_lava_tensor,
                    action_model.parameters(), lava_weight,
                    parameter_groups=lava_gradient_parameter_groups)

            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch + 1}, step={step + 1}: {info_dic}")

            loss.backward()

            current_step_loss = loss.item() * grad_accum_steps
            epoch_loss += current_step_loss
            batches_seen += 1

            if (step + 1) % grad_accum_steps == 0:
                lava_branch_grad_norm = parameter_grad_norm(lava_branch_parameters)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    action_model.parameters(), max_norm=grad_clip_norm)
                grad_norm = float(grad_norm_tensor.detach().float().item())
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch + 1}, step={step + 1}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                scale_mean = float(np.mean(update_window_scales)) if update_window_scales else 0.0
                scale_min = float(min(update_window_scales)) if update_window_scales else 0.0
                scale_max = float(max(update_window_scales)) if update_window_scales else 0.0
                runtime_info = dict(info_dic)
                runtime_info.update({
                    'lambda_lava_ratio': (
                        lava_weight / lambda_lava_max
                        if lambda_lava_max > 0.0 else 0.0),
                    'lava_weight_phase': lava_weight_phase,
                    'learning_rate': current_lr,
                    'grad_norm': grad_norm,
                    'lava_branch_grad_norm': lava_branch_grad_norm,
                    **grad_diagnostics,
                    'update_time_s': time.time() - update_window_start,
                    'data_time_s': update_window_data_time,
                    'lava_scale_mean': scale_mean,
                    'lava_scale_min': scale_min,
                    'lava_scale_max': scale_max,
                    **{
                        f'lava_scale_{scale}_count': update_window_scales.count(scale)
                        for scale in (1, 2, 4, 8, 16)
                    },
                    'gpu_peak_allocated_gb': (
                        torch.cuda.max_memory_allocated() / (1024 ** 3)
                        if torch.cuda.is_available() else 0.0),
                    'gpu_peak_reserved_gb': (
                        torch.cuda.max_memory_reserved() / (1024 ** 3)
                        if torch.cuda.is_available() else 0.0),
                })
                loss_logger.log(
                    epoch + 1, step + 1, global_step, current_step_loss, runtime_info)

                if global_step % 20 == 0:
                    log_msg = (
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {current_step_loss:.4f} "
                        f"Flow: {info_dic['loss_mse']:.4f} "
                        f"LR: {current_lr:.2e} "
                    )
                    if use_future_feat:
                        log_msg += f"FutureFeat: {info_dic.get('loss_future_feat', 0.0):.4f} "
                    if use_lava:
                        log_msg += (
                            f"LAVA: {info_dic.get('loss_lava', 0.0):.4f} "
                            f"Lambda: {info_dic.get('lambda_lava', 0.0):.4f} "
                            f"PosSim: {info_dic.get('pos_sim', 0.0):.4f} "
                            f"NegSim: {info_dic.get('negative_sim', 0.0):.4f} "
                            f"ShuffleSim: {info_dic.get('shuffle_sim', 0.0):.4f} "
                            f"OrderMargin: {info_dic.get('order_margin', 0.0):.4f} "
                            f"LAVA/Base: {info_dic.get('lava_base_ratio', 0.0):.3f} "
                            f"FlowExec: {info_dic.get('loss_flow_executed', float('nan')):.4f} "
                            f"TapFinalCos: {info_dic.get('tap_final_cos', float('nan')):.3f} "
                            f"TaskGap: {info_dic.get('task_shortcut_gap', float('nan')):.3f} "
                            f"TempMargin: {info_dic.get('temporal_margin', float('nan')):.3f} "
                            f"TempAcc: {info_dic.get('temporal_acc', float('nan')):.3f} "
                            f"OrderAcc: {info_dic.get('order_acc', float('nan')):.3f} "
                            f"CandAcc: {info_dic.get('candidate_acc', float('nan')):.3f} "
                            f"Margins[T/F/L/O]:"
                            f"{info_dic.get('cross_task_margin', float('nan')):.2f}/"
                            f"{info_dic.get('far_margin', float('nan')):.2f}/"
                            f"{info_dic.get('temporal_margin', float('nan')):.2f}/"
                            f"{info_dic.get('order_margin', float('nan')):.2f} "
                            f"Hard[T/F/L/O]:"
                            f"{info_dic.get('hardest_cross_task_fraction', float('nan')):.2f}/"
                            f"{info_dic.get('hardest_far_fraction', float('nan')):.2f}/"
                            f"{info_dic.get('hardest_local_fraction', float('nan')):.2f}/"
                            f"{info_dic.get('hardest_order_fraction', float('nan')):.2f} "
                            f"Exec/TailAcc:"
                            f"{info_dic.get('candidate_acc_executed', float('nan')):.2f}/"
                            f"{info_dic.get('candidate_acc_tail', float('nan')):.2f} "
                            f"L2Energy[A/W]:"
                            f"{info_dic.get('action_logsig_l2_energy_fraction', float('nan')):.2f}/"
                            f"{info_dic.get('world_logsig_l2_energy_fraction', float('nan')):.2f} "
                            f"Retrieval: {info_dic.get('retrieval_acc', 0.0):.3f} "
                            f"RawChange: {info_dic.get('raw_change_norm', 0.0):.3f} "
                            f"WStd: {info_dic.get('world_residual_std', 0.0):.3f} "
                            f"AStd: {info_dic.get('action_residual_std', 0.0):.3f} "
                            f"TMean: {info_dic.get('lava_t_mean', 0.0):.3f} "
                            f"Samples: {info_dic.get('lava_sample_count', 0)} "
                            f"ScaleCounts: "
                            f"1:{runtime_info['lava_scale_1_count']}/"
                            f"2:{runtime_info['lava_scale_2_count']}/"
                            f"4:{runtime_info['lava_scale_4_count']}/"
                            f"8:{runtime_info['lava_scale_8_count']}/"
                            f"16:{runtime_info['lava_scale_16_count']} ")
                    if run_grad_diagnostics:
                        log_msg += (
                            f"GradCos: {grad_diagnostics['grad_cos_shared']:.3f} "
                            f"GradCos9_10: {grad_diagnostics.get('grad_cos_b9_10', float('nan')):.3f} "
                            f"WeightedGradRatio: "
                            f"{grad_diagnostics['weighted_grad_ratio']:.3f} ")
                    log_msg += (
                        f"GradNorm: {grad_norm:.3f} "
                        f"LAVABranchGrad: {lava_branch_grad_norm:.3f} "
                        f"UpdateTime: {runtime_info['update_time_s']:.2f}s "
                        f"DataTime: {runtime_info['data_time_s']:.2f}s "
                        f"GPUPeak: {runtime_info['gpu_peak_allocated_gb']:.1f}GB")
                    logger.info(log_msg)

                last_iteration_end = time.time()
                update_window_start = last_iteration_end
                update_window_data_time = 0.0
                update_window_scales = []
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                if args.max_steps is not None and global_step >= args.max_steps:
                    logger.info(f"Reached --max_steps={args.max_steps}; stopping early.")
                    break
            else:
                last_iteration_end = time.time()

        avg_loss = epoch_loss / max(batches_seen, 1)
        elapsed = time.time() - start_time
        logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ===")

        if use_lava and epoch_lava_task_counts:
            sampling_total = sum(epoch_lava_task_counts.values())
            sampling_counts = dict(sorted(epoch_lava_task_counts.items()))
            sampling_fractions = {
                task_name: count / sampling_total
                for task_name, count in sampling_counts.items()
            }
            nonzero_counts = list(sampling_counts.values())
            sampling_audit = {
                'epoch': epoch + 1,
                'sampling_balance': str(
                    config.training.get('lava_sampling_balance', 'none')),
                'total_lava_samples': sampling_total,
                'task_counts': sampling_counts,
                'task_fractions': sampling_fractions,
                'max_min_count_ratio': (
                    max(nonzero_counts) / min(nonzero_counts)
                    if nonzero_counts else float('nan')),
            }
            sampling_audit_path = os.path.join(
                run_save_dir, 'log', f'lava_sampling_epoch_{epoch + 1:03d}.json')
            with open(sampling_audit_path, 'w', encoding='utf-8') as audit_file:
                json.dump(sampling_audit, audit_file, indent=2, sort_keys=True)
            logger.info(
                "LAVA sampling audit epoch %d: total=%d task_counts=%s max/min=%.3f",
                epoch + 1,
                sampling_total,
                sampling_counts,
                sampling_audit['max_min_count_ratio'],
            )

        if epoch_action_similarity_audit:
            nested_audit = {}
            for (task_name, scale, family), values in sorted(
                    epoch_action_similarity_audit.items()):
                count = values['count']
                nested_audit.setdefault(task_name, {}).setdefault(
                    str(scale), {})[family] = {
                        'count': count,
                        'mean_distance': values['distance_sum'] / count,
                        'mean_weight': values['weight_sum'] / count,
                        'weight_below_0_5_fraction': values['below_half'] / count,
                        'mean_arm_distance': values['arm_sum'] / count,
                        'mean_gripper_state_distance': (
                            values['gripper_state_sum'] / count),
                        'mean_gripper_change_distance': (
                            values['gripper_change_sum'] / count),
                    }
            audit_path = os.path.join(
                run_save_dir, 'log',
                f'lava_action_similarity_epoch_{epoch + 1:03d}.json')
            with open(audit_path, 'w', encoding='utf-8') as audit_file:
                json.dump({'epoch': epoch + 1, 'task_scale_family': nested_audit},
                          audit_file, indent=2, sort_keys=True)

        if not args.no_save and ((epoch + 1) % save_interval_epoch == 0 or (epoch + 1) == epochs):
            ckpt_name = (f"checkpoint_{checkpoint_tag}_epoch_{epoch+1}.pt"
                         if checkpoint_tag else f"checkpoint_epoch_{epoch+1}.pt")
            ckpt_path = os.path.join(run_save_dir, ckpt_name)
            save_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': action_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            if lava_batch_sampler is not None:
                save_dict['lava_batch_sampler_state'] = (
                    lava_batch_sampler.state_dict())
            # A preemptible Slurm job can be terminated while torch.save is
            # writing. Keep incomplete files outside the *.pt auto-resume
            # pattern, then publish the checkpoint atomically.
            temporary_ckpt_path = f"{ckpt_path}.tmp.{os.getpid()}"
            try:
                torch.save(save_dict, temporary_ckpt_path)
                os.replace(temporary_ckpt_path, ckpt_path)
            finally:
                if os.path.exists(temporary_ckpt_path):
                    os.remove(temporary_ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    logger.info("Training Complete.")
