{
  "homework_name": "Homework: Sim-to-Real Quadruped Locomotion with MuJoCo Playground",
  "robot": "Go2",
  "environment_name": "Go2JoystickFlatTerrain",
  "framework": "MuJoCo Playground + Brax PPO + MJX",
  "backend_impl": "jax",
  "actor_obs_key": "state",
  "critic_obs_key": "privileged_state",
  "use_domain_randomization": true,
  "seed": 0,
  "control": {
    "ctrl_dt": 0.02,
    "sim_dt": 0.004,
    "action_scale": 0.5,
    "action_type": "absolute_joint_position_target",
    "torque_mapping": "position_target_through_pd_actuator"
  },
  "course_budget": {
    "baseline_total_env_steps": 15000000,
    "leaderboard_max_env_steps": 120000000,
    "flat_terrain_only": true,
    "require_colab_gpu_runtime": true
  },
  "training_defaults": {
    "num_envs": 2048,
    "num_eval_envs": 256,
    "num_evals": 6,
    "batch_size": 512,
    "policy_hidden_layer_sizes": [
      512,
      256,
      128
    ],
    "value_hidden_layer_sizes": [
      512,
      512,
      256
    ]
  },
  "stage_1": {
    "name": "stage_1",
    "num_timesteps": 12000000,
    "command_range": {
      "min": [
        0.0,
        0.0,
        0.0
      ],
      "max": [
        1.2,
        0.0,
        0.0
      ]
    },
    "command_keep_prob": [
      1.0,
      0.0,
      0.0
    ],
    "tracking_sigma": 0.25,
    "reward_scales": {
      "tracking_lin_vel": 1.5,
      "action_rate": -0.015,
      "energy": -0.0012,
      "termination": -8.0
    }
  },
  "stage_2": {
    "name": "stage_2",
    "num_timesteps": 40000000,
    "command_range": {
      "min": [
        0.0,
        -0.25,
        -0.6
      ],
      "max": [
        3.2,
        0.25,
        0.6
      ]
    },
    "command_keep_prob": [
      1.0,
      0.4,
      0.7
    ],
    "student_stage2_goal": {
      "command_range": {
        "min": [
          0.0,
          -0.25,
          -0.6
        ],
        "max": [
          3.2,
          0.25,
          0.6
        ]
      },
      "command_keep_prob": [
        1.0,
        0.4,
        0.7
      ],
      "track_sampler": {
        "enable": true,
        "mode_probs": [
          0.50,
          0.35,
          0.15
        ],
        "straight_vx": [
          0.6,
          3.2
        ],
        "straight_yaw": [
          -0.05,
          0.05
        ],
        "curve_vx": [
          0.8,
          2.5
        ],
        "curve_radius": 18.25,
        "curve_yaw_noise": [
          -0.03,
          0.06
        ],
        "curve_yaw_scale": [
          0.85,
          1.25
        ],
        "curve_yaw_abs": [
          0.05,
          0.24
        ],
        "curve_vy": [
          -0.06,
          0.06
        ],
        "recovery_vx": [
          0.0,
          1.4
        ],
        "recovery_vy": [
          -0.25,
          0.25
        ],
        "recovery_yaw": [
          -0.6,
          0.6
        ]
      }
    },
    "tracking_sigma": 0.45,
    "reward_scales": {
      "tracking_lin_vel": 2.0,
      "tracking_ang_vel": 0.6,
      "pose": 0.35,
      "action_rate": -0.015,
      "energy": -0.001,
      "torques": -0.00015,
      "feet_slip": -0.18,
      "feet_clearance": -1.5,
      "feet_height": -0.15,
      "orientation": -10.0,
      "lin_vel_z": -0.6,
      "termination": -12.0
    },
    "perturbation": {
      "enable": true,
      "velocity_kick": [
        0.0,
        1.2
      ],
      "kick_durations": [
        0.05,
        0.12
      ],
      "kick_wait_times": [
        2.0,
        5.0
      ]
    },
    "restore_previous_stage_checkpoint": true
  },
  "stage_3": {
    "name": "stage_3",
    "num_timesteps": 60000000,
    "command_range": {
      "min": [
        0.0,
        -0.18,
        -0.45
      ],
      "max": [
        4.2,
        0.18,
        0.45
      ]
    },
    "command_keep_prob": [
      1.0,
      0.25,
      0.6
    ],
    "track_sampler": {
      "enable": true,
      "mode_probs": [
        0.55,
        0.35,
        0.10
      ],
      "straight_vx": [
        1.0,
        4.2
      ],
      "straight_yaw": [
        -0.04,
        0.04
      ],
      "curve_vx": [
        1.2,
        2.8
      ],
      "curve_radius": 18.25,
      "curve_yaw_noise": [
        -0.025,
        0.05
      ],
      "curve_yaw_scale": [
        0.9,
        1.2
      ],
      "curve_yaw_abs": [
        0.05,
        0.28
      ],
      "curve_vy": [
        -0.05,
        0.05
      ],
      "recovery_vx": [
        0.2,
        1.6
      ],
      "recovery_vy": [
        -0.18,
        0.18
      ],
      "recovery_yaw": [
        -0.45,
        0.45
      ]
    },
    "tracking_sigma": 0.65,
    "reward_scales": {
      "tracking_lin_vel": 2.0,
      "tracking_ang_vel": 0.55,
      "tracking_forward_vel": 1.0,
      "forward_progress": 0.35,
      "pose": 0.25,
      "action_rate": -0.012,
      "energy": -0.0008,
      "torques": -0.00012,
      "feet_slip": -0.22,
      "feet_clearance": -1.2,
      "feet_height": -0.12,
      "orientation": -11.0,
      "lin_vel_z": -0.7,
      "termination": -18.0
    },
    "perturbation": {
      "enable": true,
      "velocity_kick": [
        0.0,
        1.4
      ],
      "kick_durations": [
        0.05,
        0.12
      ],
      "kick_wait_times": [
        2.0,
        5.0
      ]
    },
    "restore_previous_stage_checkpoint": true
  },
  "demo_rollout": {
    "segment_seconds": 6.0,
    "segments": [
      [0.6, 0.0, 0.0],
      [1.2, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [2.8, 0.0, 0.0],
      [1.8, 0.0, 0.10],
      [2.2, 0.03, -0.12]
    ]
  },
  "public_eval": {
    "episode_length_seconds": 30.0,
    "safe_command_ranges": {
      "vx": [
        0.8,
        3.0
      ],
      "vy": [
        -0.08,
        0.08
      ],
      "yaw": [
        -0.28,
        0.28
      ],
      "turn_radius": 18.25
    },
    "metrics": {
      "velocity_tracking_error": {
        "direction": "lower_better",
        "weight": 0.35,
        "good": 0.1,
        "bad": 0.45
      },
      "yaw_tracking_error": {
        "direction": "lower_better",
        "weight": 0.2,
        "good": 0.1,
        "bad": 0.5
      },
      "fall_rate": {
        "direction": "lower_better",
        "weight": 0.2,
        "good": 0.0,
        "bad": 0.35
      },
      "energy_proxy": {
        "direction": "lower_better",
        "weight": 0.15,
        "good": 8.0,
        "bad": 40.0
      },
      "foot_slip_proxy": {
        "direction": "lower_better",
        "weight": 0.1,
        "good": 0.02,
        "bad": 0.2
      }
    }
  },
  "files_to_read_first": [
    "go2_pg_env/joystick.py",
    "train.py",
    "benchmark_specs.py",
    "public_eval.py"
  ],
  "submission": {
    "required_files": [
      "best_checkpoint/",
      "configs/colab_runtime_config.json",
      "public_eval_bundle/public_eval.json",
      "demo_bundle/demo.mp4",
      "short_report.pdf"
    ],
    "report_required_questions": [
      "Explain the observation, action, and reward design.",
      "Describe what you changed and why.",
      "Report which benchmark metrics improved or worsened.",
      "Explain why your changes may or may not help sim-to-real transfer.",
      "Describe at least one failed idea and what you learned from it."
    ]
  }
}
