#!/bin/bash

#Figure 3: KDE score distribution for different models


REQUEST_RATES=(5 10 15 20 25 30 35 40 45 50)   # add more rates here if needed, e.g. (5.0 10.0 20.0)

MODELS=(
    "llama3_70b"
)

MODES=(
    "APC"
)

for rate in "${REQUEST_RATES[@]}"; do
    for model in "${MODELS[@]}"; do
        for modes in "${MODES[@]}"; do
            MODEL_PATH="./models/$model"
            echo "Running experiment: $MODEL_PATH | request_rate=$rate | mode=$modes"
            CUDA_VISIBLE_DEVICES=0,1 python -m src.performance_evaluation_with_timestamps \
                --csv_paths ./data/exploitability/exploitability.csv \
                --request_rate "$rate" \
                --model_path "$MODEL_PATH" \
                --sys_mode "$modes" \
                --experiment_name "Figure_3_exploitability" \
                --tensor_parallel_size  2
        done
    done
done



REQUEST_RATES=(10)   # Add more rates here if needed, e.g. (5.0 10.0 20.0)

WORKLOADS=(
    "workload_avg"
)

MODELS=(
    "llama3_70b"
)

MODES=(
    "CacheSolidarity"
    "APC"
    "Isolation"
    "SafeKV"
)

for workload in "${WORKLOADS[@]}"; do
    for rate in "${REQUEST_RATES[@]}"; do
        for model in "${MODELS[@]}"; do
            for mode in "${MODES[@]}"; do

                CSV_PATH="./data/workloads/${workload}.csv"
                MODEL_PATH="./models/$model"

                echo "Running experiment: $MODEL_PATH | workload=$workload | request_rate=$rate | mode=$mode"

                CUDA_VISIBLE_DEVICES=0,1 python -m src.performance_evaluation_with_timestamps \
                    --csv_paths "$CSV_PATH" \
                    --request_rate "$rate" \
                    --model_path "$MODEL_PATH" \
                    --sys_mode "$mode" \
                    --experiment_name "fig8_models" \
                    ----tensor_parallel_size 2

            done
        done
    done
done