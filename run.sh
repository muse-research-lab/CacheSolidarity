#!/bin/bash

REQUIRED_MODELS=("Llama-2-13b-chat-hf" "Llama-2-7b-chat-hf" "gemma-3-4b-it" "Qwen2.5-VL-3B-Instruct" "llava-onevision-qwen2-0.5b-ov-hf" "Llama-3.2-1B-Instruct")
for m in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "./models/$m/config.json" ]; then
        echo "ERROR: missing model files for $m at ./models/$m"
        echo "Did you mount your downloaded models folder with -v \$(pwd)/models:/app/models ?"
        exit 1
    fi
done

python src/build_datasets.py
python src/patch_ai4privacy.py 


# Table 1: Create processed evaluation datasets using the PII masking script

WORKLOADS=(
    "ShareGPT"
    "Flan"
    "OASST1"
    "UltraChat"     
)

for workload in "${WORKLOADS[@]}"; do
    INPUT_CSV="./data/privacy_evaluation/${workload}.csv"
    OUTPUT_CSV="./temp_results/privacy_evaluation/eval_data/${workload}_processed.csv"
    MODEL_PATH="./models/Llama-2-13b-chat-hf"

    echo "Processing evaluation dataset: $INPUT_CSV -> $OUTPUT_CSV using model $MODEL_PATH"

    python src/pii_masking.py "$INPUT_CSV" "$OUTPUT_CSV" "$MODEL_PATH"
done


# Table 1-2: Create processed evaluation datasets using the PII masking script


WORKLOADS=(
    "ShareGPT_processed"
    "Flan_processed"
    "OASST1_processed"
    "UltraChat_processed"     
)

for workload in "${WORKLOADS[@]}"; do
    INPUT_CSV="./temp_results/privacy_evaluation/eval_data/${workload}.csv"

    echo "Matching ranks for evaluation dataset: $INPUT_CSV"

    python src/bert_pii_rank.py "$INPUT_CSV"
done


#Table 


DATA_DIR=./temp_results/privacy_evaluation/eval_data

for name in Flan OASST1 ShareGPT UltraChat; do
    echo "=== Running PII detection on $name ==="
    python src/safeKV_privacy_eval.py \
        --input_csv "${DATA_DIR}/${name}_processed_match_ranks.csv" \
        --output_dir ./temp_results/pii_results \
        --max_rows 1000 \
        --max_new_tokens 256 \
        --checkpoint_every 200
done




#Figure 3: KDE score distribution for different models


REQUEST_RATES=(5 10 15 20 25 30 35 40 45 50)   # add more rates here if needed, e.g. (5.0 10.0 20.0)

MODELS=(
    "gemma-3-4b-it"
    "Llama-2-7b-chat-hf"
    "Llama-2-13b-chat-hf"
    "llava-onevision-qwen2-0.5b-ov-hf"
    "Qwen2.5-VL-3B-Instruct"
    "llava-onevision-qwen2-0.5b-ov-hf"
)

MODES=(
    "APC"
)

for rate in "${REQUEST_RATES[@]}"; do
    for model in "${MODELS[@]}"; do
        for modes in "${MODES[@]}"; do
            MODEL_PATH="./models/$model"
            echo "Running experiment: $MODEL_PATH | request_rate=$rate | mode=$modes"
            CUDA_VISIBLE_DEVICES=0 python -m src.performance_evaluation_with_timestamps \
                --csv_paths ./data/exploitability/exploitability.csv \
                --request_rate "$rate" \
                --model_path "$MODEL_PATH" \
                --sys_mode "$modes" \
                --experiment_name "Figure_3_exploitability"
        done
    done
done


#Fig 7: Workload 1-5 and average performance evaluation for different systems

REQUEST_RATES=(10)   # Add more rates here if needed, e.g. (5.0 10.0 20.0)

WORKLOADS=(
    "workload_1"
    "workload_2"
    "workload_3"
    "workload_4"
    "workload_5"
    "workload_avg"
)

MODELS=(
    "Llama-2-13b-chat-hf"
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

                CUDA_VISIBLE_DEVICES=0 python -m src.performance_evaluation_with_timestamps \
                    --csv_paths "$CSV_PATH" \
                    --request_rate "$rate" \
                    --model_path "$MODEL_PATH" \
                    --sys_mode "$mode" \
                    --experiment_name "fig7_workloads"

            done
        done
    done
done


# Figure 8: Evaluation of the performance of different models


REQUEST_RATES=(10)   # Add more rates here if needed, e.g. (5.0 10.0 20.0)

WORKLOADS=(
    "workload_avg"
)

MODELS=(
    "gemma-3-4b-it"
    "Llama-2-7b-chat-hf"
    "Llama-2-13b-chat-hf"
    "llava-onevision-qwen2-0.5b-ov-hf"
    "Qwen2.5-VL-3B-Instruct"
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

                CUDA_VISIBLE_DEVICES=0 python -m src.performance_evaluation_with_timestamps \
                    --csv_paths "$CSV_PATH" \
                    --request_rate "$rate" \
                    --model_path "$MODEL_PATH" \
                    --sys_mode "$mode" \
                    --experiment_name "fig8_models"

            done
        done
    done
done




# Figure 9: Evaluation of the performance of different models


REQUEST_RATES=(1 5 10 15 20)   # Add more rates here if needed, e.g. (5.0 10.0 20.0)

WORKLOADS=(
    "workload_5"

)

MODELS=(
    "Llama-2-13b-chat-hf"
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

                CUDA_VISIBLE_DEVICES=0 python -m src.performance_evaluation_with_timestamps \
                    --csv_paths "$CSV_PATH" \
                    --request_rate "$rate" \
                    --model_path "$MODEL_PATH" \
                    --sys_mode "$mode" \
                    --experiment_name "fig9_RPS"

            done
        done
    done
done




# Figure 10: Sensitivity Analysis

REQUEST_RATES=(1 10 25 50 75 100) 

WORKLOADS=(
    "sensitivity"
)

MODELS=(
    "Llama-2-7b-chat-hf"
)

MODES=(
    "CacheSolidarity"
    "APC"
    "Isolation"

)

for workload in "${WORKLOADS[@]}"; do
    for rate in "${REQUEST_RATES[@]}"; do
        for model in "${MODELS[@]}"; do
            for mode in "${MODES[@]}"; do

                CSV_PATH="./data/sensitivity/${workload}.csv"
                MODEL_PATH="./models/$model"

                echo "Running experiment: $MODEL_PATH | workload=$workload | request_rate=$rate | mode=$mode"

                CUDA_VISIBLE_DEVICES=0 python -m src.performance_evaluation_with_timestamps \
                    --csv_paths "$CSV_PATH" \
                    --request_rate "$rate" \
                    --model_path "$MODEL_PATH" \
                    --sys_mode "$mode" \
                    --experiment_name "fig10_theta_performance"

            done
        done
    done
done