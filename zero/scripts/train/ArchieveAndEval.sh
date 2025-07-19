# --------------------
# region input

run_dir=$1
time_part=$2
run_name=$3

if [ -z "$run_dir" ] || [ -z "$time_part" ] || [ -z "$run_name" ]; then
    echo "Usage: $0 <run_dir> <time_part> <run_name>"
    exit 1
fi

# --------------------
# region Archieve

rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
    ssh jian@10.12.65.19 "touch /media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name_stage2}/ready.flag"

# --------------------
# region Eval
rsync -avP ${run_dir}/ jian@10.12.65.130:/data/eval_candidates/${time_part}_${run_name}/ &&
    ssh jian@10.12.65.130 "touch /data/eval_candidates/${time_part}_${run_name}/ready.flag"
