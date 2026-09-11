#!/bin/sh
# 拿 prompts/ 裡的題目實際問模型,確認輸出品質(bench 只量速度,輸出壞了也照樣有數字)。
#
#   sh /opt/llamacpp/ask.sh 35b                             # prompts/ 裡全部題目
#   sh /opt/llamacpp/ask.sh 9b prompts/02-math.txt          # 只問一題
#   sh /opt/llamacpp/ask.sh 35b prompts/02-math.txt -rea on # 之後的參數原樣傳給 llama-cli
#
# 中文題目一律放在 UTF-8 檔案裡(device/prompts/,跟著 push.bat 推上來)。
# 不要在 adb shell 直接打中文:Windows console 送上來的是 Big5,
# llama-cli 遇到不合法的 UTF-8 會直接 terminate(common_json_error ... invalid UTF-8 byte)。
#
# 回答存到 results/ask-<模型>-<時間>.txt,載入訊息另存同名 .log。

set -eu

cd "$(dirname "$0")"
die() { printf '[FAIL] %s\n' "$1" >&2; exit 1; }

KEY="${1:-}"
case "$KEY" in
    9b)  MODEL=models/Qwen_Qwen3.5-9B-Q4_0.gguf ;;
    35b) MODEL=models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf ;;
    *)   die "用法: sh $0 9b|35b [prompts/xx.txt] [llama-cli 參數...]" ;;
esac
shift
[ -f "$MODEL" ] || die "找不到 $MODEL,先在 Windows 跑 .\\push.bat -Model $KEY"

if [ $# -ge 1 ] && [ -f "$1" ]; then
    PROMPTS="$1"; shift
else
    PROMPTS="$(ls prompts/*.txt)"
fi

mkdir -p results
out="results/ask-$KEY-$(date +%Y%m%d-%H%M%S).txt"
log="${out%.txt}.log"

for p in $PROMPTS; do
    printf '\n===== %s =====\n' "$p" | tee -a "$out"
    tee -a "$out" < "$p"
    printf '%s\n' '-----' | tee -a "$out"
    # 跟 bench.sh 一樣:緊貼著啟動前搬到 root cgroup,才用得到 8 核。
    echo $$ > /sys/fs/cgroup/cgroup.procs
    # -rea off:先看不思考時的直接回答;--seed 固定,9B / 35B 才好對照。
    # -n 是生成上限:512 會把 Linux 與寫程式那兩題的回答切在半路(2026-09-11 實測),
    # 2048 在 10 tok/s 下最多約 3.5 分鐘。還不夠就在後面再帶一次 -n,後面的值優先。
    # -lm dio:不經 page cache 載入,避免權重被 swap 出去(原因見 bench.sh)。
    ./bin/llama-cli -m "$MODEL" -t 8 -st -rea off -n 2048 --seed 42 --simple-io -lm dio -f "$p" "$@" \
        2>>"$log" | tee -a "$out"
done

printf '\n[ OK ] 回答存在 %s(載入訊息在 %s)\n' "$(pwd)/$out" "$(pwd)/$log"
