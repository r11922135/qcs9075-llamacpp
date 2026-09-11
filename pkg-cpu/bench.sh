#!/bin/sh
# M1 CPU baseline:在板上跑 llama-bench,4 核與 8 核各一輪。
#
#   sh /opt/llamacpp/bench.sh models/Qwen_Qwen3.5-9B-Q4_0.gguf
#   sh /opt/llamacpp/bench.sh models/xxx.gguf -r 1      # 其餘參數原樣傳給 llama-bench
#
# 結果同時寫進 /opt/llamacpp/results/,檔名帶模型與時間。

set -eu

cd "$(dirname "$0")"
die() { printf '[FAIL] %s\n' "$1" >&2; exit 1; }

[ $# -ge 1 ] || die "用法: sh $0 <model.gguf> [llama-bench 參數...]"
MODEL="$1"; shift
[ -f "$MODEL" ] || die "找不到模型: $MODEL"

# 執行檔是用 glibc 2.43 的 sysroot 編的;板上 glibc 太舊會在這裡就報錯。
./bin/llama-cli --version >/dev/null 2>&1 || { ./bin/llama-cli --version; die "llama-cli 跑不起來(看上面是不是 GLIBC 版本不符)"; }

for svc in spirit-geniex spirit-speechd; do
    if systemctl is-active -q "$svc" 2>/dev/null; then
        printf '[WARN] %s 還在跑,會搶 CPU/記憶體,數字偏低。乾淨的數字要先 systemctl stop %s\n' "$svc" "$svc"
    fi
done
grep MemAvailable /proc/meminfo

mkdir -p results
out="results/$(basename "$MODEL" .gguf)-$(date +%Y%m%d-%H%M%S).md"

# URM 把 adb shell 起的程序關在 CPU 0-3(system.slice)。搬到 root cgroup 才有 0-7,
# 但 URM 會在程序閒著時把它搬回去,所以要緊貼著 fork llama-bench 之前做。
echo $$ > /sys/fs/cgroup/cgroup.procs
printf 'cpus: %s\n' "$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)"
# -lm dio:用 O_DIRECT 載入,不經 page cache。預設的 mmap 會讓「page cache 裡的檔案 + repack 後的權重」
# 兩份同時占記憶體(35B 合計約 40 GB > 33 GiB),kernel(swappiness=100、zram)就把剛 repack 好的權重
# swap 出去,生成時再一頁頁搬回來:tg 從 12.4 掉到 8.6–9.7 且忽快忽慢(2026-09-11 實測,results/m1-cpu-baseline.md)。
./bin/llama-bench -m "$MODEL" -p 512 -n 128 -t 4,8 -r 3 -lm dio -o md "$@" | tee "$out"

# sh 沒有 pipefail,llama-bench 失敗時 tee 仍回 0,所以看產出內容判斷。
grep -q 'tg128' "$out" || die "llama-bench 沒有產出結果(看上面的錯誤訊息)"
printf '\n[ OK ] %s\n' "$out"
printf '8 核若沒有比 4 核快,可能被 URM 搬回 0-3 了。\n'
