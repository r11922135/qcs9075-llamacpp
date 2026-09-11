#!/bin/sh
# 在板子上直接從 Hugging Face 下載 GGUF,不佔 Windows 空間。
#
#   sh /opt/llamacpp/fetch-model.sh 9b     # Qwen3.5-9B Q4_0(5.74 GB)
#   sh /opt/llamacpp/fetch-model.sh 35b    # Qwen3.6-35B-A3B Q4_0(20.84 GB)
#
# 可續傳:中斷了重跑同一行。已經完整的會直接跳過。
# 板子要能上網:WiFi 用 nmcli device wifi connect <SSID> password <密碼>
# (qwen3vl-genie 的 fetch-speech.sh 就是這樣在板上抓 huggingface 的)。

set -eu

cd "$(dirname "$0")"
die() { printf '[FAIL] %s\n' "$1" >&2; exit 1; }

# 大小與 sha256 取自 HF API(2026-09-11)
case "${1:-}" in
    9b)  REPO=bartowski/Qwen_Qwen3.5-9B-GGUF
         FILE=Qwen_Qwen3.5-9B-Q4_0.gguf
         SIZE=5741391904
         SHA256=a44b0e06af0c97cece312f6dca52b3639d038d1a74a2e194b22e159f9ccdbb21 ;;
    35b) REPO=bartowski/Qwen_Qwen3.6-35B-A3B-GGUF
         FILE=Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
         SIZE=20836243072
         SHA256=52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e ;;
    *)   die "用法: sh $0 9b|35b" ;;
esac

DST="models/$FILE"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

mkdir -p models
if [ "$(size_of "$DST")" = "$SIZE" ]; then
    printf '[ OK ] 已經有了:%s\n' "$DST"
    exit 0
fi

command -v curl >/dev/null 2>&1 || die "板上沒有 curl"
# 先確認網路,比下載到一半才失敗好判讀。
curl -sSfI --max-time 15 -o /dev/null https://huggingface.co ||
    die "連不到 huggingface.co。WiFi 有連上嗎?公司網路要 proxy 的話先 export https_proxy=http://..."

have=$(size_of "$DST")
avail_kb=$(df -Pk models | awk 'NR==2{print $4}')
need_kb=$(( (SIZE - have) / 1024 + 512 * 1024 ))
[ "$avail_kb" -ge "$need_kb" ] || die "空間不足:剩 $((avail_kb / 1024)) MB,還要 $((need_kb / 1024)) MB"

printf '[INFO] 下載 %s(共 %s bytes,已有 %s)\n' "$FILE" "$SIZE" "$have"
printf '[INFO] %s\n' "$URL"

# 網路斷了自動續傳。curl 自己的 --retry 搭配 -C - 不保證會重算續傳位置,所以自己繞迴圈。
n=0
until curl -fL -C - --progress-bar -o "$DST" "$URL"; do
    n=$((n + 1))
    [ "$(size_of "$DST")" -lt "$SIZE" ] || break
    [ "$n" -lt 20 ] || die "連續失敗 $n 次。重跑同一行會從 $(size_of "$DST") bytes 續傳"
    printf '[WARN] 下載中斷,10 秒後續傳(第 %d 次)\n' "$n"
    sleep 10
done

got=$(size_of "$DST")
[ "$got" = "$SIZE" ] || die "大小不符:$got,應為 $SIZE。刪掉重下:rm $(pwd)/$DST"

if command -v sha256sum >/dev/null 2>&1; then
    printf '[INFO] 驗證 sha256(大檔要等一下)...\n'
    sum=$(sha256sum "$DST" | cut -d' ' -f1)
    [ "$sum" = "$SHA256" ] || die "sha256 不符,檔案壞了。刪掉重下:rm $(pwd)/$DST"
fi
printf '[ OK ] %s\n' "$(pwd)/$DST"
