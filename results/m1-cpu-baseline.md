# M1 CPU baseline

## Qwen3.6-35B-A3B Q4_0 —— 2026-09-11

| 條件 | 值 |
|---|---|
| llama.cpp | `df03399`(2026-09-10),只有 CPU backend,`-march=armv8.2-a+fp16+dotprod`,OpenMP off |
| 模型 | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` / `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`(19.40 GiB,35.51 B params,sha256 已驗) |
| 板子狀態 | `spirit-geniex`、`spirit-speechd` 暫停;已搬到 root cgroup(`cpus: 0-7`);開跑前 MemAvailable 31,638,916 kB |
| 指令 | `llama-bench -p 512 -n 128 -t 4,8 -r 3`(`device/bench.sh`) |

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_0         |  19.40 GiB |    35.51 B | CPU        |       4 |           pp512 |         27.34 ± 1.32 |
| qwen35moe 35B.A3B Q4_0         |  19.40 GiB |    35.51 B | CPU        |       4 |           tg128 |         10.02 ± 0.01 |
| qwen35moe 35B.A3B Q4_0         |  19.40 GiB |    35.51 B | CPU        |       8 |           pp512 |         55.28 ± 0.41 |
| qwen35moe 35B.A3B Q4_0         |  19.40 GiB |    35.51 B | CPU        |       8 |           tg128 |         12.32 ± 0.03 |

板上原檔:`/opt/llamacpp/results/Qwen_Qwen3.6-35B-A3B-Q4_0-20260911-103136.md`

### 判讀

- **生成(tg)12.3 tok/s**,只用 CPU。拿自家既有數字當參考:Qwen3-VL-8B 走 QAIRT NPU 是 11.31 tok/s、
  Qwen3-VL-4B 走 GenieX 是 12.4 tok/s。量法不同(這裡是 llama-bench 從空 context 生 128 個 token,
  那兩個是 VLM 端到端),只能看量級 —— **35B 在 CPU 上的生成速度跟現在 NPU 上的 4B/8B 同一級**。
- **pp 從 4 核到 8 核是 2.02 倍**:prefill 吃算力,幾乎線性,也證明 URM escape 有效、8 核真的都用上了。
- **tg 從 4 核到 8 核只有 1.23 倍**:生成主要卡記憶體存取。以每個 token 約讀 1.8 GB 權重估算,
  12.3 tok/s 約 22 GB/s,是 LPDDR5 理論峰值 76.8 GB/s 的三成左右(估算值)。
- **弱點是 prefill:55 tok/s**。500 個 token 的 prompt 要約 9 秒才吐第一個字,1000 個要約 18 秒。
  短問答沒問題,長 system prompt / RAG 會很有感 —— 這是 M2(NPU)要解的。
- **4 核的數字(tg 10.0 / pp 27.3)就是做成 systemd service 後的預設表現**(URM 把 system.slice 關在 0-3)。

### 輸出品質(`ask.sh 35b`,`-rea off`,spirit 服務開著)

板上原檔:`/opt/llamacpp/results/ask-35b-20260911-105229.txt`

| 題目 | 結果 | 實際對話速度(Prompt / Generation) |
|---|---|---|
| 01 MoE 三句話 | ✅ 正確、剛好三句 | 7.6 / 7.3 t/s |
| 02 算術 | ✅ 找回 70 元,步驟清楚 | 12.1 / 10.0 t/s |
| 03 nproc 只有 4 | ⚠ 被 `-n 512` 切斷;已寫的 isolcpus / maxcpus / 裝置樹 / cpu online 都對,還沒寫到 cgroup cpuset(本板真正的原因) | 9.0 / 10.0 t/s |
| 04 UTF-8 檢查函式 | ⚠ 被 `-n 512` 切斷;已寫的位元遮罩判斷正確 | 9.6 / 10.1 t/s |

- 03、04 切斷是 ask.sh 當時的生成上限 `-n 512`,不是程式問題;已改成 2048,待重跑。
- **用語偏大陸**:字是繁體,但出現「網絡」「激活」「設備樹」「進程」(台灣是網路、啟用、裝置樹、行程)。
  產品要給台灣使用者看的話,要用 system prompt 要求台灣用語,再看改不改得掉。
- 實際對話的生成(7.3–10.1 t/s)比 llama-bench 的 12.3 低。可能原因(未驗證):這次 spirit 服務沒停、
  真的在做 sampling(llama-bench 不做)、context 隨生成變長。Prompt t/s 低是因為題目只有幾十個 token,
  固定開銷占比大,不能跟 pp512 比。

### 背景服務對生成速度的影響(2026-09-11,`llama-bench -p 0 -n 128 -t 8 -r 3`)

| 條件 | tg128(t/s) |
|---|---|
| spirit-geniex + spirit-speechd 開著(第 1 次) | 9.62 ± 2.58 |
| 兩個都停掉 | **12.43 ± 0.14** |
| 開著(第 2 次) | 9.37 ± 2.38 |
| 開著(第 3 次) | 9.68 ± 2.05 |
| 停掉(第 2 次) | **8.62 ± 2.88** |
| 停掉(第 3 次) | 12.23 ± 0.14 |

- ~~確定:服務開著時會慢~~ —— **被後兩筆推翻**:服務停掉也出現了慢的一次。
  結果只有兩種狀態:**穩定的 12.2–12.4(± 0.1)**,或**不穩定的 8.6–9.7(± 2 以上)**,跟服務開關沒有必然關係。
  前 5 筆看起來的相關性是巧合(樣本太少就下結論,教訓)。
- 服務本身確實閒置:CPU 96% idle;兩個服務重啟後約 40 分鐘只用了 3.7 s / 1.8 s CPU;
  spirit-app 沒在跑(`inactive (dead)`);speechd 是 socket 服務,不會自己收音。停掉只多出約 0.9 GiB 可用記憶體。
### 診斷:是 swap(2026-09-11 監看,兩次都是慢的:8.89 ± 2.68、8.99 ± 2.07)

bench 時另一個 shell 每 2 秒記錄 llama-bench 的 cpus / cgroup / majflt(major page fault)、`/proc/vmstat` 的 pswpin、cpu0 / cpu4 頻率:

| 階段 | majflt | pswpin | 解讀 |
|---|---|---|---|
| 第 1 次載入(11:38:53–11:39:36) | 0 → 26 萬 | 幾乎不動 | 從儲存讀模型檔,正常 |
| 第 1 次生成(11:39:38–11:40:27) | 26 萬 → 219 萬 | **+192 萬頁 ≈ 7.3 GiB** | 一邊生成一邊把權重從 swap 搬回來 |
| 第 2 次載入 | 幾乎不動 | 不動 | 檔案已在 page cache |
| 第 2 次生成 | +186 萬 | **+185 萬頁 ≈ 7.1 GiB** | 同上 |

- 生成時 majflt 的增量 ≈ pswpin 的增量 → 那些 page fault 全是 swap-in。每 2 秒的 swap-in 量隨時間遞減
  (約 21 萬頁 → 1 千頁),符合「慢慢把被丟出去的權重搬回來,搬完就恢復」。
- **排除 URM**:cpus 從頭到尾都是 0-7(只有第一筆 cgroup 是 `/urm.slice/focused.apps`,之後都是 `/`)。
- **排除降頻**:生成階段 cpu0、cpu4 都固定在 2361600 kHz(最高頻);載入與閒置時才會上下跳。
- 機制:載入時檔案進 page cache(最多 20.8 GB),同時另配 19.4 GiB 放 repack 過的權重,兩份超過 33 GiB,
  kernel 把剛寫好、還沒用到的權重 swap 出去。MoE 每個 token 只用 256 個 expert 中的 8 個,冷門 expert
  最容易被丟出去、也最晚被搬回來。穩定的 12.4 那幾次,推測是載入時沒有被 swap 出多少。
- **`-lm mlock` 救不了**:這版的 mlock 只鎖 host buffer(`src/llama-model.cpp`),而 CPU repack buffer 的
  `is_host` 是 `nullptr`(`ggml-cpu/repack.cpp`)—— 19.4 GiB 權重幾乎都在 repack buffer 裡。
- swap 是 **`/dev/zram0`**(33.8 GB,等於 RAM 大小;壓縮後放在 RAM,不磨損 flash),**`vm.swappiness = 100`**
  —— kernel 很願意 swap 匿名記憶體,而不是只丟 page cache,所以剛 repack 好的權重首當其衝。
  量化後的權重本來就很難壓縮,丟進 zram 省不了多少空間,卻要付壓縮與解壓的 CPU。

### 解法:`-lm dio`(已驗證)

| 載入方式 | tg128(t/s) | 生成期間 majflt / pswpin |
|---|---|---|
| 預設(mmap) | 8.6–12.4,忽快忽慢 | 慢的那次約 190 萬 / 約 7 GiB |
| **`-lm dio`** | **12.55 ± 0.02** | **1 / 幾乎不動(+8 頁)** |

- `-lm dio` 用 O_DIRECT 讀檔,不經 page cache,「兩份同時占記憶體」的來源直接消失。載入約 50 秒。
  不支援 O_DIRECT 的檔案系統會印警告並退回一般讀檔(`src/llama-mmap.cpp`),不會失敗。
- 只跑了一次,但不只速度,連 page fault 都歸零,機制層面已經確認。`bench.sh`、`ask.sh` 已改用 `-lm dio`,
  之後每次跑都會再驗證。
- **上面第一張表(pp512 55.28 / tg128 12.32)是 mmap 載入時剛好沒被 swap 的那次**,數字可信但不保證重現。
  正式數字以 dio 重跑 `bench.sh` 為準。
- 產品化(llama-server 常駐)要注意:dio 只解決「載入時」的問題。之後其他程式要記憶體時,kernel 仍可能把冷門 expert
  swap 出去;mlock 又鎖不到 repack buffer。候選解法是 systemd 的 `MemorySwapMax=0`(cgroup v2 `memory.swap.max`),
  讓這個 service 的記憶體不被 swap —— 待實測。

### 還沒驗證

- [ ] 03、04 用 `-n 2048` 重跑,看完整回答。
  (第一次試的時候直接在 adb shell 打中文,Big5 位元組讓 llama-cli terminate —— 是輸入編碼問題,不是模型問題。)
- [ ] 記憶體峰值:llama-bench 不報 RSS。

## Qwen3.5-9B Q4_0

(待測:`.\push.bat -Bench`)
