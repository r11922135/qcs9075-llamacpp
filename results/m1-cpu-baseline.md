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

### 還沒驗證

- [ ] 03、04 用 `-n 2048` 重跑,看完整回答。
  (第一次試的時候直接在 adb shell 打中文,Big5 位元組讓 llama-cli terminate —— 是輸入編碼問題,不是模型問題。)
- [ ] 記憶體峰值:llama-bench 不報 RSS。

## Qwen3.5-9B Q4_0

(待測:`.\push.bat -Bench`)
