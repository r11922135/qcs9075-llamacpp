# qcs9075-llamacpp — 用 llama.cpp 在 IQ-9075 上跑較大的語言模型

> 建立日期:2026-09-11
> 任務來源:主管交辦 —— 評估候選模型能不能透過 llama.cpp 跑在 IQ-9075(QCS9075)上,
> 重點是「能不能跑**比較聰明**的模型」,不是再跑一顆 4B。

候選清單(ollama tag):`glm-4.7-flash`、`qwen3:4b-instruct-2507-fp16`、`qwen3.5:9b-bf16`、
`llama3.1:8b-instruct-fp16`、`llama3.2:3b-instruct-fp16`、`gpt-oss:20b`、`qwen3.6:35b`、
`gemma4:e4b-it-bf16`、`gemma4:31b`

---

## 1. 選型結論

- **主目標:Qwen3.6-35B-A3B**(ollama `qwen3.6:35b`)
- **暖身兼對照組:Qwen3.5-9B**(用 Q4_0 量化版,**不用** bf16)

|  | Qwen3.6-35B-A3B | Qwen3.5-9B |
|---|---|---|
| 結構 | **MoE**:總 35B,每個 token 只啟用 **3B**(256 experts 取 8 + 1 shared) | dense 9B |
| 層配置 | 40 層 = 10 × (3 Gated DeltaNet + 1 Gated Attention) | 32 層 = 8 × (3 Gated DeltaNet + 1 Gated Attention) |
| MMLU-Pro | **85.2** | 82.5 |
| GPQA | **86.0** | 81.7(Diamond) |
| Q4_0 檔案大小 | 20.84 GB(bartowski) | 5.74 GB(bartowski) |
| 每 token 要讀的權重(Q4_0) | 約 1.8 GB(只讀啟用的 3B) | 約 5 GB |
| decode **理論上限**(76.8 GB/s ÷ 上一列) | 約 40 tok/s | 約 15 tok/s |

**為什麼選 35B-A3B:** 它比 9B 聰明,但 decode 時每個 token 只讀 3B 的權重,
記憶體頻寬的天花板反而接近 9B 的三倍。代價只有一個 —— **常駐記憶體約 21 GB**,
而板子有約 33 GB,放得下。理論上限不是預測值,實際一定低很多(CPU 算力、MoE 路由開銷),
要靠 M1 實測。

**為什麼先跑 9B:** 兩顆是同一個架構家族(Gated DeltaNet 混合注意力),
工具鏈、算子支援問題會一模一樣,但 9B 只有 5 GB —— 下載、推檔、上 NPU 都便宜得多。
先用它把流程打通,也順便產出主管一定會問的「dense 9B vs MoE 35B」對照。

**為什麼 9B 不用 bf16:** 每個 token 要讀約 18 GB 權重,76.8 GB/s ÷ 18 GB ≈ **4.3 tok/s 是天花板**,
實際更低;GenieX 附的 HTP kernel 裡也沒有 bf16 matmul。

其他候選為什麼不先做:

| 模型 | 原因 |
|---|---|
| `gemma4:31b` | dense 31B,Q4 每 token 要讀約 17–18 GB → 天花板約 4 tok/s,不划算 |
| `gpt-oss:20b` | **備胎**。MoE,權重原生就是 MXFP4,HTP 有 MXFP4 kernel;35B 在 NPU 卡關時可拿它驗證 NPU 流程 |
| 3B / 4B / 8B 的那幾顆、`gemma4:e4b` | 小模型,不符合「更聰明」的目的(Gemma 4 E4B 在 GenieX 上已跑過 7.63 tok/s) |
| `glm-4.7-flash` | 本次未評估 |

---

## 2. 板子條件

| 項目 | 值 | 出處 |
|---|---|---|
| RAM | MemTotal **33.0 GiB**(34,626,032 kB);MemAvailable **29.4 GiB**(兩個 spirit 服務都在跑時) | M0 實測 2026-09-11 |
| 儲存 | `/`(otaroot)219 GB,**剩 164 GB**;`/usr` 是 overlay(ostree unlock 中,重開機會消失) | M0 實測 |
| DDR | LPDDR5-6400、96-bit,理論峰值 **76.8 GB/s** | `qcs9075-tdp90-stress/TECHNICAL.md`(2026-07) |
| CPU | 8 核;features 有 `asimddp`(dotprod)、`fphp`/`asimdhp`(fp16),**沒有 i8mm / sve / bf16** | M0 實測 |
| GPU | Adreno 663(`clinfo -l`) | TDP90 TECHNICAL.md |
| NPU | **2 顆** HTP v73(`/dev/fastrpc-cdsp`、`/dev/fastrpc-cdsp1`,root:fastrpc 0640) | M0 實測 |

M0 判讀:

- **記憶體夠**:35B 的 Q4_0 是 20.84 GB(19.4 GiB),加上 context 約 21 GiB;
  即使兩個 spirit 服務都開著,也還剩約 8 GiB。
- **模型放 `/opt/llamacpp/models/`**(沿用 `/opt/vlm`、`/opt/yolov8-cam` 的慣例,在 otaroot 上)。
  ⚠ **不要放 `/tmp`**:那是 tmpfs(17 GB),吃的是 RAM。
- **CPU 只有 dotprod、沒有 i8mm** → Q4_0 在 CPU 上會 repack 成 `q4_0_4x4_q8_0`
  (上游 `ggml-cpu/repack.cpp` 的選擇邏輯:有 i8mm 才走 4x8)。MoE 的 MUL_MAT_ID 也支援 repack。
  上游 `arm64-linux-snapdragon` preset 的 `-march=armv8.2a+fp16+dotprod` 剛好對上;
  Android / Windows 的 preset 有開 `i8mm`,拿來用會 SIGILL。
- ⚠ **`nproc` 回報 4**:adb shell 的 `Cpus_allowed_list` 是 0-3,`system.slice` 是 0-3,root cgroup 是 0-7
  → URM 的限制在這版 image 仍然存在。`llama-cli -t 8` 實際只拿得到 4 核。
  要先 `echo $$ > /sys/fs/cgroup/cgroup.procs` 搬到 root cgroup,而且要**緊貼著啟動前**做
  (URM 會把閒著的程序搬回去;TDP90 壓測專案 commit 2dbc239 踩過)。`device/bench.sh` 已處理。
  另外這也代表:**將來做成 systemd service 的話,預設只有 4 核**,所以 4 核的數字也要量。
- ⚠ **兩顆 NPU 都被占用**:`spirit-geniex`(VLM,HTP device 0)、`spirit-speechd`(Whisper,HTP device 1)。
  M1 CPU 測試要乾淨的數字就先停掉;M2 NPU 測試則必須停掉。

---

## 3. llama.cpp 現況(2026-09-11 查證)

**上游已經官方支援 Snapdragon Linux。** 用 Qualcomm 提供的 docker 映像交叉編譯:
`ghcr.io/snapdragon-toolchain/arm64-linux:v0.7`(內含 Hexagon SDK),一次編出三個 backend:
CPU、OpenCL(Adreno)、Hexagon(`libggml-htp-v73/v75/v79/v81.so`)。

雙 NPU 可以直接用:`GGML_HEXAGON_DEVICES=HTP0:0,HTP1:0`;
同一顆 NPU 也能開多個 virtual session(`HTP0:0,HTP0:1`)來繞過單一 session 的記憶體上限。

**Qwen3.5 / 3.6 特有算子的支援**(上游 `docs/ops.md`):

| 算子 | CPU | HTP(NPU) | OpenCL(Adreno) |
|---|---|---|---|
| GATED_DELTA_NET | ✅ | ✅ | ❌ |
| SSM_CONV | ✅ | ✅ | ✅ |
| CUMSUM | ✅ | ✅ | ❌ |
| SOLVE_TRI / TRI | ✅ | ✅ | ❌ |
| L2_NORM | ✅ | 🟡 | ❌ |
| MUL_MAT_ID(MoE 用) | ✅ | 🟡 | 🟡 |

→ **Adreno 這條路不適合這個架構**:DeltaNet 的核心算子不支援,每層都會掉回 CPU 來回搬。
NPU 那邊算子是齊的,MoE 的 MUL_MAT_ID 是部分支援(要看實際落點)。

另一條硬限制(GenieX tarball 裡的 `libggml-htp-v75.so` 抽符號實證):
**HTP 的 matmul kernel 只有 q4_0 / q4_1 / q8_0 / iq4_nl / mxfp4**,沒有任何 K-quant。
Q4_K_M 這種社群主流格式載得起來,但會**靜默**掉回 CPU。

---

## 4. 量化檔選擇

| 選擇 | 檔案 | 大小 | 說明 |
|---|---|---|---|
| **首選** | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` Q4_0 | 20.84 GB | ARM CPU 會自動 online repack,HTP 也有 kernel —— 兩條路都吃 |
| 備選 | 同 repo IQ4_NL | 20.75 GB | 性質同上 |
| 備選 | `unsloth/Qwen3.6-35B-A3B-GGUF` MXFP4_MOE | 21.7 GB | experts 用 MXFP4,HTP 有 kernel |
| 備選 | 同 repo UD-IQ4_NL | 18 GB | 最小,但 UD 是混合量化,上 NPU 前要先 dump 張量型別確認 |

**不要用 ollama 裡的那份。** ollama 的 `qwen3.6:35b` 是 Q4_K_M 等級(約 23–24 GB),
HTP 沒有 K-quant kernel;而且我們本來就需要自己挑量化格式。直接從 HF 抓 llama.cpp 社群版。

確切檔名(HF API 查證):

| 模型 | 檔案 | 大小 |
|---|---|---|
| 9B | `bartowski/Qwen_Qwen3.5-9B-GGUF` / `Qwen_Qwen3.5-9B-Q4_0.gguf` | 5.74 GB |
| 35B | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` / `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` | 20.84 GB |

- bartowski 是用 llama.cpp **b9222** 量化的;板上的 llama.cpp 要夠新才認得這個架構,直接用上游 master。
- 兩個 repo 都有 `mmproj-*-f16.gguf`(約 0.9 GB)可以看圖。**第一階段只做純文字。**
- 35B 的 repo 另有 `mtp-Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`(1.19 GB),是 MTP 草稿模型,
  可用 `--spec-type draft-mtp` 做 speculative decoding 加速 —— 留到 M3 之後再試。
- **模型不經過伺服器**:照 `qwen3vl-genie/push.ps1` 的做法,在 Windows 用 `curl.exe -C -` 下載(可續傳),
  再 `adb push`。伺服器磁碟已經 98% 滿,而且伺服器到 Windows 本來就只靠 git,21 GB 走不了。

---

## 5. 里程碑

### M0 板子盤點(10 分鐘,唯讀)—— ✅ 2026-09-11 完成,結果見第 2 節

在板子上(adb shell)整段貼上:

```sh
grep -E 'MemTotal|MemAvailable' /proc/meminfo
df -h
nproc; grep -m1 -i features /proc/cpuinfo
grep Cpus_allowed_list /proc/self/status
cat /sys/fs/cgroup/system.slice/cpuset.cpus.effective /sys/fs/cgroup/cpuset.cpus.effective
ls -l /dev/fastrpc-*
systemctl list-units --type=service --state=running | grep -iE 'geniex|spirit|vlm'
```

要回答:
- [x] 記憶體還是約 33 GB 嗎?扣掉常駐服務後剩多少? → 33.0 GiB / 可用 29.4 GiB
- [x] 哪個可寫分割區放得下 21 GB? → `/`(otaroot)剩 164 GB,放 `/opt/llamacpp/`
- [x] CPU features 有沒有 `asimddp` / `i8mm` → 有 dotprod、沒有 i8mm
- [x] 目前有哪些 AI 服務在跑 → `spirit-geniex`、`spirit-speechd`

多行貼進 adb shell 時,命令回顯和輸出會交錯(第一段就是),M1 起改用推上去的腳本。

### M1 CPU baseline

**建置(伺服器,已完成):** `host/build-cpu.sh` 用 Yocto 交叉工具鏈編 CPU 版,不需要 docker。
產物在 `pkg-cpu/`(46 MB,**進 git** —— 只有伺服器編得出來,git 是送到 Windows 的唯一路徑):
`bin/` 四支靜態連結 libstdc++ 的執行檔 + `bench.sh` + `VERSION`。
執行檔只依賴 `libc.so.6` / `libm.so.6`,但要 **GLIBC_2.43**(sysroot 的版本);
`bench.sh` 第一步會先跑 `llama-cli --version`,板上 glibc 太舊會在那裡直接報錯。
改了 `device/*.sh` 也要重跑 `host/build-cpu.sh`,它會一起複製進 `pkg-cpu/`。

**送上板並跑 bench(Windows,git pull 之後):**

```bat
push.bat -Bench
```

push.ps1 會依序:在 Windows 下載 9B(`models\`,可續傳)→ 推 llama.cpp 與腳本到 `/opt/llamacpp`
→ 推模型(板上大小一致就跳過)→ 暫停原本在跑的 spirit 服務 → `bench.sh` → 恢復服務。
結果存在板上 `/opt/llamacpp/results/`。

再做一次實際問答,確認輸出正常(adb shell;`-rea off` 關掉思考模式,只看速度與是否正常):

```sh
cd /opt/llamacpp && echo $$ > /sys/fs/cgroup/cgroup.procs && ./bin/llama-cli -m models/Qwen_Qwen3.5-9B-Q4_0.gguf -t 8 -st -rea off -n 256 -p "請用繁體中文三句話解釋什麼是 MoE 架構"
```

9B 過關後換 35B:`push.bat -Model 35b -Bench`(20.84 GB,下載與推送都要一段時間,中斷重跑即可續傳)。
不帶 `-Bench` 就只部署不跑。

記錄:pp512 / tg128 tok/s(4 核、8 核)、MemAvailable、輸出是否正常。

過關條件:載得起來、輸出正常、tg 有數字。

### M2 Hexagon NPU offload

- 雙 NSP + 多 virtual session 分攤 21 GB 權重
- `GGML_HEXAGON_VERBOSE=1` 確認實際 offload 落點(不支援的會靜默掉回 CPU)
- `sysMonAppLE_64Bit getstate --q6 cdsp` 看到 `HMX Power: ON` 才算真的跑在 NPU 上

### M3 對照表(交主管)

35B-A3B vs 9B vs 現行 Qwen3-VL-8B(QAIRT 11.31 tok/s)的速度、TTFT、記憶體、回答品質。

### M4(選配)

視覺(mmproj)、`llama-server` 提供 OpenAI 相容 API 給既有 app 用。

---

## 6. 已知風險 / 待解

1. **伺服器 docker 沒權限**:`permission denied ... /var/run/docker.sock`(patrick 不在 docker group)。
   CPU 版已經繞開(`host/build-cpu.sh`,Yocto 工具鏈,cmake 用 Yocto 的 cmake-native);
   **NPU 版一定要 Hexagon SDK**,上游只提供 docker 映像 → M2 開始前要請管理員加群組。
2. **伺服器磁碟剩 92 GB(98%)**:模型改成 Windows 直接下載,不經過伺服器;**不要**抓 BF16(69 GB)回來自己量化。
3. **板上記憶體**:35B 約 21 GiB,兩個 spirit 服務開著時還剩約 8 GiB。測試時建議停掉。
   模型不要放 `/tmp`(tmpfs,吃 RAM)。
4. **板上 glibc 版本未確認**:執行檔要 GLIBC_2.43。太舊的話改成全靜態連結(`-static`)再編一次。
5. **HTP 單一 session 的記憶體上限**:上游範例一個 session 回報 2048 MB,GenieX 筆記約 3.5 GB。
   21 GB 要拆成 6–10 個 session 分到兩顆 NSP,**可不可行是 M2 最大的未知數**。
6. **這塊 image 的 `ADSP_LIBRARY_PATH` 對 QNN skel 無效**(`qwen3vl-genie/device/setup-dsp1-skel.sh` 就是為此寫的)。
   上游的執行方式靠它找 `libggml-htp-v73.so`,可能要比照做 symlink。
   GenieX 內建的 llama.cpp 當初有把 htp-v73 成功載上 cdsp,代表有解,機制待查。
7. **CPU 的 prefill 會慢**:MoE 每個 token 仍要算 3B,長 prompt 的 TTFT 是 CPU 路徑的弱點 —— 這正是 M2 要解的。

---

## 7. 目錄

```
qcs9075-llamacpp/
├── README.md
├── push.bat / push.ps1  Windows 端:下載模型、部署到板子、(選用)跑 bench
├── host/
│   └── build-cpu.sh     伺服器端:交叉編譯 CPU 版 llama.cpp → pkg-cpu/
├── device/
│   └── bench.sh         板端:escape URM 後跑 llama-bench(4 核、8 核)
├── pkg-cpu/             要推上板的東西:bin/ + bench.sh + VERSION(進 git)
├── results/             llama-bench 輸出與比較表
├── models/              Windows 端下載的 GGUF(不進 git)
├── llama.cpp/           上游原始碼(git clone --depth 1,不進 git)
└── build-cpu/           建置目錄(不進 git)
```

`push.bat` 刻意是純 ASCII + CRLF(繁中 Windows 的 cmd 用 cp950 讀,中文位元組會讓它閃退);
`push.ps1` 是 UTF-8 BOM(PS 5.1 沒有 BOM 會用 cp950 讀);`.gitattributes` 讓 git 不動這兩個檔的換行。

板上的位置:`/opt/llamacpp/{bin,bench.sh,VERSION,models,results}`。

---

## 8. 參考來源

- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 架構、benchmark
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — 架構、benchmark
- [bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) — Q4_0 / IQ4_NL 大小、b9222
- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — MXFP4_MOE / UD 系列大小
- [ollama qwen3.6 tags](https://ollama.com/library/qwen3.6/tags)
- [llama.cpp Snapdragon README](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) / [linux.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md) — 建置、`GGML_HEXAGON_DEVICES`
- [llama.cpp docs/ops.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) — 各 backend 算子支援表
- 內部:`docs/Qwen3VL_Tutorial研究_與_Qwen3.5-9B可行性評估.md`(8/20,結論「先用 GGUF 跑 9B 驗證」)
- 內部:`qcs9075-tdp90-stress/TECHNICAL.md`(RAM、DDR 頻寬、Adreno 663)
