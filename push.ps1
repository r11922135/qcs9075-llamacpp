# =============================================================================
# push.ps1 — 把 llama.cpp(CPU 版)與 GGUF 模型部署到 QCS9075 板子
#
#   只要 adb 在 PATH 上。模型由板子自己從 Hugging Face 下載(device/fetch-model.sh),
#   不佔 Windows 空間,板子要能上網(WiFi)。
#   板子重燒 image 後要重跑一次(/opt 在 rootfs 上,會被清掉)。
#
#   用法(PowerShell 不會執行目前資料夾的指令,要加 .\ 前綴;或直接點兩下 push.bat):
#       .\push.bat                   # llama.cpp + Qwen3.5-9B Q4_0(5.74 GB)
#       .\push.bat -Model 35b        # llama.cpp + Qwen3.6-35B-A3B Q4_0(20.84 GB)
#       .\push.bat -Model both
#       .\push.bat -Model none       # 只更新 llama.cpp 與腳本
#       .\push.bat -Bench            # 推完直接跑 bench.sh(會暫停 spirit 服務,跑完恢復)
#
#   中斷了直接重跑:板上的下載會續傳,已經完整的模型會跳過。
# =============================================================================
param(
    [ValidateSet('9b', '35b', 'both', 'none')]
    [string]$Model = '9b',
    [switch]$Bench,
    [string]$Dest = '/opt/llamacpp'
)

$ErrorActionPreference = 'Stop'

# 板端腳本用繁中輸出;不把 console 切到 UTF-8,Write-Host 出來會是亂碼。
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$Here   = $PSScriptRoot
$PkgDir = Join-Path $Here 'pkg-cpu'

# 下載來源、大小、sha256 都在 device/fetch-model.sh;這裡只需要檔名給 bench 用。
$Catalog = @{
    '9b'  = [pscustomobject]@{ Key = '9b';  File = 'Qwen_Qwen3.5-9B-Q4_0.gguf' }
    '35b' = [pscustomobject]@{ Key = '35b'; File = 'Qwen_Qwen3.6-35B-A3B-Q4_0.gguf' }
}
# 外層 @() 一定要包:只選一顆時 switch 回傳的是純量,
# 而 PS 5.1 的 [pscustomobject] 沒有 .Count,後面的判斷會失準。
$Wanted = @(switch ($Model) {
    '9b'   { $Catalog['9b'] }
    '35b'  { $Catalog['35b'] }
    'both' { $Catalog['9b']; $Catalog['35b'] }
    'none' { }
})

function Ok   ($m) { Write-Host "[ OK ] $m" -ForegroundColor Green }
function Info ($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

# adb 會把正常訊息寫到 stderr。用 `& adb ... 2>&1` 的話,在 ErrorActionPreference='Stop'
# 之下會被轉成終止性錯誤。沿用 qwen3vl-genie push.ps1 的做法:用 Start-Process 避開。
$script:AdbExit = 0
function AdbRaw ([string[]] $ArgList) {
    $o = [IO.Path]::GetTempFileName()
    $e = [IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath 'adb' -ArgumentList $ArgList -NoNewWindow -Wait -PassThru `
                           -RedirectStandardOutput $o -RedirectStandardError $e
        $script:AdbExit = $p.ExitCode
        # 板端輸出是 UTF-8;PS 5.1 的 Get-Content 預設用 CP950 讀,不指定會亂碼。
        $t = ''
        $t += [string](Get-Content $o -Raw -Encoding UTF8 -ErrorAction SilentlyContinue)
        $t += [string](Get-Content $e -Raw -Encoding UTF8 -ErrorAction SilentlyContinue)
        return $t
    } finally {
        Remove-Item $o, $e -Force -ErrorAction SilentlyContinue
    }
}

function AdbShell ($cmd) { return (AdbRaw @('shell', $cmd)).Trim() }

# 大檔與 bench 要看得到即時輸出,所以不重導。
# Start-Process 會把參數用空白直接接起來,本機路徑要自己加引號,否則路徑有空白就斷掉。
function AdbLive ([string[]] $ArgList) {
    $p = Start-Process -FilePath 'adb' -ArgumentList $ArgList -NoNewWindow -Wait -PassThru
    $script:AdbExit = $p.ExitCode
    return $p.ExitCode
}

function Quote ($path) { return '"' + $path + '"' }

Write-Host ''
Write-Host '==============================================================' -ForegroundColor White
Write-Host ' llama.cpp (CPU) -> QCS9075' -ForegroundColor White
Write-Host '==============================================================' -ForegroundColor White
Write-Host ''

# ---------------------------------------------------------------------------
# 0. 本機檔案與 adb
# ---------------------------------------------------------------------------
$bins = @(Get-ChildItem (Join-Path $PkgDir 'bin') -File -ErrorAction SilentlyContinue)
if ($bins.Count -lt 4) { Die "找不到 $PkgDir\bin 的執行檔。先 git pull(執行檔是伺服器用 host/build-cpu.sh 編好 commit 進來的)。" }
$scripts = @(Get-ChildItem $PkgDir -File)
Ok "本機 llama.cpp:$((Get-Content (Join-Path $PkgDir 'VERSION') -TotalCount 1))"

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Die 'PATH 上找不到 adb。裝 platform-tools 或把它加進 PATH 再跑一次。'
}
# @() 一定要包 —— 只接一台板子時 Where-Object 回傳的是純量,PS 5.1 下 .Count 會失準
$devs = @((AdbRaw @('devices')) -split "`n" | Where-Object { $_ -match "`tdevice" })
if ($devs.Count -lt 1) { Die 'adb 看不到板子(adb devices 是空的)。檢查 USB 線與板子是否開機。' }
if ($devs.Count -gt 1) { Warn "接了多台裝置,adb 會用預設那台:$($devs -join ' / ')" }
Ok "板子已連線:$(($devs[0] -split "`t")[0])"

# ---------------------------------------------------------------------------
# 1. 推 llama.cpp 與腳本(不到 50 MB,每次都推,省得比對)
# ---------------------------------------------------------------------------
AdbShell "mkdir -p $Dest/bin $Dest/models $Dest/results" | Out-Null

foreach ($b in $bins) {
    AdbRaw @('push', (Quote $b.FullName), "$Dest/bin/$($b.Name)") | Out-Null
    if ($script:AdbExit -ne 0) { Die "push $($b.Name) 失敗" }
}
# adb push 不保留執行權限
AdbShell "chmod 755 $Dest/bin/*" | Out-Null

# .sh 若被 Windows 的 git 轉成 CRLF,板上的 sh 會跑不動;推之前一律轉回 LF。
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($f in $scripts) {
    $tmp = Join-Path $env:TEMP $f.Name
    [IO.File]::WriteAllText($tmp, ([IO.File]::ReadAllText($f.FullName) -replace "`r`n", "`n"), $utf8NoBom)
    AdbRaw @('push', (Quote $tmp), "$Dest/$($f.Name)") | Out-Null
    if ($script:AdbExit -ne 0) { Die "push $($f.Name) 失敗" }
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}
Ok "llama.cpp 與腳本已推到 $Dest($($bins.Count) 個執行檔、$($scripts.Count) 個檔案)"

# ---------------------------------------------------------------------------
# 2. 板子自己下載模型(已完整就跳過;中斷後重跑會續傳)
# ---------------------------------------------------------------------------
# 關掉這個視窗會連帶中斷板上的下載,重跑 push.bat 即可續傳。
foreach ($m in $Wanted) {
    Write-Host ''
    Info "板上下載 $($m.File)"
    if ((AdbLive @('shell', "sh $Dest/fetch-model.sh $($m.Key)")) -ne 0) {
        Die '板上下載沒完成(原因看上面)。重跑 push.bat 會從中斷點續傳。'
    }
}

# ---------------------------------------------------------------------------
# 3.(選用)跑 bench.sh
# ---------------------------------------------------------------------------
if ($Bench) {
    if ($Wanted.Count -eq 0) { Die '-Bench 需要搭配 -Model 9b / 35b / both' }
    # spirit 服務會搶 CPU 與記憶體。只停原本在跑的,跑完恢復原狀。
    $active = @()
    foreach ($svc in @('spirit-geniex', 'spirit-speechd')) {
        if ((AdbShell "systemctl is-active $svc") -eq 'active') { $active += $svc }
    }
    if ($active.Count -gt 0) {
        Info "暫停 $($active -join ', ')(跑完會恢復)"
        AdbShell "systemctl stop $($active -join ' ')" | Out-Null
    }
    try {
        foreach ($m in $Wanted) {
            Write-Host ''
            Info "bench:$($m.File)(4 核、8 核各三輪,要幾分鐘)"
            if ((AdbLive @('shell', "sh $Dest/bench.sh models/$($m.File)")) -ne 0) { Warn "bench.sh 離開碼 $($script:AdbExit)" }
        }
    } finally {
        if ($active.Count -gt 0) {
            AdbShell "systemctl start $($active -join ' ')" | Out-Null
            Info "已恢復 $($active -join ', ')"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. 接下來
# ---------------------------------------------------------------------------
Write-Host ''
Ok '完成'
if ($Wanted.Count -gt 0) {
    $f = $Wanted[0].File
    Write-Host ''
    Write-Host '實際問答(adb shell 裡貼上;-rea off 關掉思考模式):' -ForegroundColor White
    Write-Host "  cd $Dest && echo `$`$ > /sys/fs/cgroup/cgroup.procs && ./bin/llama-cli -m models/$f -t 8 -st -rea off -n 256 -p '請用繁體中文三句話解釋什麼是 MoE 架構'"
}
