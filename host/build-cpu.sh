#!/bin/sh
# 在伺服器上交叉編譯 CPU 版 llama.cpp,不需要 docker / Hexagon SDK。
#
# 工具鏈借用 Yocto build tree(與 qwen3vl-genie/native/build.sh 同一套):
# aarch64 gcc/g++ 15.2、帶 libstdc++ 的 sysroot、cmake/ninja native。
# NPU 版要 Hexagon SDK,只能走上游的 snapdragon docker,不在這支腳本的範圍。
#
# 產物是靜態連結的單一執行檔,推上板子不需要 LD_LIBRARY_PATH。
# 路徑都可以用環境變數覆寫,因為那棵 Yocto tree 是別人的,可能會搬。

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
TOP="$(dirname "$HERE")"
SRC="${SRC:-$TOP/llama.cpp}"
OUT="${OUT:-$TOP/build-cpu}"
PKG="${PKG:-$TOP/pkg-cpu}"

YOCTO="${YOCTO:-/srv/iq9075-users/coojiin/QLI2-GA/hlos/build/tmp}"
COMPONENTS="$YOCTO/sysroots-components"
CROSS="${CROSS:-$COMPONENTS/x86_64/gcc-cross-aarch64/usr/bin/aarch64-qcom-linux}"
BINUTILS="${BINUTILS:-$COMPONENTS/x86_64/binutils-cross-aarch64/usr/bin/aarch64-qcom-linux}"
SYSROOT="${SYSROOT:-$YOCTO/work/qcs9075_aihub-qcom-linux/make-mod-scripts/1.0/recipe-sysroot}"
CMAKE="${CMAKE:-$COMPONENTS/x86_64/cmake-native/usr/bin/cmake}"
NINJA="${NINJA:-$COMPONENTS/x86_64/ninja-native/usr/bin/ninja}"

# M0 實測(2026-09-11):板上 CPU 有 asimddp/fphp,沒有 i8mm/sve/bf16。
# 與上游 arm64-linux-snapdragon preset 的 -march 相同;開了 i8mm 板上會 SIGILL。
ARM_ARCH="${ARM_ARCH:-armv8.2-a+fp16+dotprod}"
TARGETS="${TARGETS:-llama-cli llama-completion llama-bench llama-server}"

die() { printf '[FAIL] %s\n' "$1" >&2; exit 1; }

[ -x "$CROSS/aarch64-qcom-linux-g++" ] || die "cross g++ not found: $CROSS"
[ -d "$SYSROOT" ]                      || die "sysroot not found: $SYSROOT"
[ -x "$CMAKE" ]                        || die "cmake not found: $CMAKE"
[ -f "$SRC/CMakeLists.txt" ]           || die "llama.cpp source not found: $SRC
       git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $SRC"

# Yocto 的 cmake-native 自己要用 native sysroot 的函式庫,
# 不給會報 "libbz2.so.1: cannot open shared object file"。
NATIVE_LIBS="$(ls -d "$COMPONENTS"/x86_64/*/usr/lib | tr '\n' ':')"
cmake_() { LD_LIBRARY_PATH="$NATIVE_LIBS" "$CMAKE" "$@"; }

# gcc 找的是裸名的 as/ld,不給會抓到主機的 x86 binutils
# ("as: unrecognized option '-EL'")。用 -B 指到一個放對名字 symlink 的目錄。
TOOLDIR="$OUT/.toolchain"
mkdir -p "$TOOLDIR"
for tool in as ld ar nm objcopy objdump ranlib strip readelf; do
    src="$BINUTILS/aarch64-qcom-linux-$tool"
    [ -f "$src" ] && ln -sf "$src" "$TOOLDIR/$tool"
done

cat > "$OUT/toolchain.cmake" <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT $SYSROOT)
set(CMAKE_C_COMPILER $CROSS/aarch64-qcom-linux-gcc)
set(CMAKE_CXX_COMPILER $CROSS/aarch64-qcom-linux-g++)
set(CMAKE_AR $TOOLDIR/ar)
set(CMAKE_RANLIB $TOOLDIR/ranlib)
set(CMAKE_STRIP $TOOLDIR/strip)
set(CMAKE_C_FLAGS_INIT "-B$TOOLDIR")
set(CMAKE_CXX_FLAGS_INIT "-B$TOOLDIR")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF

printf 'llama.cpp : %s (%s)\n' "$(git -C "$SRC" log -1 --format=%h)" "$(git -C "$SRC" log -1 --format=%cd --date=short)"
printf 'toolchain : %s\n' "$CROSS/aarch64-qcom-linux-g++"
printf 'sysroot   : %s\n' "$SYSROOT"
printf 'march     : %s\n\n' "$ARM_ARCH"

# OpenMP 關掉:不確定板上有沒有 libgomp,改用 ggml 自己的 thread pool。
# OpenSSL 關掉:板上只會讀本機 GGUF,不需要 HTTPS 下載功能。
cmake_ -S "$SRC" -B "$OUT" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_TOOLCHAIN_FILE="$OUT/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH="$ARM_ARCH" \
    -DGGML_OPENMP=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"

# shellcheck disable=SC2086  # TARGETS 是空白分隔的清單
cmake_ --build "$OUT" --target $TARGETS -j "$(nproc)"

rm -rf "$PKG"
mkdir -p "$PKG/bin"
for t in $TARGETS; do
    cp "$OUT/bin/$t" "$PKG/bin/"
done
"$TOOLDIR/strip" "$PKG"/bin/*
cp "$TOP"/device/*.sh "$PKG/"
git -C "$SRC" log -1 --format='llama.cpp %H %cd' --date=short > "$PKG/VERSION"
printf 'march %s\n' "$ARM_ARCH" >> "$PKG/VERSION"

# 板上只需要 glibc 系列的共享函式庫;列出來確認沒有意外的相依。
printf '\n'
for f in "$PKG"/bin/*; do
    need="$("$TOOLDIR/readelf" -d "$f" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' ')"
    glibc="$("$TOOLDIR/readelf" -V "$f" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
    size="$(du -h "$f" | cut -f1)"
    printf '  %-18s %5s  %s(max %s)\n' "$(basename "$f")" "$size" "$need" "$glibc"
done

printf '\n[ OK ] %s\n' "$PKG"
