#!/usr/bin/env bash
# 本地 ASR 验证案例。在 /ai/project/nevoflux-agent 下运行。
set -uo pipefail
cd /ai/project/nevoflux-agent
export ORT_DYLIB_PATH="$PWD/target/release/lib/libonnxruntime.so"
BIN=./target/release/examples/transcribe
pass=0; fail=0
check() { # check <描述> <期望子串> <实际>
  if [[ "$3" == *"$2"* ]]; then echo "  ✅ $1"; pass=$((pass+1));
  else echo "  ❌ $1"; echo "       期望含: $2"; echo "       实际:   $3"; fail=$((fail+1)); fi
}

echo "── 1. 中文（核心诉求）"
out=$($BIN crates/asr/tests/fixtures/zh.wav zh 2>&1)
check "转写正确"        "开饭时间早上9点至下午5点" "$(grep '^text' <<<"$out")"
check "ITN 生效（9 而非九）" "9点"                  "$(grep '^text' <<<"$out")"
check "语言判定 zh"      "zh"                       "$(grep '^language' <<<"$out")"
echo "     $(grep -E '^inference' <<<"$out")"

echo "── 2. 语言自动检测（不传 language）"
for w in zh en ja yue; do
  out=$($BIN crates/asr/tests/fixtures/$w.wav 2>&1)
  check "$w.wav 自动判为 $w" "$w" "$(grep '^language' <<<"$out")"
done

echo "── 3. VAD 分段：多语言混合（不分段会丢内容）"
python3 - <<'PY'
import struct
def data(p):
    b=open(p,'rb').read(); pos=12
    while pos+8<=len(b):
        cid=b[pos:pos+4]; n=struct.unpack('<I',b[pos+4:pos+8])[0]; body=pos+8
        if cid==b'data': return b[body:body+n]
        pos=body+n+(n&1)
d=b''.join(data(f"crates/asr/tests/fixtures/{w}.wav")+b'\x00\x00'*16000 for w in ("zh","en","yue","ja"))
h=b'RIFF'+struct.pack('<I',36+len(d))+b'WAVEfmt '+struct.pack('<IHHIIHH',16,1,1,16000,32000,2,16)+b'data'+struct.pack('<I',len(d))
open('/tmp/verify-mixed.wav','wb').write(h+d)
PY
out=$(FORCE_VAD=1 $BIN /tmp/verify-mixed.wav 2>&1)
t=$(grep '^text' <<<"$out")
check "保留中文" "时间早上9点"       "$t"
check "保留英文" "tribal chieftain"  "$t"
check "保留粤语" "表达唔到"          "$t"
check "保留日语" "弁当制"            "$t"
echo "     $(grep -E '^(vad|inference)' <<<"$out" | tr '\n' ' ')"
echo "  对照——同一段不分段（应只剩一种语言）："
one=$($BIN /tmp/verify-mixed.wav 2>&1 | grep '^text' | cut -c1-80)
echo "     $one"

echo "── 4. 时间戳落在音频内"
out=$(FORCE_VAD=1 $BIN /tmp/verify-mixed.wav 2>&1)
dur=$(python3 -c "import os;print(int((os.path.getsize('/tmp/verify-mixed.wav')-44)/2/16))")
last=$(grep -oE '\.\.[[:space:]]*[0-9]+ ms' <<<"$out" | grep -oE '[0-9]+' | tail -1)
if [ -n "$last" ] && [ "$last" -le "$dur" ]; then echo "  ✅ 末段 ${last}ms ≤ 音频 ${dur}ms"; pass=$((pass+1));
else echo "  ❌ 末段 ${last}ms 超出音频 ${dur}ms"; fail=$((fail+1)); fi

echo "── 5. Whisper（需 --features whisper 构建 + just whisper-model）"
if [ -f "$HOME/.cache/nevoflux/models/whisper-tiny/model.safetensors" ]; then
  out=$(WHISPER=tiny $BIN crates/asr/tests/fixtures/ja.wav 2>&1)
  check "日语自动检测" "ja" "$(grep '^language' <<<"$out")"
  check "输出含假名"   "うち" "$(grep '^text' <<<"$out")"
else
  echo "  ⏭  跳过：未下载 whisper 模型"
fi

echo
echo "═══ ${pass} 通过 / ${fail} 失败 ═══"
exit $((fail > 0))
