# 星算语工具生态 · 全链验证入口
# 一条命令跑通:星算语源码 → 编译器(带门禁) → XL01 → 校验器 → golden
# 用法: pwsh7 xl_chain.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\..\toolchain\env.ps1" | Out-Null
$H = "D:\StarSnow_Home\languages\fromzero"

Write-Host ""
Write-Host "=== 星算语工具生态 · 全链验证 ===" -ForegroundColor Cyan
Write-Host "链: 源码 -> 编译器(门禁) -> XL01 -> 校验器 -> golden" -ForegroundColor DarkGray

# 1. 源码 -> 编译器(带门禁) -> XL01 字节码
Set-Location "$H\xl_compiler"
Write-Host "`n[1/3] 编译器 xl_compiler (门禁内嵌)" -ForegroundColor Yellow
& cargo run --quiet 2>&1 | Select-String "门禁通过|写出|error"

# 2. 字节码 -> Rust 校验器门禁
Set-Location "$H\rust_gate"
Write-Host "`n[2/3] 校验器 rust_gate" -ForegroundColor Yellow
& cargo run --quiet -- "$H\xl_compiler\out.xlbin" 2>&1 | Select-String "指令数|校验通过|error"

# 3. golden 全链回归
Write-Host "`n[3/3] golden 回归" -ForegroundColor Yellow
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 "$H\_golden_test.py" 2>&1 | Select-Object -Last 2

Write-Host ""
Write-Host "=== 星算语工具生态 · 全链验证完成 ===" -ForegroundColor Green
