$html = [System.IO.File]::ReadAllBytes('D:\affan_ai_engine_2_mini\ui\index.html')
$logo = [System.IO.File]::ReadAllBytes('D:\affan_ai_engine_2_mini\ui\logo.png')

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('#pragma once')
[void]$sb.AppendLine('// Embedded UI Assets (Procedural Logo + Minimalist ChatGPT/LM Studio UI)')
[void]$sb.AppendLine('const unsigned char kIndexHtml[] = {')

for ($i=0; $i -lt $html.Length; $i++) {
    if ($i % 16 -eq 0) { [void]$sb.Append('  ') }
    [void]$sb.Append(('0x{0:x2}, ' -f $html[$i]))
    if ($i % 16 -eq 15) { [void]$sb.AppendLine() }
}
if ($html.Length % 16 -ne 0) { [void]$sb.AppendLine() }
[void]$sb.AppendLine('};')
[void]$sb.AppendLine(('const unsigned int kIndexHtmlLen = {0};' -f $html.Length))

[void]$sb.AppendLine('const unsigned char kLogoPng[] = {')
for ($i=0; $i -lt $logo.Length; $i++) {
    if ($i % 16 -eq 0) { [void]$sb.Append('  ') }
    [void]$sb.Append(('0x{0:x2}, ' -f $logo[$i]))
    if ($i % 16 -eq 15) { [void]$sb.AppendLine() }
}
if ($logo.Length % 16 -ne 0) { [void]$sb.AppendLine() }
[void]$sb.AppendLine('};')
[void]$sb.AppendLine(('const unsigned int kLogoPngLen = {0};' -f $logo.Length))

[System.IO.File]::WriteAllText('D:\affan_ai_engine_2_mini\engine_cpp\ui_assets.h', $sb.ToString())
Write-Host ('ui_assets.h regenerated: HTML len=' + $html.Length + ', Logo len=' + $logo.Length)
