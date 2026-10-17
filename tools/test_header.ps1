$f = [System.IO.File]::OpenRead('D:\affan_ai_engine_2_mini\models\affan_ai_engine_2_mini.affa')
$br = New-Object System.IO.BinaryReader($f)
$magic = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
$ver = $br.ReadUInt32()
$hlen = $br.ReadUInt64()
Write-Host "Magic: $magic Version: $ver HeaderLen: $hlen"
$hdrBytes = $br.ReadBytes([int]$hlen)
$hdrText = [System.Text.Encoding]::UTF8.GetString($hdrBytes)
$lines = $hdrText.Split([char]10)
for ($i=0; $i -lt [Math]::Min(25, $lines.Length); $i++) {
    Write-Host $lines[$i]
}
$f.Close()
