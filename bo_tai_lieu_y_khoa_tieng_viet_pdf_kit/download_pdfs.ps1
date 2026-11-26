# Tải PDF gốc từ danh sách trong sources.json — KHÔNG chuyển Markdown.
# PowerShell 5.1+ hoặc PowerShell 7+. Chỉ tải 8 bài có giấy phép CC ghi trên trang bài.
# Các bài melanoma thiếu bằng chứng rights chỉ có trong danh mục để tra cứu, không tải tự động.
param([string]$OutputDir = "", [int]$MaxFileMB = 10)
$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PSScriptRoot 'sources.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $PSScriptRoot 'pdf_originals' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$maxBytes = [int64]$MaxFileMB * 1024 * 1024
$results = [System.Collections.Generic.List[object]]::new()
foreach ($item in $manifest.documents) {
    if (-not $item.download_by_default) { continue }
    $uri = [Uri]$item.pdf_url
    if ($uri.Scheme -ne 'https' -or $uri.Host -notin @('thaibinhjmp.vn', 'tapchikhcnnd.mod.gov.vn')) {
        Write-Warning "Blocked unreviewed host: $($item.id)"
        continue
    }
    $target = Join-Path $OutputDir $item.filename
    $tmp = "$target.part"
    if (Test-Path -LiteralPath $target) {
        Write-Host "SKIP exists: $($item.filename)"
        $results.Add([pscustomobject]@{id=$item.id;file=$target;status='existing_not_overwritten';bytes=(Get-Item -LiteralPath $target).Length;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant();source_url=$item.article_url;pdf_url=$item.pdf_url;license=$item.license})
        continue
    }
    try {
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
        Write-Host "DOWNLOAD $($item.id)"
        Invoke-WebRequest -Uri $uri.AbsoluteUri -OutFile $tmp -MaximumRedirection 0 -TimeoutSec 45 -UseBasicParsing
        $length = (Get-Item -LiteralPath $tmp).Length
        if ($length -gt $maxBytes -or $length -lt 1000) { throw "PDF size invalid: $length bytes" }
        $stream = [System.IO.File]::OpenRead($tmp)
        try {
            $first = New-Object byte[] 5
            $n = $stream.Read($first,0,5)
            if ($n -ne 5 -or [System.Text.Encoding]::ASCII.GetString($first) -ne '%PDF-') { throw 'Downloaded file is not a PDF' }
        } finally { $stream.Dispose() }
        Move-Item -LiteralPath $tmp -Destination $target
        $sha = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host "OK $($item.filename) size=$length sha256=$sha"
        $results.Add([pscustomobject]@{id=$item.id;file=$target;status='downloaded_valid_pdf_header';bytes=$length;sha256=$sha;source_url=$item.article_url;pdf_url=$item.pdf_url;license=$item.license})
    } catch {
        Write-Warning "FAILED $($item.id): $($_.Exception.Message)"
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
        $results.Add([pscustomobject]@{id=$item.id;file=$target;status='failed';bytes=0;sha256='';source_url=$item.article_url;pdf_url=$item.pdf_url;license=$item.license})
    }
    Start-Sleep -Milliseconds 1200
}
$outReport = Join-Path $OutputDir 'download_report.csv'
$results | Export-Csv -LiteralPath $outReport -NoTypeInformation -Encoding UTF8
Write-Host "`nReport: $outReport"
Write-Host "Successful or previously existing: $(($results | Where-Object {$_.status -ne 'failed'} | Measure-Object).Count) / $(($results | Measure-Object).Count)"
