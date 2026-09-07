param(
    [string]$Source = "$PSScriptRoot/../docs/experiment_plan.md",
    [string]$Output = "$PSScriptRoot/../tmp/plan-review/experiment_plan.docx"
)
$ErrorActionPreference = 'Stop'
$Source = [IO.Path]::GetFullPath($Source)
$Output = [IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Output)) | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
    $doc = $word.Documents.Add()
    $doc.PageSetup.PaperSize = 7
    $doc.PageSetup.TopMargin = 56.7
    $doc.PageSetup.BottomMargin = 56.7
    $doc.PageSetup.LeftMargin = 62.4
    $doc.PageSetup.RightMargin = 62.4
    $normal = $doc.Styles.Item(-1)
    $normal.Font.Name = 'Calibri'
    $normal.Font.NameFarEast = '宋体'
    $normal.Font.Size = 10.5
    $normal.ParagraphFormat.SpaceAfter = 4
    $normal.ParagraphFormat.LineSpacingRule = 0
    foreach ($line in [IO.File]::ReadAllLines($Source, [Text.Encoding]::UTF8)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $range = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
        $text = $line
        $style = -1
        if ($line.StartsWith('# ')) { $text = $line.Substring(2); $style = -63 }
        elseif ($line.StartsWith('## ')) { $text = $line.Substring(3); $style = -2 }
        $range.Text = $text + "`r"
        $range.Style = $doc.Styles.Item($style)
        $range.Font.Color = 0
        if ($style -eq -63) {
            $range.Font.NameFarEast = '黑体'
            $range.Font.Size = 20
            $range.Font.Bold = -1
            $range.ParagraphFormat.SpaceAfter = 15
        } elseif ($style -eq -2) {
            $range.Font.NameFarEast = '黑体'
            $range.Font.Size = 13
            $range.Font.Bold = -1
            $range.ParagraphFormat.SpaceBefore = 10
            $range.ParagraphFormat.SpaceAfter = 6
            $range.ParagraphFormat.KeepWithNext = -1
        }
    }
    $tail = $doc.Range($doc.Content.End - 2, $doc.Content.End - 1)
    $tail.Delete() | Out-Null
    $doc.Repaginate()
    $doc.SaveAs2($Output, 16)
    $pdf = [IO.Path]::ChangeExtension($Output, '.pdf')
    $doc.ExportAsFixedFormat($pdf, 17)
    Write-Output "DOCX: $Output"
    Write-Output "PDF: $pdf"
    Write-Output "Pages: $($doc.ComputeStatistics(2))"
} finally {
    if ($null -ne $doc) { $doc.Close(0) }
    $word.Quit()
    [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
