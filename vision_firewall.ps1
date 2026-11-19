#Requires -RunAsAdministrator
<#
For the selected vision ports on a Windows Public Wi-Fi network. Run the vision server first.
Allow only the specified iPhone on the listening interface. -Undo restores the preceding TCP rule.
#>
[CmdletBinding(SupportsShouldProcess)]
param([ipaddress]$PhoneIP, [ValidateRange(1025,65534)][int]$CameraPort=8843, [switch]$Undo)
$ErrorActionPreference = 'Stop'
$name = 'TwinDepot-Vision-iPhone'
$setupPort = $CameraPort - 1
if ($CameraPort -ne 8843) { $name = "TwinDepot-Vision-iPhone-$CameraPort" }
$remainder = @("1-$($setupPort-1)", "$($CameraPort+1)-65535")
$existing = Get-NetFirewallRule -PolicyStore PersistentStore -Name $name -ErrorAction SilentlyContinue
if ($Undo) {
    if (-not $existing) { return }
    if ($PSCmdlet.ShouldProcess($name, 'Restore previous TCP block and remove camera allowance')) {
        if ($existing.Description) {
            $block = Get-NetFirewallRule -PolicyStore PersistentStore -Name $existing.Description
            $ports = ($block | Get-NetFirewallPortFilter).LocalPort
            if (Compare-Object @($ports) $remainder) { throw 'The TCP rule changed independently; review it before restoring.' }
            $block | Set-NetFirewallRule -Protocol TCP -LocalPort Any
        }
        $existing | Remove-NetFirewallRule
    }
    return
}
if (-not $PhoneIP -or $PhoneIP.AddressFamily -ne 'InterNetwork') { throw 'Specify the iPhone Wi-Fi IPv4 address with -PhoneIP.' }
if ($existing) { throw 'A camera rule already exists. Use -Undo before pairing another address.' }
$listener = @(Get-NetTCPConnection -State Listen -LocalPort $CameraPort | Where-Object {$_.LocalAddress -notin @('127.0.0.1','::1')})
if ($listener.Count -ne 1) { throw 'Start twin workcell or twin vision on the requested camera port first.' }
$proc = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $listener[0].OwningProcess)
if ($proc.CommandLine -notmatch 'twin\.py[" ]+(vision|workcell)\b') { throw 'The camera port is owned by a different application.' }
$nic = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $listener[0].LocalAddress
if ((Get-NetConnectionProfile -InterfaceIndex $nic.InterfaceIndex).NetworkCategory -ne 'Public') { throw 'This helper applies only to a Public Wi-Fi profile.' }
$blocks = @(Get-NetFirewallRule -PolicyStore PersistentStore -Direction Inbound -Action Block -Enabled True |
    Where-Object { $_.Profile -eq 'Public' -and $_.Name -like 'TCP Query User*' } |
    Where-Object { ($_ | Get-NetFirewallApplicationFilter).Program -eq $proc.ExecutablePath } |
    Where-Object { $p = $_ | Get-NetFirewallPortFilter; $p.Protocol -eq 'TCP' -and $p.LocalPort -eq 'Any' })
if ($blocks.Count -gt 1) { throw 'Multiple overlapping TCP block rules; review them in Windows Firewall.' }
if ($PSCmdlet.ShouldProcess($proc.ExecutablePath, "Allow $PhoneIP on $($nic.InterfaceAlias) TCP $setupPort-$CameraPort")) {
    $options = @{Name=$name; DisplayName='Twin Depot Camera (paired iPhone)'; Direction='Inbound'; Action='Allow';
        Profile='Public'; InterfaceAlias=$nic.InterfaceAlias; Program=$proc.ExecutablePath; Protocol='TCP';
        LocalAddress=$nic.IPAddress; LocalPort=@("$setupPort","$CameraPort"); RemoteAddress=$PhoneIP.IPAddressToString;
        PolicyStore='PersistentStore'}
    if ($blocks.Count) { $options.Description=$blocks[0].Name }
    $allow = New-NetFirewallRule @options
    if ($blocks.Count) {
        try { $blocks[0] | Set-NetFirewallRule -Protocol TCP -LocalPort $remainder }
        catch { $allow | Remove-NetFirewallRule; throw }
    }
}
