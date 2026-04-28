$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Resolve-PythonCommand {
    $localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

    if (Test-Path $localPython) {
        return @{
            Command = $localPython
            Arguments = @()
        }
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        return @{
            Command = $pyCommand.Source
            Arguments = @("-3")
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return @{
            Command = $pythonCommand.Source
            Arguments = @()
        }
    }

    throw "Python 3 was not found. Install Python and rerun start.ps1."
}

$bootstrapPython = Resolve-PythonCommand
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    & $bootstrapPython.Command @($bootstrapPython.Arguments + @("-m", "venv", ".venv"))
}

& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path (Join-Path $PSScriptRoot "node_modules\concurrently"))) {
    npm install
}

if (-not (Test-Path (Join-Path $PSScriptRoot "frontend\node_modules\vite"))) {
    npm install --prefix frontend
}

npm run dev
