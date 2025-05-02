# run_simulations.ps1

# Check if directory is provided
param (
    [Parameter(Mandatory=$true)]
    [string]$ConfigDirectory
)

# Check if parameter is missing (despite mandatory)
if (-not $ConfigDirectory) {
    Write-Host "Usage: .\run_simulations.ps1 -ConfigDirectory <config_directory>"
    exit 1
}

$EXPERIMENTS_DIR = "experiments_dir"

# Check if the config directory exists
if (-not (Test-Path -Path $ConfigDirectory -PathType Container)) {
    Write-Host "Error: Directory $ConfigDirectory not found"
    exit 1
}

# Find all JSON files in the config directory
$CONFIG_FILES = Get-ChildItem -Path $ConfigDirectory -Filter "*.json" -File -Recurse

if ($CONFIG_FILES.Count -eq 0) {
    Write-Host "No JSON configuration files found in $ConfigDirectory"
    exit 1
}

# Process each configuration file
foreach ($CONFIG_FILE in $CONFIG_FILES) {
    # Get the config filename without extension
    $DIRNAME = Split-Path -Path (Split-Path -Path $CONFIG_FILE.FullName -Parent) -Leaf
    $CONFIG_NAME = [System.IO.Path]::GetFileNameWithoutExtension($CONFIG_FILE.Name)
    
    Write-Host "Running simulation with config: $CONFIG_NAME"
    
    # Create experiment result directory
    if (-not (Test-Path -Path "$EXPERIMENTS_DIR\$DIRNAME")) {
        New-Item -Path "$EXPERIMENTS_DIR\$DIRNAME" -ItemType Directory -Force
    }

    $RESULT_DIR = "$EXPERIMENTS_DIR\$DIRNAME\$CONFIG_NAME"
    if (-not (Test-Path -Path $RESULT_DIR)) {
        New-Item -Path $RESULT_DIR -ItemType Directory -Force
    }
    
    # Run the simulation with this config
    python main.py --config $CONFIG_FILE.FullName --output $RESULT_DIR
    
    # Check if simulation was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Simulation completed successfully for $CONFIG_NAME"
    } else {
        Write-Host "Error running simulation for $CONFIG_NAME"
    }
    
    Write-Host ("-" * 40)
}

Write-Host "All simulations completed!"