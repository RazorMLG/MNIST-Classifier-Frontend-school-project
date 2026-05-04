# Docs

This directory holds architecture notes that support the MNIST project.

- Use `../CONTEXT.md` for shared project language and relationships.
- Use `./adr/` for decisions that are hard to reverse, surprising without context, and the result of a real trade-off.

## Install And Setup In VS Code

This project is meant to be run locally from the repository root in Visual Studio Code. The default and supported startup path on Windows is the root `start.ps1` script.

### What You Need Before You Start

Install these tools on your machine first:

1. Visual Studio Code
2. Git
3. Python 3
4. Node.js with `npm`
5. Windows PowerShell

Recommended VS Code extensions:

1. Python
2. Pylance

Notes:

- The project uses a Python backend with FastAPI and a React frontend with Vite.
- The first startup can take a while because Python packages and frontend packages are installed locally.
- The backend startup expects either `py -3` or `python` to be available on your machine.

### Step 1: Clone The Repository

Clone the repository to a local folder:

```powershell
git clone <your-repository-url>
cd "MNIST Projekat"
```

If you already have the source code, just make sure the whole project is available locally and that you can open the repository root in VS Code.

### Step 2: Open The Project In VS Code

1. Open Visual Studio Code.
2. Select `File -> Open Folder...`.
3. Choose the repository root folder.
4. If VS Code asks whether you trust the workspace, accept it.

You should open the root folder, not only `frontend/` or `backend/`, because the startup script and shared project files live at the repository root.

### Step 3: Open A PowerShell Terminal

1. In VS Code, open `Terminal -> New Terminal`.
2. Make sure the terminal is PowerShell.
3. Make sure the working directory is the repository root.

If needed, switch to PowerShell from the terminal profile menu in VS Code.

### Step 4: Run The Project With The Supported Startup Script

From the repository root, run:

```powershell
.\start.ps1
```

What `start.ps1` does for you:

1. Creates a local Python virtual environment in `.venv` if it does not exist yet.
2. Installs Python dependencies from `requirements.txt`.
3. Installs root Node dependencies if they are missing.
4. Installs frontend dependencies in `frontend/node_modules` if they are missing.
5. Starts the backend and frontend development servers together.

The root npm development command starts:

1. FastAPI backend on `http://127.0.0.1:8000`
2. Vite frontend on `http://127.0.0.1:5173`

### Step 5: If PowerShell Blocks Script Execution

If PowerShell refuses to run `start.ps1`, run this once in the same VS Code terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then run:

```powershell
.\start.ps1
```

This only changes the execution policy for the current terminal session.

### Step 6: Verify That Everything Started Correctly

After startup completes, open these URLs:

1. Frontend: `http://127.0.0.1:5173`
2. Backend health endpoint: `http://127.0.0.1:8000/api/health`
3. FastAPI docs: `http://127.0.0.1:8000/docs`

If the backend is healthy, the health endpoint should return a JSON response with status `ok`.

### Step 7: Point VS Code At The Local Python Environment

After the first run creates `.venv`, configure VS Code to use it:

1. Open the Command Palette.
2. Run `Python: Select Interpreter`.
3. Choose the interpreter inside `.venv\Scripts\python.exe`.

This helps VS Code use the correct interpreter for running tests, resolving imports, and showing Python diagnostics.

### Step 8: Manual Setup If You Do Not Want To Use `start.ps1`

The project is designed around `start.ps1`, but you can also run the setup manually.

#### Python Backend Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py` is not available, use `python -m venv .venv` instead.

#### Node Dependency Setup

```powershell
npm install
npm install --prefix frontend
```

#### Start The Development Servers

```powershell
npm run dev
```

That command starts both the backend and frontend together.

### Step 9: Useful Development Commands

Run these from the repository root:

```powershell
npm test
```

Runs backend and frontend tests.

```powershell
npm run test:backend
```

Runs only the backend test suite.

```powershell
npm run test:frontend
```

Runs only the frontend test suite.

```powershell
npm run typecheck
```

Runs the frontend TypeScript typecheck.

```powershell
npm run train:shipped-models
```

Retrains the shipped model set using `train.csv` and the `data/` storage area.

### Step 10: Project Files And Folders That Matter For Setup

These are the main files a new developer should know about:

1. `start.ps1` - the main Windows startup script
2. `requirements.txt` - Python dependencies
3. `package.json` - root scripts for running tests and both servers together
4. `frontend/package.json` - frontend-specific scripts and dependencies
5. `backend/app/main.py` - FastAPI application entry point
6. `data/` - model storage and registry data

### Step 11: Common Problems And Fixes

#### Python Is Not Found

Install Python 3 and make sure one of these works in a terminal:

```powershell
py -3 --version
python --version
```

#### `npm` Is Not Found

Install Node.js, reopen VS Code, and make sure this works:

```powershell
npm --version
```

#### The First Startup Takes A Long Time

That is expected on the first run because the script installs Python dependencies and Node dependencies locally.

#### Ports 8000 Or 5173 Are Already In Use

Stop the conflicting process and rerun `npm run dev` or `.\start.ps1`.

#### VS Code Does Not Recognize Python Imports

Make sure the selected interpreter is the one inside `.venv`.

## Version-One Guidance

- Use the root `start.ps1` script as the only visible startup command on Windows. It bootstraps the Python environment and frontend dependencies before launching the app, so version one does not add a separate user-facing setup step.
- Custom training requires a manual model name. The interface does not auto-suggest or auto-generate defaults.
- Failed training runs only show inline feedback for the current run and do not create failed entries in the shared model catalog.
