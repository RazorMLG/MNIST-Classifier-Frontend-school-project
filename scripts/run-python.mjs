import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const localPython = resolve(process.cwd(), ".venv", "Scripts", "python.exe");

function resolvePythonCommand() {
  if (existsSync(localPython)) {
    return { command: localPython, prefixArgs: [] };
  }

  const candidates = [
    { command: "py", prefixArgs: ["-3"] },
    { command: "python", prefixArgs: [] },
  ];

  for (const candidate of candidates) {
    const result = spawnSync(candidate.command, [...candidate.prefixArgs, "--version"], {
      stdio: "ignore",
    });

    if (!result.error && result.status === 0) {
      return candidate;
    }
  }

  throw new Error("Python 3 was not found. Create .venv or install Python and rerun the command.");
}

const python = resolvePythonCommand();
const child = spawn(python.command, [...python.prefixArgs, ...process.argv.slice(2)], {
  stdio: "inherit",
});

child.on("error", (error) => {
  console.error(error.message);
  process.exit(1);
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
