import json
import os
import re
import subprocess
import time
from pathlib import Path
from openai import OpenAI  # 修改：由 anthropic 转为 openai
from dotenv import load_dotenv

load_dotenv(override=True)

# 修改：适配 OpenAI 环境变量逻辑
if os.getenv("OPENAI_BASE_URL"):
    # OpenAI 官方 SDK 会自动处理环境变量，此处保留逻辑一致性
    pass

WORKDIR = Path.cwd()
# 修改：使用 OpenAI 客户端初始化
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
MODEL = os.environ.get("MODEL_ID", "gpt-4-turbo")  # 修改：获取 OpenAI 模型 ID


def detect_repo_root(cwd: Path) -> Path | None:
    """Return git repo root if cwd is inside a repo, else None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)


# -- EventBus: append-only lifecycle events for observability --
class EventBus:
    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(
            self,
            event: str,
            task: dict | None = None,
            worktree: dict | None = None,
            error: str | None = None,
    ):
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)


# -- TaskManager: persistent task board with optional worktree binding --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def list_tasks(self) -> str:
        tasks = []
        for f in self.dir.glob("task_*.json"):
            try:
                tasks.append(json.loads(f.read_text()))
            except Exception:
                pass
        tasks.sort(key=lambda x: x["id"])
        return json.dumps(tasks, indent=2)


# -- WorktreeManager: persistent git worktree lanes --
class WorktreeManager:
    def __init__(self, wt_base: Path):
        self.base = wt_base
        self.base.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base / "index.json"
        if not self.index_file.exists():
            self.index_file.write_text(json.dumps({"worktrees": []}))
        self.git_available = REPO_ROOT.joinpath(".git").exists()

    def _load(self) -> list:
        return json.loads(self.index_file.read_text())["worktrees"]

    def _save(self, worktrees: list):
        self.index_file.write_text(json.dumps({"worktrees": worktrees}, indent=2))

    def create(self, name: str, task_id: int, base_branch: str = "main") -> str:
        if not self.git_available:
            return "Error: Not a git repo."
        wts = self._load()
        if any(w["name"] == name for w in wts):
            return f"Error: Worktree {name} already exists."

        wt_path = self.base / name
        branch = f"wt/{name}"

        try:
            # git worktree add <path> <branch>
            # -b creates a new branch
            subprocess.run(
                ["git", "worktree", "add", "-b", branch, str(wt_path), base_branch],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return f"Error creating worktree: {e.stderr}"

        new_wt = {
            "name": name,
            "path": str(wt_path),
            "branch": branch,
            "task_id": task_id,
            "status": "active",
        }
        wts.append(new_wt)
        self._save(wts)
        return json.dumps(new_wt, indent=2)

    def list_worktrees(self) -> str:
        return json.dumps(self._load(), indent=2)

    def run_in(self, name: str, command: str) -> str:
        wts = self._load()
        wt = next((w for w in wts if w["name"] == name), None)
        if not wt:
            return f"Error: Worktree {name} not found."
        try:
            r = subprocess.run(
                command,
                cwd=wt["path"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        except Exception as e:
            return f"Error running command: {e}"

    def remove(self, name: str, keep_branch: bool = False) -> str:
        wts = self._load()
        wt = next((w for w in wts if w["name"] == name), None)
        if not wt:
            return f"Error: Worktree {name} not found."

        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", name],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            if not keep_branch:
                subprocess.run(
                    ["git", "branch", "-D", wt["branch"]],
                    cwd=REPO_ROOT,
                    check=False,
                )
        except subprocess.CalledProcessError as e:
            return f"Error removing worktree: {e.stderr}"

        new_wts = [w for w in wts if w["name"] != name]
        self._save(new_wts)
        return f"Worktree {name} removed."


# -- Tool definitions --
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")
TASKS = TaskManager(REPO_ROOT / ".tasks")
WORKTREES = WorktreeManager(REPO_ROOT / ".worktrees")


def tool_task_create(subject: str, description: str = ""):
    res = TASKS.create(subject, description)
    EVENTS.emit("task_created", task=json.loads(res))
    return res


def tool_task_list():
    return TASKS.list_tasks()


def tool_task_get(task_id: int):
    return TASKS.get(task_id)


def tool_worktree_create(name: str, task_id: int, base_branch: str = "main"):
    if not TASKS.exists(task_id):
        return f"Error: Task {task_id} does not exist."
    res = WORKTREES.create(name, task_id, base_branch)
    if "Error" not in res:
        wt = json.loads(res)
        TASKS.bind_worktree(task_id, name)
        EVENTS.emit("worktree_created", task={"id": task_id}, worktree=wt)
    return res


def tool_worktree_run(name: str, command: str):
    return WORKTREES.run_in(name, command)


def tool_worktree_close(name: str, keep_branch: bool = False):
    wts = json.loads(WORKTREES.list_worktrees())
    wt = next((w for w in wts if w["name"] == name), None)
    if not wt:
        return f"Error: Worktree {name} not found."
    res = WORKTREES.remove(name, keep_branch)
    TASKS.unbind_worktree(wt["task_id"])
    EVENTS.emit("worktree_closed", task={"id": wt["task_id"]}, worktree=wt)
    return res


def tool_worktree_events(limit: int = 20):
    return EVENTS.list_recent(limit)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "Create a new task in the task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "List all tasks and their status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_create",
            "description": "Create a git worktree for a specific task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique name for the worktree lane."},
                    "task_id": {"type": "integer"},
                    "base_branch": {"type": "string", "default": "main"},
                },
                "required": ["name", "task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_run",
            "description": "Run a shell command inside a specific worktree lane.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["name", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_close",
            "description": "Close a worktree lane and optionally delete its branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "keep_branch": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_events",
            "description": "List recent worktree/task lifecycle events.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
            },
        },
    },
]

TOOL_HANDLERS = {
    "task_create": tool_task_create,
    "task_list": tool_task_list,
    "worktree_create": tool_worktree_create,
    "worktree_run": tool_worktree_run,
    "worktree_close": tool_worktree_close,
    "worktree_events": tool_worktree_events,
}


def agent_loop(messages):
    """主代理循环，适配 OpenAI API 调用"""
    while True:
        # 修改：OpenAI 的 chat.completions.create 接口
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message
        # 修改：OpenAI 消息存储逻辑
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": msg.tool_calls if msg.tool_calls else None
        })

        if not msg.tool_calls:
            print(f"\033[32ms12 >>\033[0m {msg.content}")
            return

        results = []
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {name}"
            except Exception as e:
                output = f"Error: {e}"

            print(f"> {name}: {str(output)[:200]}")

            # 修改：适配 OpenAI 的 tool 角色响应格式
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(output),
            })

        messages.extend(results)


if __name__ == "__main__":
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")

    history = []
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
