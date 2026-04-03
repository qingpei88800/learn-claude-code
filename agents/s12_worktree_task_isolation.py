#!/usr/bin/env python3
# 【shebang 行】告诉 Unix/Linux 系统用 python3 解释器执行此脚本。
# 类似于 Java 中没有直接对应的概念，但相当于脚本文件的"执行入口声明"。
# Harness: directory isolation -- parallel execution lanes that never collide.

# 【模块级文档字符串(docstring)】Python 中三引号包裹的多行字符串可以用作模块/类/函数的文档说明。
# 类似于 Java 中的 Javadoc 注释（/** ... */），但 docstring 在运行时可以通过 __doc__ 属性访问。
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation

【整体架构说明】
本文件实现了一个基于"目录隔离"的并行任务执行框架。
核心思想：任务(Task)是控制平面，工作树(Worktree)是执行平面。
- 任务管理：用 JSON 文件持久化存储在 .tasks/ 目录下
- 工作树管理：用 git worktree 创建隔离的工作目录，元数据存储在 .worktrees/index.json
- 事件总线：记录所有生命周期事件到 .worktrees/events.jsonl（JSONL 格式，每行一个 JSON）

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.

    # 【目录结构示例】
    .tasks/task_12.json          # 每个任务一个 JSON 文件，类似 Java 中每个实体对应一行数据库记录
      {
        "id": 12,                # 任务 ID（自增主键）
        "subject": "Implement auth refactor",  # 任务标题
        "status": "in_progress",  # 状态：pending / in_progress / completed
        "worktree": "auth-refactor"  # 绑定的工作树名称
      }

    .worktrees/index.json        # 工作树索引文件，类似注册中心/服务发现
      {
        "worktrees": [           # 工作树列表
          {
            "name": "auth-refactor",        # 工作树名称
            "path": ".../.worktrees/auth-refactor",  # 工作树在磁盘上的绝对路径
            "branch": "wt/auth-refactor",   # 关联的 git 分支名
            "task_id": 12,                   # 绑定的任务 ID
            "status": "active"               # 工作树状态
          }
        ]
      }

Key insight: "Isolate by directory, coordinate by task ID."
# 核心设计理念：通过目录隔离实现并行，通过任务 ID 进行协调。
# 这是一种"命令模式 + 工作单元(Unit of Work)"的组合设计。
"""

# ============================================================
# 【导入语句】import 语句相当于 Java 的 import
# Python 的标准库不需要额外安装（类似 JDK 自带的包）
# ============================================================
import json
# json 模块：JSON 序列化/反序列化。相当于 Java 的 Jackson/Gson 库。
# 提供 json.dumps()（对象→JSON字符串）和 json.loads()（JSON字符串→对象）。

import os
# os 模块：操作系统接口。提供环境变量访问(os.getenv/os.environ)、文件路径操作等。
# 相当于 Java 的 System.getenv() 以及 java.io.File 的部分功能。

import re
# re 模块：正则表达式。相当于 Java 的 java.util.regex 包。

import subprocess
# subprocess 模块：用于执行外部命令（类似 Java 的 ProcessBuilder / Runtime.exec()）。

import time
# time 模块：时间相关功能。time.time() 返回当前 Unix 时间戳（秒），类似 Java 的 System.currentTimeMillis() / 1000.0。

from pathlib import Path
# 【from ... import ... 语法】从模块中导入特定的类/函数。
# Path 是 Python 3.4+ 引入的面向对象路径操作类（类似 Java 的 java.nio.file.Path）。
# Path.cwd() 获取当前工作目录，Path.mkdir() 创建目录，Path.read_text() 读取文件内容等。
# 相比传统的 os.path，Path 提供了更优雅的链式调用风格。

# 【第三方库导入】需要通过 pip install 安装
from anthropic import Anthropic
# anthropic：Anthropic 公司的 Claude API 官方 Python SDK。
# Anthropic 类是 API 客户端，类似于 Java 中的 HttpClient 或 Retrofit 的接口实例。

from dotenv import load_dotenv
# python-dotenv：从 .env 文件加载环境变量到 os.environ 中。
# 类似于 Java 中 Spring Boot 的 @PropertySource("classpath:.env") 或者直接在 IDE 中配置环境变量。

# 【调用 load_dotenv】读取项目根目录下的 .env 文件，将其中定义的环境变量加载到进程环境中。
# override=True 表示：如果 .env 中的变量与系统已有环境变量同名，则覆盖它。
# 类似于 Java 中 Properties 文件加载并覆盖 System.setProperty()。
load_dotenv(override=True)

# 【条件判断】如果设置了自定义的 API 基础 URL，则移除认证 token。
# 这是因为使用代理/中转服务时，可能不需要原始的 AUTH_TOKEN。
# os.getenv() 获取环境变量（不存在返回 None），os.environ.pop() 移除环境变量。
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    # pop() 的第二个参数是默认值（key 不存在时不报 KeyError），类似 Java Map.get(key, defaultValue)。

# 【模块级全局常量】Python 中没有 final 关键字，但约定全大写变量名为常量。
# WORKDIR：当前工作目录的 Path 对象。Path.cwd() 类似 Java 的 Paths.get("").toAbsolutePath()。
WORKDIR = Path.cwd()

# 【创建 Anthropic API 客户端实例】
# base_url 参数支持自定义 API 端点（例如使用 API 代理/网关时）。
# 类似于 Java 中 new AnthropicClient.Builder().baseUrl(...).build()。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 【从环境变量读取模型名称】os.environ 是一个类似字典(dict)的对象。
# os.environ["MODEL_ID"] 如果 key 不存在会抛出 KeyError，而 os.getenv("MODEL_ID") 会返回 None。
# 这里用 [] 访问，说明 MODEL_ID 是必需的环境变量。
MODEL = os.environ["MODEL_ID"]


# ============================================================
# 【函数定义】def 关键字定义函数，类似于 Java 的方法定义。
# def detect_repo_root(cwd: Path) -> Path | None:
#   - cwd: Path 是参数声明，类似 Java 的 Path cwd
#   - -> Path | None 是返回类型注解（Type Hint），表示返回 Path 或 None。
#     Python 的类型注解是可选的，不影响运行时行为（类似 Java 的 @Nullable Path）。
#     | 语法是 Python 3.10+ 的联合类型写法，等价于 Union[Path, None] 或 Optional[Path]。
# ============================================================
def detect_repo_root(cwd: Path) -> Path | None:
    """Return git repo root if cwd is inside a repo, else None."""
    # 【try-except 异常处理】类似于 Java 的 try-catch。
    # Python 使用 Exception 作为所有异常的基类（类似 Java 的 Exception）。
    # 这里捕获所有异常是为了保证"探测"操作的健壮性。
    try:
        # 【subprocess.run】执行外部命令，返回 CompletedProcess 对象。
        # 参数说明：
        #   ["git", "rev-parse", "--show-toplevel"] - 命令列表（列表形式避免 shell 注入，类似 Java ProcessBuilder 的 command().args()）
        #   cwd=cwd - 设置工作目录（关键字参数，类似 Java ProcessBuilder.directory()）
        #   capture_output=True - 同时捕获 stdout 和 stderr（相当于 Java 中分别读取两个流）
        #   text=True - 将输出作为字符串而非字节返回（类似 Java 中 new String(process.getInputStream().readAllBytes())）
        #   timeout=10 - 超时时间10秒（类似 Java Process.waitFor(10, TimeUnit.SECONDS)）
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # 【returncode】命令的退出码。0 表示成功，非 0 表示失败。
        # 类似 Java 中 Process.exitValue()。
        if r.returncode != 0:
            return None
        # 【r.stdout.strip()】获取标准输出并去除首尾空白字符。
        # strip() 类似 Java 的 String.trim()。
        root = Path(r.stdout.strip())
        # 【三元表达式】如果 root 存在则返回 root，否则返回 None。
        # Python 的 "x if condition else y" 类似 Java 的 condition ? x : y。
        return root if root.exists() else None
    except Exception:
        # 【except Exception】捕获所有异常（但不会捕获 KeyboardInterrupt/SystemExit，类似 Java 的 catch(Exception)）。
        # 静默返回 None，实现"探测失败则回退"的容错策略。
        return None


# 【模块级变量赋值】调用 detect_repo_root 获取 git 仓库根目录。
# 如果探测失败（返回 None），则使用 WORKDIR 作为回退值。
# 【or 运算符短路求值】Python 中 or 运算符返回第一个"真值"操作数。
# None or WORKDIR → WORKDIR。类似 Java 中 Optional.ofNullable(result).orElse(WORKDIR)。
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

# 【SYSTEM 系统提示词常量】
# 用圆括号 () 包裹的字符串会自动拼接（隐式字符串拼接）。
# f"You are a coding agent at {WORKDIR}" 是 f-string（格式化字符串），用花括号嵌入变量。
# 类似 Java 的 "You are a coding agent at " + WORKDIR 或 String.format("...%s...", WORKDIR)。
# 这个字符串作为 Claude API 的 system 参数，定义了 AI 助手的角色和行为指引。
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)


# ============================================================
# 【EventBus 类 - 事件总线】
# 设计模式：观察者模式的简化版本（仅记录事件，不通知订阅者）。
# 采用"追加写入"(append-only)的 JSONL 格式存储事件日志。
# JSONL(JSON Lines)是一种日志格式：每行一个独立的 JSON 对象，便于追加写入和流式读取。
# 在 Java 生态中类似的设计：SLF4J/Logback 的日志框架，或者 Kafka 的 Append-Only Log。
# ============================================================
class EventBus:
    # 【__init__ 构造方法】Python 中以双下划线开头和结尾的方法称为"魔术方法"(dunder method)。
    # __init__ 类似于 Java 的构造函数。self 类似于 Java 的 this。
    # Python 中方法定义必须显式声明 self 参数（实例引用），Java 中 this 是隐式的。
    def __init__(self, event_log_path: Path):
        # 【实例属性赋值】self.path = event_log_path 将参数保存为实例属性。
        # 类似 Java 中 this.path = eventLogPath。
        self.path = event_log_path
        # 【self.path.parent】获取路径的父目录。类似 Java Path.getParent()。
        # 【mkdir】创建目录。parents=True 表示递归创建所有中间目录（类似 Java Files.createDirectories()）。
        # exist_ok=True 表示如果目录已存在不报错（类似 Java 中先检查 !exists() 再创建）。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 如果日志文件不存在，则创建空文件。Path.write_text("") 会创建并写入空字符串。
        if not self.path.exists():
            self.path.write_text("")

    # 【emit 方法 - 发射事件】
    # 参数类型注解：dict | None = None 表示参数可以是 dict 或 None，默认值为 None。
    # 类似 Java 中 @Nullable Map<String, Object> task。
    def emit(
        self,
        event: str,                   # 事件名称，如 "worktree.create.before"
        task: dict | None = None,     # 关联的任务信息（可选）
        worktree: dict | None = None, # 关联的工作树信息（可选）
        error: str | None = None,     # 错误信息（可选）
    ):
        # 【dict 字典字面量】{} 创建字典，类似 Java 的 new HashMap<>() 并逐个 put。
        # 也可以用 dict(key=value) 形式，但 {} 更常用。
        payload = {
            "event": event,
            "ts": time.time(),      # 当前 Unix 时间戳（秒数，浮点数）
            "task": task or {},      # 【or 短路】如果 task 不是 None 则用 task，否则用空字典 {}
            "worktree": worktree or {},
        }
        # 【if error:】Python 中的"真值"(truthy)判断。
        # None、空字符串""、空列表[]、空字典{}、0 都是"假值"。
        # 非空字符串、非空容器、非零数字都是"真值"。类似 Java 中 if (error != null && !error.isEmpty())。
        if error:
            payload["error"] = error
        # 【with 语句】上下文管理器(context manager)，类似于 Java 的 try-with-resources。
        # 确保文件在代码块结束后自动关闭（即使发生异常）。
        # self.path.open("a", encoding="utf-8") 以"追加模式"(a=append)打开文件。
        # 类似 Java 的 new FileWriter(file, true)（true 表示追加模式）。
        with self.path.open("a", encoding="utf-8") as f:
            # json.dumps(payload) 将字典序列化为 JSON 字符串。
            # + "\n" 在末尾添加换行符，确保每条事件占一行（JSONL 格式）。
            f.write(json.dumps(payload) + "\n")

    # 【list_recent 方法 - 获取最近的事件记录】
    # limit: int = 20 表示参数默认值为 20，类似 Java 方法重载或 @DefaultValue 注解。
    def list_recent(self, limit: int = 20) -> str:
        # 【嵌套函数调用】min(max(1, ...), 200) 确保取值范围在 [1, 200] 之间。
        # int(limit or 20) 如果 limit 是 None/0/False 则使用默认值 20。
        n = max(1, min(int(limit or 20), 200))
        # Path.read_text() 读取整个文件内容为字符串，.splitlines() 按行分割为列表。
        # 类似 Java 中 Files.readAllLines()。
        lines = self.path.read_text(encoding="utf-8").splitlines()
        # 【列表切片】lines[-n:] 取最后 n 个元素。
        # 负索引 -1 表示最后一个，-n 表示倒数第 n 个。
        # 类似 Java 中 list.subList(list.size() - n, list.size())。
        recent = lines[-n:]
        items = []
        # 【for 循环】遍历列表。for line in recent: 类似 Java 的 for (String line : recent)。
        for line in recent:
            try:
                # json.loads(line) 将 JSON 字符串反序列化为 Python 字典(dict)。
                # 类似 Java 的 objectMapper.readValue(line, Map.class)。
                items.append(json.loads(line))
            except Exception:
                # 解析失败的行用特殊对象标记，保证不会因为一条坏数据丢失所有数据。
                items.append({"event": "parse_error", "raw": line})
        # 【返回值】json.dumps(items, indent=2) 将列表序列化为格式化的 JSON 字符串。
        # indent=2 表示缩进 2 个空格，便于人类阅读。类似 Java 的 ObjectMapper.writerWithDefaultPrettyPrinter()。
        return json.dumps(items, indent=2)


# ============================================================
# 【TaskManager 类 - 任务管理器】
# 设计模式：Repository 模式（仓库模式）+ ActiveRecord 的混合。
# 每个任务以独立的 JSON 文件存储在磁盘上，文件名包含任务 ID。
# 这种"每个实体一个文件"的方式类似 Git 的对象存储设计（每个 blob 一个文件）。
# 在 Java 生态中，类似 JPA Repository 但底层不是数据库，而是文件系统。
# ============================================================
class TaskManager:
    # 【构造方法】接收任务存储目录路径。
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        # 【私有属性初始化】通过调用 _max_id() 计算出当前最大 ID，+1 得到下一个可用 ID。
        # Python 中以单下划线 _ 开头的方法表示"内部使用"（类似 Java 的 private/protected 约定）。
        self._next_id = self._max_id() + 1

    # 【私有方法】_max_id 扫描目录中所有 task_*.json 文件，找出最大 ID。
    # 用于初始化自增 ID 计数器。类似数据库的 SELECT MAX(id) FROM tasks。
    def _max_id(self) -> int:
        ids = []
        # 【Path.glob("task_*.json")】按通配符匹配文件名。
        # 类似 Java 中 Files.list(dir).filter(path -> path.getFileName().toString().startsWith("task_"))。
        for f in self.dir.glob("task_*.json"):
            try:
                # 【f.stem】获取文件名（不含扩展名）。例如 "task_5.json" → "task_5"。
                # 【.split("_")[1]】按下划线分割取第二部分，即数字部分。
                # 类似 Java 中 filename.split("_")[1]。
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                # 【pass 语句】空操作，什么都不做。
                # 相当于 Java 中 catch 块里什么都不写（只是 Python 需要显式的 pass 来表示空块）。
                pass
        # 【三元表达式】如果 ids 列表非空则取最大值，否则返回 0。
        # max(ids) 类似 Java 的 Collections.max(ids)。
        return max(ids) if ids else 0

    # 【私有方法】_path 根据任务 ID 构建文件路径。
    # 【f-string】f"task_{task_id}.json" 是格式化字符串，{task_id} 会被替换为变量值。
    # 【/ 运算符重载】Path 重载了 / 运算符用于路径拼接，self.dir / "task_1.json" 等价于 self.dir.resolve("task_1.json")。
    # 类似 Java 的 dir.resolve("task_" + taskId + ".json")。
    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    # 【私有方法】_load 从 JSON 文件加载任务数据，返回字典。
    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            # 【raise 抛出异常】类似 Java 的 throw new ValueError(...)。
            # ValueError 是 Python 内置异常，类似 Java 的 IllegalArgumentException。
            raise ValueError(f"Task {task_id} not found")
        # Path.read_text() 读取全部文本内容。
        return json.loads(path.read_text())

    # 【私有方法】_save 将任务字典写入 JSON 文件。
    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    # 【create 方法 - 创建新任务】
    # subject 是必需参数，description 是可选参数（默认空字符串）。
    # 返回 JSON 字符串，便于 Claude API 将结果返回给用户。
    def create(self, subject: str, description: str = "") -> str:
        # 构造任务字典（类似 Java 中 new Task() 然后逐个 set 属性）。
        task = {
            "id": self._next_id,       # 自增 ID
            "subject": subject,          # 任务标题
            "description": description,  # 任务描述
            "status": "pending",         # 初始状态
            "owner": "",                 # 负责人（空字符串表示未分配）
            "worktree": "",              # 绑定的工作树（空字符串表示未绑定）
            "blockedBy": [],             # 被哪些任务阻塞（预留字段）
            "created_at": time.time(),   # 创建时间戳
            "updated_at": time.time(),   # 更新时间戳
        }
        self._save(task)
        # 【自增操作】Python 中没有 ++ 运算符，需要用 += 1。
        self._next_id += 1
        return json.dumps(task, indent=2)

    # 【get 方法 - 根据 ID 获取任务详情】
    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    # 【exists 方法 - 判断任务是否存在】
    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    # 【update 方法 - 更新任务状态和负责人】
    # status: str = None 参数默认值为 None（Python 习惯用 None 表示"未提供"）。
    # 注意：不要对可变默认参数使用 []、{} 等，它们只在函数定义时创建一次。
    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        task = self._load(task_id)
        # 【if status:】如果 status 不为 None 且不为空字符串
        if status:
            # 【in 操作符】检查值是否在元组中。("pending", "in_progress", "completed") 是元组(tuple)，类似 Java 的 List.of()。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        # 【if owner is not None:】注意这里用 is not None 而不是 if owner:。
        # 因为空字符串 "" 也是有效的 owner 值（表示清除负责人），而 if owner: 会把空字符串当作"假值"跳过。
        # 这是 Python 中常见的陷阱：区分"未提供参数(None)"和"提供了空值("")"。
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    # 【bind_worktree 方法 - 将任务绑定到工作树】
    # 绑定后如果任务还是 pending 状态，自动转为 in_progress。
    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        # 如果任务还是 pending，自动开始执行（状态转换）
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    # 【unbind_worktree 方法 - 解除任务与工作树的绑定】
    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    # 【list_all 方法 - 列出所有任务】
    # 返回格式化的文本列表，而非 JSON。方便用户直接阅读。
    def list_all(self) -> str:
        tasks = []
        # 【sorted()】对文件列表排序，确保按文件名顺序输出（类似 Java 的 stream().sorted()）。
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            # 【字典 .get() 方法】dict.get(key, default) 获取值，key 不存在时返回默认值。
            # 类似 Java 中 map.getOrDefault(key, default)。
            # 这里根据状态映射到不同的标记符号。
            marker = {
                "pending": "[ ]",      # 未开始
                "in_progress": "[>]",  # 进行中
                "completed": "[x]",    # 已完成
            }.get(t["status"], "[?]")  # 未知状态用 [?]
            # 【条件 f-string】f" owner={t['owner']}" if t.get("owner") else ""
            # 如果有 owner 则拼接 owner 信息，否则为空字符串。
            # 类似 Java 中 (owner != null ? " owner=" + owner : "")。
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
        # 【"\n".join(lines)】将字符串列表用换行符连接为一个字符串。
        # 这是 Python 中常见的列表→字符串转换方式。类似 Java 中 String.join("\n", lines)。
        return "\n".join(lines)


# 【创建全局单例实例】在模块加载时创建，整个程序中共享使用。
# 类似 Java 中 Spring 的 @Bean 单例注入，或者直接定义 public static final 变量。
# REPO_ROOT / ".tasks" 利用 Path 的 / 运算符拼接路径。
TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


# ============================================================
# 【WorktreeManager 类 - 工作树管理器】
# 设计模式：Facade 模式（门面模式）——封装了对 git worktree、TaskManager、EventBus 的操作。
# 职责：
#   1. 创建/删除 git worktree（通过子进程调用 git 命令）
#   2. 维护工作树索引文件（.worktrees/index.json）
#   3. 协调任务绑定与事件记录
# 在 Java 生态中，类似一个 Service 层类，协调多个 Repository/组件完成业务逻辑。
# ============================================================
class WorktreeManager:
    # 【构造方法】通过依赖注入接收外部依赖。
    # repo_root: Git 仓库根目录路径
    # tasks: TaskManager 实例（用于任务绑定）
    # events: EventBus 实例（用于记录事件）
    # 类似 Java 中 @Autowired 注入依赖。
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        # 工作树索引文件路径，存储所有工作树的元数据。
        self.index_path = self.dir / "index.json"
        # 如果索引文件不存在，初始化一个空的工作树列表。
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        # 【启动时检查】检测当前是否在 git 仓库中。
        self.git_available = self._is_git_repo()

    # 【私有方法】检查当前目录是否在 git 仓库中。
    def _is_git_repo(self) -> bool:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],  # git 命令：检查是否在 work tree 中
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            # returncode == 0 表示命令成功执行（即在 git 仓库中）。
            return r.returncode == 0
        except Exception:
            return False

    # 【私有方法】执行 git 命令并返回输出。封装了错误处理逻辑。
    # list[str] 是 Python 3.9+ 的列表类型注解语法，等价于 List[str]（需要 from typing import List）。
    # 类似 Java 中 List<String> args。
    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            # 【RuntimeError】运行时异常，类似 Java 的 IllegalStateException。
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],  # 【*args 解包】* 运算符将列表元素解包为独立参数。
            # ["git", *["worktree", "add"]] 等价于 ["git", "worktree", "add"]。
            # 类似 Java 中没有直接对应的概念，但效果等同于把列表元素逐个添加。
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,  # 120 秒超时
        )
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).strip()
            # 【f-string 中的表达式】f"git {' '.join(args)} failed"
            # {' '.join(args)} 花括号内的表达式先用空格连接参数列表，再嵌入字符串。
            # 类似 Java 的 "git " + String.join(" ", args) + " failed"。
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        # 【or 短路】如果输出为空字符串(假值)，则返回 "(no output)"。
        return (r.stdout + r.stderr).strip() or "(no output)"

    # 【私有方法】加载索引文件。
    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    # 【私有方法】保存索引文件。
    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))

    # 【私有方法】按名称查找工作树条目。返回字典或 None。
    def _find(self, name: str) -> dict | None:
        idx = self._load_index()
        # 遍历索引中的工作树列表。
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    # 【私有方法】验证工作树名称合法性。
    # re.fullmatch() 要求整个字符串匹配正则表达式（类似 Java 中 Pattern.matches()）。
    # r"[A-Za-z0-9._-]{1,40}" 原始字符串(raw string)，r 前缀取消反斜杠转义。
    # 正则含义：1-40 个字符，允许字母、数字、点号、下划线、连字符。
    def _validate_name(self, name: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    # 【create 方法 - 创建工作树】
    # task_id: int = None 表示可选参数，默认 None。
    # base_ref: str = "HEAD" 表示基于哪个 git 引用创建（默认当前 HEAD）。
    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        # 检查是否已存在同名工作树（唯一性约束，类似数据库 UNIQUE 约束）。
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        # 如果指定了 task_id，验证任务是否存在。
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name  # 工作树的物理路径
        branch = f"wt/{name}"   # 工作树的分支名，wt/ 前缀便于识别
        # 发射"创建前"事件（类似日志的 INFO 级别，但结构化存储）。
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        try:
            # 【执行 git worktree add 命令】
            # git worktree add -b <新分支名> <路径> <基于哪个引用>
            # str(path) 将 Path 对象转换为字符串（git 命令行需要字符串参数）。
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            # 构建工作树条目字典（元数据记录）。
            entry = {
                "name": name,
                "path": str(path),       # Path → str，因为 JSON 不支持 Path 类型
                "branch": branch,
                "task_id": task_id,       # None 在 JSON 中会变成 null
                "status": "active",
                "created_at": time.time(),
            }

            # 将新条目追加到索引列表中。
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            # 如果指定了任务，自动绑定工作树和任务（双向关联）。
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            # 发射"创建后"成功事件。
            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            return json.dumps(entry, indent=2)
        except Exception as e:
            # 【异常处理】发射失败事件，然后重新抛出异常（raise 不带参数 = 重新抛出当前异常）。
            # 类似 Java 中 catch(Exception e) { log.error(...); throw e; }。
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),  # str(e) 获取异常的字符串表示，类似 Java 的 e.getMessage()
            )
            raise  # 【裸 raise】重新抛出当前异常，保留原始堆栈信息。类似 Java 的 throw e。

    # 【list_all 方法 - 列出所有工作树】
    def list_all(self) -> str:
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            # 可选拼接任务 ID 信息。
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            # f-string 中可以跨行拼接（只要在括号内）。
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    # 【status 方法 - 查看工作树的 git 状态】
    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        # 在工作树目录中执行 git status 命令。
        # --short 显示简洁格式，--branch 显示分支信息。
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,  # 在工作树目录（而非仓库根目录）中执行命令
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        # 【or 短路】如果输出为空，说明工作树是干净的（无修改）。
        return text or "Clean worktree"

    # 【run 方法 - 在工作树中执行命令】
    def run(self, name: str, command: str) -> str:
        # 【安全黑名单】防止执行危险的系统命令。
        # 类似 Java 中的安全过滤器模式。
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        # 【any() 内置函数】如果任何一个元素为真则返回 True。
        # 生成器表达式 any(d in command for d in dangerous) 类似 Java 的 stream().anyMatch()。
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            # 【shell=True】通过 shell 执行命令（允许使用管道、重定向等 shell 特性）。
            # 注意：使用 shell=True 时 command 是单个字符串（而非列表），且需注意注入风险。
            # 上面已经做了黑名单检查来降低风险。
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,  # 关键：在工作树的隔离目录中执行命令
                capture_output=True,
                text=True,
                timeout=300,  # 5 分钟超时
            )
            out = (r.stdout + r.stderr).strip()
            # 【字符串切片】out[:50000] 截取前 50000 个字符，防止输出过大。
            # 类似 Java 的 out.substring(0, Math.min(50000, out.length()))。
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            # 捕获超时异常（类似 Java 的 TimeoutException）。
            return "Error: Timeout (300s)"

    # 【remove 方法 - 移除工作树】
    # force: 是否强制移除（即使有未提交的修改）
    # complete_task: 是否同时将关联的任务标记为已完成
    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            # 动态构建 git 命令参数列表。
            args = ["worktree", "remove"]
            if force:
                args.append("--force")  # 强制移除（丢弃未提交的修改）
            args.append(wt["path"])
            self._run_git(args)

            # 如果设置了 complete_task 且关联了任务，则完成任务。
            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                # json.loads 反序列化获取任务详情（在状态变更前保存）。
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                # 发射任务完成事件。
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            # 更新索引：将工作树状态标记为 "removed"（软删除，保留历史记录）。
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    # 【keep 方法 - 保留工作树】
    # 不删除工作树，只是将其生命周期状态标记为 "kept"。
    # 这表示用户有意保留这个工作树（区别于 active 的正在使用状态）。
    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        # 查找并更新指定名称的工作树状态。
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )
        # 【三元表达式】如果找到了工作树返回 JSON，否则返回错误信息。
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"


# 【创建 WorktreeManager 全局单例实例】
# 注入 REPO_ROOT、TASKS、EVENTS 三个依赖。
WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


# ============================================================
# 【基础工具函数】提供文件读写和命令执行的原子操作。
# 这些函数作为 Claude AI 的"工具"(Tool)被调用。
# 类似 Java 中的 Service/Util 工具类方法。
# ============================================================

# 【safe_path - 路径安全验证】
# 防止路径遍历攻击（Path Traversal），确保操作的文件不超出工作目录。
# 类似 Java 中的安全校验工具方法。
def safe_path(p: str) -> Path:
    # 【Path.resolve()】解析为绝对路径并消除 .. 和符号链接。
    # 类似 Java 的 Path.toAbsolutePath().normalize()。
    path = (WORKDIR / p).resolve()
    # 【is_relative_to()】Python 3.9+ 方法，检查 path 是否在 WORKDIR 下。
    # 如果 path 是 /etc/passwd 而 WORKDIR 是 /home/project，则返回 False（拒绝访问）。
    # 类似 Java 中 !path.startsWith(WORKDIR) 的检查。
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# 【run_bash - 在工作目录执行 shell 命令】
def run_bash(command: str) -> str:
    # 【危险命令黑名单】简单但有效的安全防护。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,  # 通过 shell 执行，支持管道和重定向
            cwd=WORKDIR,  # 在主工作目录中执行（非工作树目录）
            capture_output=True,
            text=True,
            timeout=120,  # 2 分钟超时
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# 【run_read - 读取文件内容】
# limit: int = None 表示可选参数，None 表示不限制行数。
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        # 【切片 + 列表拼接】如果指定了 limit 且文件行数超过 limit，
        # 则截取前 limit 行，并添加省略提示行。
        # 【列表拼接】lines[:limit] + [...] 用 + 号拼接两个列表。类似 Java 的 list1.addAll(list2)。
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


# 【run_write - 写入文件】
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        # 【自动创建父目录】如果文件的父目录不存在则先创建。
        # 类似 Java 中 Files.createDirectories(path.getParent())。
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        # 【f-string】len(content) 返回字符串长度（字节数取决于编码）。
        # Python 的 len() 对字符串返回字符数（类似 Java 的 String.length()）。
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


# 【run_edit - 编辑文件（查找并替换文本）】
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        # 【in 操作符】检查子字符串是否存在。类似 Java 的 string.contains(oldText)。
        if old_text not in c:
            return f"Error: Text not found in {path}"
        # 【str.replace(old, new, count)】替换字符串。count=1 表示只替换第一个匹配项。
        # 类似 Java 中 replaceFirst()。注意：Python 的 replace() 不会使用正则表达式。
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# 【TOOL_HANDLERS - 工具注册表】
# 设计模式：命令模式(Command Pattern) + 策略模式(Strategy Pattern)。
# 用字典(dict)将工具名称映射到对应的处理函数（Lambda 表达式）。
# Claude API 返回工具调用时，通过名称查找对应的处理函数并执行。
# 类似 Java 中 Map<String, Function<Map, String>> handlers 的注册表模式。
# ============================================================
TOOL_HANDLERS = {
    # 【Lambda 表达式】lambda **kw: ... 是匿名函数。
    # **kw 表示接收任意数量的关键字参数，打包为字典 kw。
    # 类似 Java 中 (Map<String, Object> kw) -> { ... }。
    # kw["command"] 获取必需参数（不存在会抛出 KeyError）。
    # kw.get("limit") 获取可选参数（不存在返回 None）。
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 任务管理工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    # 工作树管理工具
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    # 事件查询工具
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

# ============================================================
# 【TOOLS 列表 - 工具定义（Schema）】
# 这是传给 Claude API 的工具定义列表，告诉 AI 有哪些工具可用。
# 每个工具定义包含：
#   - name: 工具名称（与 TOOL_HANDLERS 中的 key 对应）
#   - description: 工具描述（AI 通过这个描述理解何时该使用哪个工具）
#   - input_schema: JSON Schema 格式的参数定义（类似 OpenAPI/Swagger 的参数定义）
#
# input_schema 遵循 JSON Schema 规范：
#   - type: "object" 表示参数是一个对象（类似 Java 中的 DTO/POJO）
#   - properties: 定义每个参数的类型和含义
#   - required: 必需参数列表（未提供会报错）
#   - enum: 枚举值约束（类似 Java 的 enum）
#
# 类似 Java 中使用注解定义 REST API 参数（如 @RequestParam, @RequestBody）。
# ============================================================
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace (blocking).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    # --- 任务管理工具定义 ---
    {
        "name": "task_create",
        "description": "Create a new task on the shared task board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "task_list",
        "description": "List all tasks with status, owner, and worktree binding.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "task_get",
        "description": "Get task details by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "task_update",
        "description": "Update task status or owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],  # 枚举约束
                },
                "owner": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "worktree": {"type": "string"},
                "owner": {"type": "string"},
            },
            "required": ["task_id", "worktree"],
        },
    },
    # --- 工作树管理工具定义 ---
    {
        "name": "worktree_create",
        "description": "Create a git worktree and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "integer"},
                "base_ref": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_list",
        "description": "List worktrees tracked in .worktrees/index.json.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "worktree_status",
        "description": "Show git status for one worktree.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_run",
        "description": "Run a shell command in a named worktree directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "worktree_remove",
        "description": "Remove a worktree and optionally mark its bound task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "force": {"type": "boolean"},
                "complete_task": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept in lifecycle state without removing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_events",
        "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
]


# ============================================================
# 【agent_loop 函数 - Agent 主循环】
# 这是整个程序的核心：一个"工具调用循环"(Tool-Use Loop)。
# 工作流程：
#   1. 将消息历史发送给 Claude API
#   2. 如果 Claude 返回工具调用 → 执行工具 → 将结果反馈给 Claude → 回到步骤 1
#   3. 如果 Claude 返回纯文本（不再需要工具）→ 退出循环
#
# 这是一种经典的 ReAct(Reasoning + Acting) 模式实现。
# 类似 Java 中的 while 循环 + 状态机。
# ============================================================
def agent_loop(messages: list):
    # 【while True:】无限循环，通过 return 或 break 退出。
    # 类似 Java 的 while (true) { ... }。
    while True:
        # 【调用 Claude API】client.messages.create() 发送消息并获取响应。
        # 参数说明：
        #   model=MODEL - 使用的模型名称（如 "claude-sonnet-4-20250514"）
        #   system=SYSTEM - 系统提示词（定义 AI 的角色和行为）
        #   messages=messages - 对话历史（列表格式）
        #   tools=TOOLS - 可用的工具定义列表
        #   max_tokens=8000 - 最大输出 token 数
        # 类似 Java 中 httpClient.send(HttpRequest.of(...))。
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # 【将 AI 的回复追加到消息历史】
        # messages 是一个列表(list)，类似 Java 的 List<Map<String, Object>>。
        # response.content 是一个包含文本块和工具调用块的内容列表。
        messages.append({"role": "assistant", "content": response.content})

        # 【stop_reason 检查】如果 AI 不需要调用工具（返回纯文本或结束），则退出循环。
        # stop_reason 的可能值：
        #   "tool_use" - AI 想要调用工具（继续循环）
        #   "end_turn" - AI 完成回复（退出循环）
        #   "max_tokens" - 达到最大 token 限制（退出循环）
        if response.stop_reason != "tool_use":
            return  # 【return 无返回值】相当于 Java 中 return;（从 void 方法返回）。

        # --- 处理工具调用 ---
        results = []
        # 【遍历响应内容块】response.content 是一个列表，每个元素是一个内容块。
        # 可能的类型：TextBlock（文本）、ToolUseBlock（工具调用）。
        for block in response.content:
            # 【类型检查】block.type == "tool_use" 判断是否为工具调用块。
            # response.content 中的 block 是对象，通过 .type 属性访问类型。
            if block.type == "tool_use":
                # 【从注册表查找处理函数】dict.get(key) 方法查找，不存在返回 None。
                # 类似 Java 中 handlerMap.get(block.getName())。
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # 【**block.input 解包】将字典解包为关键字参数传入函数。
                    # handler(**{"command": "ls"}) 等价于 handler(command="ls")。
                    # 类似 Java 中通过反射调用方法并传递参数。
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                # 【打印工具执行结果】f-string 格式化输出。
                print(f"> {block.name}:")
                # 【str(output)[:200]】将输出转为字符串并截取前 200 个字符用于打印。
                print(str(output)[:200])
                # 【构建工具结果消息】按照 Claude API 的格式要求。
                results.append(
                    {
                        "type": "tool_result",         # 消息类型：工具结果
                        "tool_use_id": block.id,       # 关联的工具调用 ID（用于匹配请求和响应）
                        "content": str(output),        # 工具执行结果（转为字符串）
                    }
                )
        # 【将工具结果追加到消息历史】作为"user"角色发送。
        # 在 Claude API 中，工具结果以"user"角色的消息形式返回。
        messages.append({"role": "user", "content": results})


# ============================================================
# 【程序入口 - main 块】
# 【if __name__ == "__main__":】这是 Python 的惯用写法。
# __name__ 是模块的特殊属性：当文件被直接运行时为 "__main__"，被 import 时为模块名。
# 这个判断确保以下代码只在直接运行脚本时执行，被其他模块 import 时不会执行。
# 类似 Java 的 public static void main(String[] args) 方法入口。
# ============================================================
if __name__ == "__main__":
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")

    # 【history 对话历史】维护整个对话的消息列表。
    # 每轮对话的 user 消息和 assistant 回复都追加到此列表中。
    # 类似 Java 中 List<Map<String, Object>> history = new ArrayList<>()。
    history = []
    while True:
        try:
            # 【input() 内置函数】从标准输入读取一行用户输入。
            # 类似 Java 的 Scanner.nextLine() 或 BufferedReader.readLine()。
            # 【ANSI 转义码】\033[36m 设置文本颜色为青色(cyan)，\033[0m 重置颜色。
            # \033 是 ESC 字符的八进制表示。这些控制码让终端显示彩色提示符。
            query = input("\033[36ms12 >> \033[0m")
        # 【捕获多个异常类型】(EOFError, KeyboardInterrupt) 是元组，表示同时捕获这两种异常。
        # EOFError: 输入流结束（类似 Java 中读到 null）
        # KeyboardInterrupt: 用户按下 Ctrl+C 中断（类似 Java 中未被处理的 InterruptException）
        except (EOFError, KeyboardInterrupt):
            break
        # 【strip() 去除首尾空白，lower() 转小写】
        # 【in 操作符】检查值是否在元组中。
        # 如果用户输入 q、exit 或空行，则退出 REPL 循环。
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 将用户输入追加到对话历史。
        history.append({"role": "user", "content": query})
        # 【调用 agent_loop】AI 在循环中可能执行多次工具调用，直到给出最终文本回复。
        agent_loop(history)
        # 【获取 AI 的回复内容】history[-1] 是最后一个元素（即 AI 的最新回复）。
        # 负索引 -1 表示列表最后一个元素（类似 Java 中 list.get(list.size() - 1)）。
        response_content = history[-1]["content"]
        # 【isinstance() 类型检查】检查对象是否是某个类型的实例。
        # 类似 Java 中的 instanceof 操作符：if (responseContent instanceof List)。
        # response_content 可能是 list（包含文本块和工具调用块）或其他类型。
        if isinstance(response_content, list):
            for block in response_content:
                # 【hasattr() 属性检查】检查对象是否具有指定属性。
                # 类似 Java 中通过反射检查字段是否存在，或简单地检查类型。
                if hasattr(block, "text"):
                    # 只打印文本块（忽略工具调用块，那些已经在 agent_loop 中处理过了）。
                    print(block.text)
        print()  # 打印空行，分隔对话轮次。
