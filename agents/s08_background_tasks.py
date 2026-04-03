#!/usr/bin/env python3
# 【Shebang 行】告诉操作系统用 python3 解释器来执行此脚本。
# 类似 Java 中虽然没有 shebang，但 Java 程序依赖 JVM 来运行，而 Python 脚本通过 shebang 指定解释器。

# Harness: background execution -- the model thinks while the harness waits.
# 【框架说明】后台执行 —— 模型思考的同时，框架在等待后台任务完成。

"""
s08_background_tasks.py - Background Tasks（后台任务系统）

Run commands in background threads. A notification queue is drained
before each LLM call to deliver results.
【核心功能】在线程中后台运行命令。通知队列在每次 LLM 调用之前被排空，以将结果投递给模型。

    Main thread                Background thread
    【主线程】                   【后台线程】
    +-----------------+        +-----------------+
    | agent loop      |        | task executes   |
    | ...             |        | ...             |
    | [LLM call] <---+------- | enqueue(result) |
    |  ^drain queue   |        +-----------------+
    +-----------------+
    # 上图展示了主线程和后台线程之间的交互方式：
    # 主线程在每次调用 LLM 之前，从通知队列中取出所有已完成的后台任务结果，注入到消息中。

    Timeline:
    Agent ----[spawn A]----[spawn B]----[other work]----
                 |              |
                 v              v
              [A runs]      [B runs]        (parallel)
                 |              |
                 +-- notification queue --> [results injected]
    # 上图展示了时间线：Agent 启动任务 A 和 B 后可以继续其他工作，
    # A 和 B 并行执行，完成后通过通知队列将结果注入到 Agent 的对话中。

Key insight: "Fire and forget -- the agent doesn't block while the command runs."
【核心洞察】"点火即忘" —— Agent 在命令运行时不会阻塞，可以继续处理其他事情。
这类似于 Java 中的 CompletableFuture.runAsync() 或 ExecutorService.submit()。
"""

# ============================================================
# 导入模块
# ============================================================
import os
# 【os 模块】提供操作系统相关功能，如环境变量、文件路径等。
# 类似 Java 的 System.getenv()、System.getProperty() 等。

import subprocess
# 【subprocess 模块】用于创建新进程、执行外部命令。
# 类似 Java 的 ProcessBuilder 或 Runtime.exec()。

import threading
# 【threading 模块】提供线程相关的功能，如 Thread、Lock 等。
# 类似 Java 的 java.lang.Thread 和 java.util.concurrent.locks.Lock。

import uuid
# 【uuid 模块】用于生成唯一标识符（UUID）。
# 类似 Java 的 java.util.UUID.randomUUID()。

from pathlib import Path
# 【Path 类】面向对象的文件路径操作，比字符串拼接更安全。
# 类似 Java 的 java.nio.file.Path。
# 注意：Python 3.10+ 才有 is_relative_to() 方法。

from anthropic import Anthropic
# 【Anthropic SDK】调用 Claude API 的客户端库。
# 类似 Java 中使用 OkHttpClient 或 RestTemplate 调用 REST API。

from dotenv import load_dotenv
# 【dotenv】从 .env 文件加载环境变量，方便本地开发。
# 类似 Spring Boot 的 application.properties 或 application.yml 加载配置。

load_dotenv(override=True)
# 【加载 .env 文件】override=True 表示覆盖已有环境变量。
# 类似 Java 中 System.setProperty() 覆盖系统属性。

# ============================================================
# 环境变量处理
# ============================================================
if os.getenv("ANTHROPIC_BASE_URL"):
    # 【条件判断】如果设置了自定义 API 地址，则移除认证 token，避免冲突。
    # os.getenv() 类似 Java 的 System.getenv()，获取环境变量，不存在则返回 None（不抛异常）。
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    # 【pop 方法】从字典中移除指定 key，第二个参数是默认值（key 不存在时不会报错）。
    # os.environ 是一个字典，类似 Java 的 System.getenv() 但可修改。
    # 注意：Python 字典的 pop(key, default) 类似 Java Map 的 remove()，但可以提供默认值避免 NPE。

# ============================================================
# 全局变量
# ============================================================
WORKDIR = Path.cwd()
# 【当前工作目录】Path.cwd() 获取当前工作目录，返回 Path 对象。
# 类似 Java 的 Path.of(".").toAbsolutePath()。

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 【Anthropic 客户端】创建 API 客户端，如果设置了自定义 URL 则使用自定义 URL。
# 类似 Java 中 new OkHttpClient.Builder().baseUrl(url).build()。

MODEL = os.environ["MODEL_ID"]
# 【模型 ID】从环境变量获取模型名称。使用 [] 访问，key 不存在会抛 KeyError。
# 类似 Java 中 System.getenv("MODEL_ID")，但 Java 返回 null，Python 直接抛异常。

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."
# 【f-string】Python 3.6+ 的格式化字符串，用花括号 {} 嵌入变量。
# 类似 Java 的 "You are a coding agent at " + WORKDIR 或 String.format("...%s...", WORKDIR)。
# 注意：Java 也有 text block + String.formatted()（Java 15+），但 Python f-string 更简洁。


# ============================================================
# BackgroundManager 类：后台任务管理器
# ============================================================
# -- BackgroundManager: threaded execution + notification queue --
# 【设计模式】生产者-消费者模式：
#   - 后台线程是"生产者"，执行完命令后将结果放入通知队列
#   - 主线程（agent_loop）是"消费者"，在每次 LLM 调用前排空队列
#   - 通过 threading.Lock 保证线程安全（类似 Java 的 ReentrantLock）

class BackgroundManager:
    # 【类定义】Python 中用 class 关键字定义类，不需要像 Java 那样有访问修饰符。
    # 所有类默认继承 object（Python 3 中可以不显式写出）。

    def __init__(self):
        # 【构造方法】__init__ 是 Python 的构造方法，__ 前后双下划线是特殊方法（dunder method）。
        # 类似 Java 的 public BackgroundManager() 构造函数。
        # 注意：Python 中所有方法第一个参数必须是 self，相当于 Java 的 this（但 Java 是隐式的，Python 是显式的）。

        self.tasks = {}  # task_id -> {status, result, command}
        # 【实例属性】self.xxx 定义实例属性，类似 Java 的 this.xxx = xxx。
        # 这里用字典存储所有任务，key 是任务 ID，value 是包含 status/result/command 的字典。
        # 类似 Java 中的 HashMap<String, TaskInfo> tasks = new HashMap<>()。

        self._notification_queue = []  # completed task results
        # 【通知队列】用列表模拟队列，存储已完成任务的通知。
        # _ 前缀是 Python 约定，表示"内部使用，外部不应该直接访问"（类似 Java 的 private）。
        # 但 Python 没有真正的访问控制，这只是一个约定。
        # 类似 Java 的 private List<Notification> notificationQueue = new ArrayList<>()。

        self._lock = threading.Lock()
        # 【线程锁】用于保护 _notification_queue 的线程安全。
        # threading.Lock() 类似 Java 的 ReentrantLock()。
        # Python 没有 synchronized 关键字，必须显式使用锁。

    def run(self, command: str) -> str:
        # 【方法签名】command: str 表示参数类型提示（Type Hint），-> str 表示返回类型。
        # Python 的类型提示是可选的，不会影响运行时行为（不会强制类型检查）。
        # 类似 Java 的 public String run(String command)。

        """Start a background thread, return task_id immediately."""
        # 【文档字符串】三引号字符串作为方法文档，类似 Java 的 Javadoc /** ... */。

        task_id = str(uuid.uuid4())[:8]
        # 【生成任务 ID】uuid.uuid4() 生成随机 UUID，str() 转为字符串，[:8] 截取前 8 位。
        # 类似 Java 的 UUID.randomUUID().toString().substring(0, 8)。

        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        # 【字典赋值】将任务信息存入字典。Python 字典用 {} 定义，类似 Java 的 HashMap。
        # 注意：None 是 Python 的空值，类似 Java 的 null。

        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        # 【创建线程】
        # target: 线程要执行的方法（类似 Java 的 Runnable）
        # args: 传给方法的参数元组（元组用 () 定义，类似 Java 的数组）
        # daemon=True: 设为守护线程，主线程结束时守护线程自动退出
        # 类似 Java 中 new Thread(() -> execute(taskId, command))，然后 thread.setDaemon(true)。

        thread.start()
        # 【启动线程】开始执行后台任务。类似 Java 的 thread.start()。
        # 启动后 run() 方法会立即返回，不会阻塞等待线程完成。

        return f"Background task {task_id} started: {command[:80]}"
        # 【返回结果】立即返回任务 ID，command[:80] 截取前 80 个字符防止返回内容过长。
        # 切片 [start:end] 是 Python 特色语法，类似 Java 的 String.substring()。

    def _execute(self, task_id: str, command: str):
        # 【私有方法】_ 前缀表示内部方法，类似 Java 的 private。
        # 这个方法在后台线程中执行，运行 shell 命令并捕获输出。

        """Thread target: run subprocess, capture output, push to queue."""

        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            # 【执行子进程】
            # shell=True: 通过 shell 执行命令（类似 Java 的 /bin/sh -c command）
            # cwd: 工作目录（类似 Java ProcessBuilder.directory()）
            # capture_output=True: 捕获 stdout 和 stderr（类似 Java 的 redirectErrorStream）
            # text=True: 以文本模式返回输出（类似 Java 中用 BufferedReader 读取）
            # timeout=300: 超时时间 300 秒（类似 Java 的 Process.waitFor(300, TimeUnit.SECONDS)）

            output = (r.stdout + r.stderr).strip()[:50000]
            # 【合并输出】将标准输出和标准错误合并，strip() 去除首尾空白。
            # [:50000] 截取前 50000 字符，防止输出过大。

            status = "completed"
            # 设置任务状态为已完成。

        except subprocess.TimeoutExpired:
            # 【捕获超时异常】命令执行超过 300 秒时触发。
            output = "Error: Timeout (300s)"
            status = "timeout"

        except Exception as e:
            # 【捕获所有异常】Python 用 except 关键字，类似 Java 的 catch。
            # Exception as e 类似 Java 的 catch (Exception e)。
            # Python 中捕获基类 Exception 可以捕获所有非系统退出异常（类似 Java）。
            output = f"Error: {e}"
            status = "error"

        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        # 【更新任务状态】将执行结果写入任务字典。
        # Python 的 `or` 运算符：如果 output 为空字符串（falsy），则使用 "(no output)"。
        # 类似 Java 中的 output.isEmpty() ? "(no output)" : output。

        with self._lock:
            # 【上下文管理器】with 语句确保锁的获取和释放。
            # 进入 with 块时自动获取锁（lock.acquire()），退出时自动释放（lock.release()）。
            # 类似 Java 的 try-finally 或 try-with-resources 模式：
            #   lock.lock();
            #   try { ... } finally { lock.unlock(); }

            self._notification_queue.append({
                # 【字典追加】向通知队列添加一条通知。
                # 字典用 {} 定义，键值对用 key: value 语法。
                # 类似 Java 中创建 Map 并 put：Map<String, Object> map = new HashMap<>(); map.put("task_id", taskId);

                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
                # 结果截取前 500 字符，因为通知只是给 LLM 看的摘要，不需要完整输出。
            })

    def check(self, task_id: str = None) -> str:
        # 【默认参数】task_id: str = None 表示参数默认值为 None。
        # 类似 Java 中的方法重载（一个带参数，一个不带参数）。
        # 但 Python 用默认参数实现，比 Java 更灵活。

        """Check status of one task or list all."""

        if task_id:
            # 【条件判断】Python 中 if 判断，None、空字符串、0 等都是 falsy 值。
            # 类似 Java 中 if (taskId != null)。

            t = self.tasks.get(task_id)
            # 【字典 get 方法】get(key) 返回值，key 不存在返回 None（不抛异常）。
            # 类似 Java Map 的 get()，但 Java 返回 null 时需要检查 NPE。

            if not t:
                # 【not 运算符】not t 等价于 t is None 或 not bool(t)。
                # 类似 Java 的 t == null。

                return f"Error: Unknown task {task_id}"

            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
            # 【字典访问】t['status'] 用方括号访问字典值。
            # 类似 Java 的 t.get("status")。
            # t.get('result') or '(running)' —— 如果任务还在运行中，result 为 None，显示 '(running)'。

        # 以下是没有指定 task_id 时，列出所有任务的逻辑：
        lines = []
        # 创建空列表，类似 Java 的 List<String> lines = new ArrayList<>()。

        for tid, t in self.tasks.items():
            # 【字典遍历】.items() 返回键值对的迭代器。
            # 类似 Java 的 for (Map.Entry<String, TaskInfo> entry : tasks.entrySet())。

            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
            # 【列表追加】append() 向列表末尾添加元素。
            # 类似 Java 的 lines.add()。

        return "\n".join(lines) if lines else "No background tasks."
        # 【三元表达式】A if C else B —— 条件为真返回 A，否则返回 B。
        # 类似 Java 的 C ? A : B。
        # "\n".join(lines) 用换行符连接列表元素，类似 Java 的 String.join("\n", lines)。

    def drain_notifications(self) -> list:
        # 【排空通知队列】返回并清空所有待处理的通知。
        # 这是消费者操作，由主线程调用。

        """Return and clear all pending completion notifications."""

        with self._lock:
            # 【加锁】确保线程安全地读取和清空队列。

            notifs = list(self._notification_queue)
            # 【复制列表】list() 创建列表的浅拷贝，避免清空操作影响返回值。
            # 类似 Java 的 new ArrayList<>(notificationQueue)。

            self._notification_queue.clear()
            # 【清空列表】清除所有元素。类似 Java 的 list.clear()。

        return notifs


# ============================================================
# 全局 BackgroundManager 实例
# ============================================================
BG = BackgroundManager()
# 【全局单例】创建一个全局的后台任务管理器实例。
# Python 模块级别的变量相当于 Java 的 static final 变量。
# 这里虽然没有用单例模式，但模块只会被导入一次，所以 BG 实际上是单例的。


# ============================================================
# 工具实现函数（Agent 可以调用的操作）
# ============================================================
# -- Tool implementations --

def safe_path(p: str) -> Path:
    # 【安全路径工具】防止路径遍历攻击（Path Traversal）。
    # 类似 Java 中的安全文件操作，确保文件不会逃逸出工作目录。

    path = (WORKDIR / p).resolve()
    # 【路径拼接】WORKDIR / p 使用 Path 的 / 运算符拼接路径（比 os.path.join 更简洁）。
    # .resolve() 将路径转为绝对路径，解析所有 .. 和 . 符号。
    # 类似 Java 的 WORKDIR.resolve(p).normalize().toAbsolutePath()。

    if not path.is_relative_to(WORKDIR):
        # 【路径安全检查】确保路径在工作目录内，防止 "../../../etc/passwd" 这类攻击。
        # is_relative_to() 是 Python 3.9+ 的方法。
        # 类似 Java 中检查 path.startsWith(WORKDIR)。

        raise ValueError(f"Path escapes workspace: {p}")
        # 【抛出异常】raise 关键字抛出异常，类似 Java 的 throw new Exception()。
        # ValueError 类似 Java 的 IllegalArgumentException。

    return path

def run_bash(command: str) -> str:
    # 【执行 Bash 命令】同步执行 shell 命令（会阻塞直到命令完成）。
    # 与 background_run 不同，这个是同步的，类似于 Java 的 ProcessBuilder.start().waitFor()。

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # 【危险命令列表】定义不允许执行的命令关键词。

    if any(d in command for d in dangerous):
        # 【any() 函数】检查是否有任何元素满足条件。类似 Java Stream 的 anyMatch()。
        # 生成器表达式 (d in command for d in dangerous) 类似 Java 的 stream().anyMatch(d -> command.contains(d))。

        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        # 【执行命令】超时时间 120 秒（比后台任务的 300 秒短，因为同步命令应该更快）。

        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
        # 【三元表达式】如果输出不为空则截取前 50000 字符，否则返回 "(no output)"。
        # 类似 Java 的 (out.isEmpty() ? "(no output)" : out.substring(0, Math.min(50000, out.length())))。

    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    # 【读取文件】读取文件内容，可选限制行数。
    # limit: int = None —— 参数默认值 None，类似 Java 中用 null 表示"无限制"。

    try:
        lines = safe_path(path).read_text().splitlines()
        # 【读取文件】read_text() 读取整个文件为字符串（类似 Java 的 Files.readString()）。
        # .splitlines() 按行分割成列表（类似 Java 的 String.split("\\n")，但更智能，支持不同换行符）。

        if limit and limit < len(lines):
            # 【条件判断】limit and ... —— Python 的短路求值，limit 为 None 或 0 时不会继续判断。
            # 类似 Java 的 limit != null && limit < lines.size()。

            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
            # 【切片 + 拼接】取前 limit 行，然后追加一行提示"还有多少行被省略"。
            # 列表拼接用 + 运算符，类似 Java 中 List 的 addAll()。

        return "\n".join(lines)[:50000]
        # 【合并行】用换行符连接所有行，并截取前 50000 字符。

    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    # 【写入文件】将内容写入指定路径。

    try:
        fp = safe_path(path)
        # 【获取安全路径】

        fp.parent.mkdir(parents=True, exist_ok=True)
        # 【创建目录】
        # .parent 获取父目录路径（类似 Java 的 Path.getParent()）。
        # mkdir(parents=True) 类似 Java 的 Files.createDirectories()（递归创建所有不存在的父目录）。
        # exist_ok=True 表示目录已存在时不抛异常（类似 Java 中先检查 !exists() 再创建）。

        fp.write_text(content)
        # 【写入文件】将字符串写入文件。类似 Java 的 Files.writeString(path, content)。

        return f"Wrote {len(content)} bytes"
        # 【返回写入字节数】len() 获取字符串长度（字节数）。

    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    # 【编辑文件】在文件中替换指定文本（只替换第一个匹配）。

    try:
        fp = safe_path(path)

        c = fp.read_text()
        # 【读取文件内容】

        if old_text not in c:
            # 【查找文本】`in` 运算符检查子字符串是否存在。类似 Java 的 str.contains()。

            return f"Error: Text not found in {path}"

        fp.write_text(c.replace(old_text, new_text, 1))
        # 【替换文本】str.replace(old, new, count) 只替换前 count 个匹配。
        # 第三个参数 1 表示只替换第一个匹配，类似 Java 中手动查找索引然后替换。

        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# ============================================================
# TOOL_HANDLERS：工具名称 -> 处理函数的映射（策略模式）
# ============================================================
# 【设计模式】策略模式 + 命令模式：
# 每个工具名映射到一个处理函数，Agent 调用工具时通过名称查找对应的处理函数。
# 类似 Java 中的 Map<String, Function<Map<String, Object>, String>> toolHandlers。

TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    # 【Lambda 表达式】lambda **kw: ... 定义匿名函数。
    # **kw 表示接收任意关键字参数，打包成字典 kw（类似 Java 的 Map<String, Object>）。
    # 类似 Java 的 (Map<String, Object> kw) -> runBash((String) kw.get("command"))

    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    # 【kw.get("limit")】可能返回 None（如果参数不存在），对应 run_read 的默认参数 limit=None。
    # 类似 Java 中 map.getOrDefault("limit", null)，但 Java 没有可选参数概念。

    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run":   lambda **kw: BG.run(kw["command"]),
    # 【后台执行】调用 BG（全局 BackgroundManager）的 run 方法，在后台线程中执行命令。

    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    # 【检查后台任务】task_id 是可选参数，所以用 kw.get() 而不是 kw["task_id"]。
    # kw.get() 在 key 不存在时返回 None，而 kw["task_id"] 会抛 KeyError。
    # 类似 Java 中 map.get() 返回 null vs map.get() 可能 NPE（但 Java 的 map.get() 不会 NPE）。
}

# ============================================================
# TOOLS：工具定义列表（Schema 声明，告诉 LLM 有哪些工具可用）
# ============================================================
# 【Schema 定义】每个工具的 JSON Schema 描述，告诉 Claude 这个工具的名称、描述和参数格式。
# 这类似于 Java 中的接口定义或 Swagger/OpenAPI 规范。

TOOLS = [
    # 【列表】用 [] 定义列表，元素用逗号分隔。类似 Java 的 List.of(...) 或 Arrays.asList(...)。

    {"name": "bash", "description": "Run a shell command (blocking).",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    # 【字典嵌套】字典中可以嵌套字典。类似 Java 中的 Map<String, Object>，value 可以是另一个 Map。
    # input_schema 是 JSON Schema 格式，描述工具的输入参数。
    # required 数组指定必填参数。

    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    # limit 是可选参数（不在 required 数组中），类似 Java 中 @RequestParam(required = false)。

    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},

    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    # 【核心工具】后台运行命令，立即返回 task_id。这是本文件的核心功能。

    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    # 【检查任务】没有 required 字段，因为 task_id 是可选的（省略则列出所有任务）。
]

# ============================================================
# agent_loop：Agent 主循环（ReAct 模式）
# ============================================================
# 【ReAct 模式】Reasoning + Acting 循环：
# 1. 排空后台通知队列（如果有结果，注入到消息中）
# 2. 调用 LLM
# 3. 如果 LLM 决定使用工具，执行工具并将结果返回给 LLM
# 4. 重复直到 LLM 不再使用工具

def agent_loop(messages: list):
    # 【Agent 主循环】接收消息列表，循环调用 LLM 并执行工具。

    while True:
        # 【无限循环】while True 类似 Java 的 while (true)。

        # Drain background notifications and inject as system message before LLM call
        # 【排空后台通知】在每次 LLM 调用之前，检查是否有后台任务完成。

        notifs = BG.drain_notifications()
        # 获取并清空所有已完成的后台任务通知。

        if notifs and messages:
            # 【短路求值】如果 notifs 非空且 messages 非空，才执行注入。
            # Python 中空列表 [] 是 falsy 值，非空列表是 truthy 值。

            notif_text = "\n".join(
                # 【生成器表达式】在 join() 中使用生成器表达式，类似 Java Stream 的 map + collect。
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            # 将所有通知格式化为文本，每行一个通知。

            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
            # 【注入通知】将后台任务结果作为用户消息注入到对话中。
            # 使用 XML 标签 <background-results> 包裹，方便 LLM 理解这是后台任务结果。
            # .append() 向列表末尾添加元素，类似 Java 的 list.add()。

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 【调用 Claude API】发送消息给 Claude，获取响应。
        # model: 模型名称
        # system: 系统提示词（类似 Java 中的系统配置）
        # messages: 对话历史（类似 Java 中的 List<Message>）
        # tools: 可用工具列表
        # max_tokens: 最大生成 token 数（类似 Java 中的 maxOutputTokens）

        messages.append({"role": "assistant", "content": response.content})
        # 【保存助手回复】将 Claude 的回复添加到消息历史中。

        if response.stop_reason != "tool_use":
            # 【检查停止原因】如果 Claude 不需要使用工具（即直接给出了最终回复），则退出循环。
            # stop_reason 类似 Java 中的枚举值。

            return
            # 【退出函数】return 无返回值，类似 Java 的 return;。

        # 以下处理 Claude 要求使用工具的情况：
        results = []
        # 存储所有工具的执行结果。

        for block in response.content:
            # 【遍历内容块】response.content 是一个列表，每个元素是一个内容块。
            # 类似 Java 的 for (ContentBlock block : response.getContent())。

            if block.type == "tool_use":
                # 【工具调用块】如果内容块类型是工具调用。

                handler = TOOL_HANDLERS.get(block.name)
                # 【查找处理函数】通过工具名在 TOOL_HANDLERS 字典中查找对应的处理函数。

                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    # 【条件表达式】如果 handler 存在，调用 handler(**block.input)。
                    # **block.input 是字典解包，将字典的键值对展开为函数的关键字参数。
                    # 类似 Java 中通过反射将 Map 的 key-value 映射到方法参数（但 Python 是语法级别支持）。

                except Exception as e:
                    output = f"Error: {e}"
                    # 【异常处理】工具执行出错时返回错误信息，而不是让程序崩溃。

                print(f"> {block.name}:")
                # 【打印工具名】在控制台输出正在执行的工具名称。

                print(str(output)[:200])
                # 【打印输出】只打印前 200 个字符，防止输出过长。

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                # 【追加结果】将工具执行结果添加到结果列表中。
                # tool_use_id 用于关联请求和响应（类似 Java 中的 requestId）。

        messages.append({"role": "user", "content": results})
        # 【发送工具结果】将所有工具执行结果作为用户消息发送回 Claude。
        # Claude 收到工具结果后会继续思考，可能调用更多工具或给出最终回复。


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 【入口保护】__name__ 是 Python 内置变量，当脚本被直接运行时 __name__ == "__main__"。
    # 当脚本被 import 导入时 __name__ == 模块名，不会执行以下代码。
    # 类似 Java 的 public static void main(String[] args)，但更灵活（因为 Java 没有 import 时执行代码的概念）。

    history = []
    # 【对话历史】存储所有对话消息，类似 Java 的 List<Message> history = new ArrayList<>()。

    while True:
        # 【交互式循环】类似 Java 的 Scanner 循环读取用户输入。

        try:
            query = input("\033[36ms08 >> \033[0m")
            # 【读取输入】input() 从标准输入读取一行，类似 Java 的 Scanner.nextLine()。
            # \033[36m 是 ANSI 转义码，设置终端颜色为青色（cyan）。
            # \033[0m 重置颜色。类似 Java 中使用 ANSI_COLOR_CODES 但 Python 更常用。

        except (EOFError, KeyboardInterrupt):
            # 【异常捕获】(异常1, 异常2) 可以同时捕获多种异常。
            # EOFError: 输入流结束（类似 Java 中 Scanner 遇到 EOF）
            # KeyboardInterrupt: 用户按 Ctrl+C（类似 Java 中 Thread.interrupted()）
            # 类似 Java 的 catch (EOFException | InterruptedException e)。

            break
            # 【退出循环】类似 Java 的 break;。

        if query.strip().lower() in ("q", "exit", ""):
            # 【退出条件】
            # .strip() 去除首尾空白（类似 Java 的 String.trim()）。
            # .lower() 转小写（类似 Java 的 String.toLowerCase()）。
            # in ("q", "exit", "") 检查值是否在元组中（元组用 () 定义，类似 Java 的 List.of() 或数组）。

            break

        history.append({"role": "user", "content": query})
        # 【保存用户消息】

        agent_loop(history)
        # 【调用 Agent 循环】处理用户请求，包括调用 LLM 和执行工具。

        response_content = history[-1]["content"]
        # 【获取最后一条消息】history[-1] 是负索引，表示倒数第一个元素。
        # 类似 Java 的 history.get(history.size() - 1)。

        if isinstance(response_content, list):
            # 【类型检查】isinstance() 检查变量是否是指定类型。
            # 类似 Java 的 if (responseContent instanceof List)。
            # response_content 可能是字符串（普通文本回复）或列表（包含多个内容块，如工具调用）。

            for block in response_content:
                # 【遍历内容块】

                if hasattr(block, "text"):
                    # 【属性检查】hasattr() 检查对象是否有指定属性。
                    # 类似 Java 中的反射：field != null 或 instanceof 检查。
                    # 这里 block 可能是 TextBlock（有 text 属性）或 ToolUseBlock（没有 text 属性）。

                    print(block.text)
                    # 【打印文本内容】

        print()
        # 【空行】输出一个空行，让控制台更易读。
