#!/usr/bin/env python3  # Unix shebang 行，指定使用 python3 解释器执行此脚本，类似于 Java 中没有直接对应的机制（Java 通过 JVM 运行）
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents.py - 自主代理（Autonomous Agents）

本文件实现了一个自主代理系统，核心理念是"代理自己找工作"。
在空闲时轮询任务板、自动认领未认领的任务，以及在上下文压缩后重新注入身份信息。
基于 s10 的协议构建。

    队友（Teammate）生命周期:
    +-------+
    | spawn |  生成/启动阶段
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |  工作阶段：LLM 反复调用工具
    +---+---+              +-------+
        |
        | stop_reason != tool_use（LLM 不再请求工具时退出工作循环）
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s  空闲阶段：每 5 秒轮询一次，最多 60 秒
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK    检查收件箱，有消息就恢复工作
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK  扫描任务板，有未认领任务就认领并恢复工作
        |
        +---> timeout (60s) -> shutdown  超时 60 秒则关闭

    上下文压缩后的身份重新注入:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

核心要点: "代理自己找工作" —— 这是自主代理的关键特征
"""

# ============================================================
# 导入标准库模块 —— Python 的 import 类似于 Java 的 import，但不需要指定完整包路径
# ============================================================
import json       # JSON 序列化/反序列化库，类似于 Java 的 com.google.gson.Gson 或 Jackson
import os         # 操作系统接口模块，提供文件路径、环境变量等操作，类似于 Java 的 System.getenv() + java.io.File
import subprocess # 子进程管理模块，用于执行 shell 命令，类似于 Java 的 ProcessBuilder
import threading  # 线程模块，提供基本的线程和锁机制，类似于 Java 的 java.util.concurrent 包
import time       # 时间相关函数，类似于 Java 的 System.currentTimeMillis() 等
import uuid       # UUID 生成模块，类似于 Java 的 java.util.UUID
from pathlib import Path  # 面向对象的文件路径库，类似于 Java 的 java.nio.file.Path

# ============================================================
# 导入第三方库
# ============================================================
from anthropic import Anthropic  # Anthropic 官方 Python SDK，用于调用 Claude API
from dotenv import load_dotenv   # python-dotenv 库，从 .env 文件加载环境变量，类似于 Spring Boot 的 application.properties

# ============================================================
# 环境初始化
# ============================================================
load_dotenv(override=True)  # 加载 .env 文件中的环境变量；override=True 表示覆盖已有的同名变量
# 如果设置了自定义 API 地址（ANTHROPIC_BASE_URL），则移除认证令牌，避免冲突
if os.getenv("ANTHROPIC_BASE_URL"):  # os.getenv() 类似于 Java 的 System.getenv()，获取环境变量
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)  # os.environ 是一个类似 Map 的字典，pop 类似于 Map.remove()

# ============================================================
# 全局常量定义 —— Python 中没有 final 关键字，通常用全大写命名约定来表示常量
# ============================================================
WORKDIR = Path.cwd()  # Path.cwd() 获取当前工作目录，类似于 Java 的 Paths.get("").toAbsolutePath()
# 创建 Anthropic API 客户端实例，如果设置了自定义 BASE_URL 则使用它（用于代理或本地部署）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]  # 从环境变量获取模型 ID（如 claude-sonnet-4-20250514）
TEAM_DIR = WORKDIR / ".team"    # Path 对象支持 / 运算符拼接路径，类似于 Java 的 Path.resolve()
INBOX_DIR = TEAM_DIR / "inbox"  # 收件箱目录，每个队友有一个 {name}.jsonl 文件
TASKS_DIR = WORKDIR / ".tasks"  # 任务板目录，存放 task_*.json 格式的任务文件

# ============================================================
# 轮询参数
# ============================================================
POLL_INTERVAL = 5   # 空闲轮询间隔：5 秒（单位：秒）
IDLE_TIMEOUT = 60    # 空闲超时时间：60 秒（单位：秒）

# ============================================================
# 系统提示词 —— f-string 是 Python 的格式化字符串，类似于 Java 的 String.format() 或文本块
# f"...{expression}..." 中的 {expression} 会被替换为表达式的值
# ============================================================
SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

# ============================================================
# 合法的消息类型集合 —— 使用 set（集合）定义，类似于 Java 的 Set<String>
# Python 集合用花括号 {} 表示，元素之间用逗号分隔
# ============================================================
VALID_MSG_TYPES = {
    "message",                  # 普通消息
    "broadcast",                # 广播消息
    "shutdown_request",         # 关闭请求
    "shutdown_response",        # 关闭响应
    "plan_approval_response",   # 计划审批响应
}

# ============================================================
# 请求追踪器（Request trackers）—— 模块级别的全局变量
# Python 中模块级别的变量类似于 Java 的 static 变量
# ============================================================
shutdown_requests = {}  # 关闭请求追踪字典，key 是 request_id，value 是请求详情，类似于 Java 的 ConcurrentHashMap<String, Request>
plan_requests = {}     # 计划请求追踪字典
_tracker_lock = threading.Lock()  # 追踪器的线程锁，类似于 Java 的 ReentrantLock()
_claim_lock = threading.Lock()    # 任务认领的线程锁，保证同一时间只有一个线程能认领任务


# ============================================================
# MessageBus: 消息总线 —— 基于文件系统的消息传递机制
# 设计模式：类似于 Java 中的 Message Queue（消息队列）的简化版本
# 每个队友有一个独立的 JSONL 格式收件箱文件（JSONL = 每行一个 JSON 对象）
# ============================================================
class MessageBus:  # class 定义类，类似于 Java 的 public class MessageBus
    def __init__(self, inbox_dir: Path):  # __init__ 是构造方法，类似于 Java 的构造函数；self 类似于 Java 的 this
        self.dir = inbox_dir  # 实例属性赋值，类似于 Java 的 this.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)  # 创建目录（含父目录），exist_ok=True 类似于 Java 的 Files.createDirectories()

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        # 发送消息到指定队友的收件箱
        # 参数类型注解（: str, : int 等）是 Python 3.5+ 的类型提示，类似于 Java 的类型声明但不强制检查
        # msg_type: str = "message" 是默认参数，类似于 Java 中方法重载的默认值版本
        # extra: dict = None 是可选参数，None 类似于 Java 的 null
        if msg_type not in VALID_MSG_TYPES:  # 检查消息类型是否合法
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {  # 使用字典字面量创建消息对象，类似于 Java 中 new HashMap<>() 然后 put()
            "type": msg_type,
            "from": sender,     # 注意 "from" 是 Python 关键字，但作为字典的字符串键是合法的
            "content": content,
            "timestamp": time.time(),  # 当前 Unix 时间戳（秒），类似于 Java 的 System.currentTimeMillis() / 1000
        }
        if extra:  # Python 中 None 视为 False，非 None 视为 True；类似于 Java 的 if (extra != null)
            msg.update(extra)  # dict.update() 合并字典，类似于 Java 的 Map.putAll()
        inbox_path = self.dir / f"{to}.jsonl"  # 拼接收件箱文件路径，f-string 类似于 String.format()
        with open(inbox_path, "a") as f:  # with 语句是上下文管理器，类似于 Java 的 try-with-resources
            # "a" 表示追加模式（append），类似于 Java 的 new FileWriter(path, true)
            f.write(json.dumps(msg) + "\n")  # json.dumps() 序列化为 JSON 字符串，类似于 Jackson 的 objectMapper.writeValueAsString()
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:  # 返回类型 list，类似于 Java 的 List<Object>
        # 读取并清空指定队友的收件箱（读取后清空，即"消费"模式）
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():  # 检查文件是否存在，类似于 Java 的 Files.exists()
            return []  # 返回空列表
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            # read_text() 读取整个文件为字符串，类似于 Java 的 Files.readString()
            # .strip() 去除首尾空白，类似于 Java 的 String.trim()
            # .splitlines() 按行分割，类似于 Java 的 String.split("\\n")
            if line:  # 跳过空行
                messages.append(json.loads(line))  # json.loads() 反序列化 JSON，类似于 Jackson 的 objectMapper.readValue()
        inbox_path.write_text("")  # 读取后清空收件箱文件（写空字符串即清空）
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        # 向所有队友广播消息（不包括发送者自己）
        count = 0
        for name in teammates:  # for...in 循环遍历列表，类似于 Java 的 for (String name : teammates)
            if name != sender:  # 不发送给自己
                self.send(sender, name, content, "broadcast")  # 递归调用 send 方法
                count += 1
        return f"Broadcast to {count} teammates"


# 创建全局消息总线实例（单例模式），类似于 Java 的 private static final BUS = new MessageBus(INBOX_DIR)
BUS = MessageBus(INBOX_DIR)


# ============================================================
# 任务板扫描与认领功能
# 任务板是一个基于文件系统的看板系统，每个任务是一个 task_{id}.json 文件
# 设计模式：类似于 Java 中简单的文件数据库 + Repository 模式
# ============================================================
def scan_unclaimed_tasks() -> list:
    # 扫描所有未认领的任务
    # -> list 是返回类型注解，表示返回一个列表，类似于 Java 的 List<Task>
    TASKS_DIR.mkdir(exist_ok=True)  # 确保任务目录存在
    unclaimed = []  # 未认领任务列表
    # Path.glob("task_*.json") 使用通配符匹配文件，类似于 Java 的 Files.list().filter()
    # sorted() 对结果排序，确保每次扫描顺序一致
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())  # 读取并解析任务 JSON 文件
        # dict.get(key) 安全获取字典值，key 不存在时返回 None（不会抛异常），类似于 Java 的 Map.getOrDefault(key, null)
        if (task.get("status") == "pending"       # 状态为待处理
                and not task.get("owner")          # 尚未被认领（无 owner）
                and not task.get("blockedBy")):    # 没有被其他任务阻塞
            unclaimed.append(task)  # 添加到未认领列表
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    # 认领任务（线程安全）
    # task_id: 任务 ID；owner: 认领者名称
    # with 语句用于获取锁，类似于 Java 的 synchronized 块或 lock.lock() / try / finally
    with _claim_lock:  # 获取 _claim_lock 锁，退出 with 块时自动释放（类似于 Java 的 try-with-resources）
        path = TASKS_DIR / f"task_{task_id}.json"  # 任务文件路径
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):  # 检查是否已被认领
            existing_owner = task.get("owner") or "someone else"  # or 是短路或运算符，如果 owner 为 None 则用 "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":  # 检查状态是否为待处理
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):  # 检查是否被阻塞
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner  # 设置认领者
        task["status"] = "in_progress"  # 更新状态为进行中
        path.write_text(json.dumps(task, indent=2))  # 写回文件，indent=2 表示缩进 2 格（美化 JSON）
    return f"Claimed task #{task_id} for {owner}"


# ============================================================
# 身份重新注入 —— 上下文压缩后的身份恢复机制
# 当对话历史过长被截断时，代理可能丢失自己的身份信息
# 通过在消息列表头部插入身份块来恢复代理的身份认知
# ============================================================
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    # 构建身份消息块，返回 Claude API 所需的消息格式
    # -> dict 表示返回一个字典，类似于 Java 的 Map<String, String>
    return {
        "role": "user",  # 消息角色：user（用户），assistant（助手），system（系统）
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
        # 使用 XML 标签 <identity>...</identity> 包裹身份信息，这是一种 Prompt Engineering 技巧，便于 LLM 理解结构化信息
    }


# ============================================================
# TeammateManager: 队友管理器 —— 管理自主代理的生命周期
# 设计模式：类似于 Java 中 Actor 系统的 ActorRef 管理器 + 线程池管理
# 负责生成（spawn）、管理队友，每个队友运行在独立的线程中
# ============================================================
class TeammateManager:
    def __init__(self, team_dir: Path):  # 构造方法，接收团队配置目录路径
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)  # 创建团队目录
        self.config_path = self.dir / "config.json"  # 配置文件路径
        self.config = self._load_config()  # 加载或创建默认配置
        self.threads = {}  # 线程字典：{队友名: Thread对象}，类似于 Java 的 Map<String, Thread>

    def _load_config(self) -> dict:
        # 加载团队配置（私有方法，以 _ 开头表示内部使用，类似于 Java 的 private 方法）
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())  # 读取并解析 JSON 配置
        return {"team_name": "default", "members": []}  # 返回默认配置

    def _save_config(self):
        # 保存团队配置到 JSON 文件
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        # 在成员列表中查找指定名称的队友，返回成员字典或 None
        for m in self.config["members"]:  # 遍历成员列表
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        # 更新队友状态并持久化
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()  # 状态变更后立即保存到文件

    def spawn(self, name: str, role: str, prompt: str) -> str:
        # 生成（spawn）一个队友代理 —— 核心方法
        # name: 队友名称；role: 角色（如 backend、frontend）；prompt: 初始任务描述
        member = self._find_member(name)
        if member:  # 如果队友已存在（重新生成的情况）
            if member["status"] not in ("idle", "shutdown"):  # 只有空闲或已关闭的队友才能重新生成
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"  # 更新状态为工作中
            member["role"] = role
        else:  # 新队友
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)  # list.append() 类似于 Java 的 List.add()
        self._save_config()
        # 创建并启动一个新线程来运行队友的主循环
        # threading.Thread 类似于 Java 的 Thread 类
        thread = threading.Thread(
            target=self._loop,   # 线程要执行的方法（类似于 Java 的 Runnable）
            args=(name, role, prompt),  # 传递给目标方法的参数（元组），类似于 Java 构造 Thread 时传递参数
            daemon=True,  # daemon=True 表示守护线程，主线程退出时自动终止，类似于 Java 的 Thread.setDaemon(true)
        )
        self.threads[name] = thread  # 保存线程引用
        thread.start()  # 启动线程，类似于 Java 的 thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        # 队友主循环 —— 这是自主代理的核心逻辑，运行在独立线程中
        # 包含两个阶段交替运行：WORK（工作阶段）和 IDLE（空闲轮询阶段）
        team_name = self.config["team_name"]
        # 构建系统提示词，使用括号 () 隐式拼接多行字符串（Python 特性），类似于 Java 的字符串拼接
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]  # 初始对话历史，包含用户的初始任务
        tools = self._teammate_tools()  # 获取队友可用的工具列表

        while True:  # 外层循环：WORK -> IDLE -> WORK -> ... 无限循环直到关闭
            # ============================================================
            # WORK PHASE: 工作阶段 —— 标准 Agent 循环（LLM 反复调用工具）
            # 最多执行 50 轮工具调用，防止无限循环
            # ============================================================
            for _ in range(50):  # range(50) 生成 0-49 的整数序列，类似于 Java 的 IntStream.range(0, 50)
                # _ 是 Python 惯例，表示循环变量不使用（类似于 Java 中不使用循环变量的写法）
                inbox = BUS.read_inbox(name)  # 每轮开始时检查收件箱
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":  # 收到关闭请求，立即退出
                        self._set_status(name, "shutdown")
                        return  # 从方法返回（线程结束），类似于 Java 中从 Runnable.run() 中 return
                    messages.append({"role": "user", "content": json.dumps(msg)})  # 将收到的消息添加到对话历史
                try:
                    # 调用 Claude API —— 核心步骤
                    # client.messages.create() 类似于 Java 中调用 HTTP API 发送 POST 请求
                    response = client.messages.create(
                        model=MODEL,          # 使用的模型 ID
                        system=sys_prompt,    # 系统提示词
                        messages=messages,    # 对话历史（包含所有之前的消息）
                        tools=tools,          # 可用工具定义
                        max_tokens=8000,      # 最大生成 token 数（控制响应长度）
                    )
                except Exception:  # 捕获所有异常（API 调用失败等），类似于 Java 的 catch (Exception e)
                    self._set_status(name, "idle")
                    return
                # 将 LLM 的响应添加到对话历史
                messages.append({"role": "assistant", "content": response.content})
                # response.stop_reason 表示 LLM 停止的原因：
                # "tool_use" = LLM 想调用工具（继续循环）
                # "end_turn" = LLM 认为对话结束
                if response.stop_reason != "tool_use":
                    break  # LLM 不再请求工具，退出工作循环
                # 处理 LLM 请求的工具调用
                results = []  # 收集所有工具调用的结果
                idle_requested = False  # 标记 LLM 是否请求进入空闲状态
                # response.content 是一个列表，包含文本块和工具调用块
                for block in response.content:
                    if block.type == "tool_use":  # 只处理工具调用块
                        if block.name == "idle":  # LLM 调用了 idle 工具，表示没有更多工作
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:  # 执行其他工具
                            output = self._exec(name, block.name, block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")  # 打印工具调用日志，[:120] 截取前 120 字符
                        results.append({
                            "type": "tool_result",   # 工具结果类型
                            "tool_use_id": block.id,  # 对应工具调用的 ID（用于关联请求和结果）
                            "content": str(output),    # 工具执行结果
                        })
                messages.append({"role": "user", "content": results})  # 将工具结果作为用户消息发回 LLM
                if idle_requested:
                    break  # LLM 请求空闲，退出工作循环

            # ============================================================
            # IDLE PHASE: 空闲轮询阶段 —— 每 5 秒检查一次收件箱和任务板
            # 这就是"自主"的体现：代理自己找工作，不需要被外部调度
            # ============================================================
            self._set_status(name, "idle")  # 将状态设为空闲
            resume = False  # 是否恢复工作的标志
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)  # 计算最大轮询次数：60 / 5 = 12 次
            # max() 函数取最大值，防止 POLL_INTERVAL 为 0 导致除零错误
            for _ in range(polls):  # 最多轮询 12 次（60 秒超时）
                time.sleep(POLL_INTERVAL)  # 等待 5 秒，类似于 Java 的 Thread.sleep(5000)
                # 检查收件箱
                inbox = BUS.read_inbox(name)
                if inbox:  # 有新消息，恢复工作
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":  # 收到关闭请求
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break  # 跳出轮询循环，恢复工作
                # 检查任务板是否有未认领的任务
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:  # 有未认领任务，自动认领第一个
                    task = unclaimed[0]  # 取第一个任务（列表索引从 0 开始，类似于 Java 的 list.get(0)）
                    result = claim_task(task["id"], name)
                    if result.startswith("Error:"):  # 认领失败（可能已被其他队友认领）
                        continue  # 继续轮询，尝试下一个任务
                    # 认领成功，构建任务提示
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    # 身份重新注入：如果对话历史很短（<=3 条消息），说明可能经过了截断，需要重新注入身份
                    if len(messages) <= 3:
                        # list.insert(index, item) 在指定位置插入元素，类似于 Java 的 List.add(index, item)
                        messages.insert(0, make_identity_block(name, role, team_name))  # 在头部插入身份信息
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})  # 插入助手的确认
                    messages.append({"role": "user", "content": task_prompt})  # 添加任务描述
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})  # 模拟助手的回应
                    resume = True
                    break  # 跳出轮询循环，恢复工作

            if not resume:  # 超时且没有恢复工作
                self._set_status(name, "shutdown")
                return  # 线程结束
            self._set_status(name, "working")  # 恢复工作状态，回到外层 while True 循环

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # 工具调度方法 —— 根据 LLM 请求的工具名分发执行
        # sender: 发送请求的队友名；tool_name: 工具名；args: 工具参数（字典）
        # 使用 if-elif 链进行分发，类似于 Java 中没有 switch 表达式时的 if-else 链
        if tool_name == "bash":
            return _run_bash(args["command"])  # 执行 shell 命令
        if tool_name == "read_file":
            return _run_read(args["path"])  # 读取文件
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])  # 写入文件
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])  # 编辑文件
        if tool_name == "send_message":
            # args.get("msg_type", "message") 获取参数，如果不存在则返回默认值 "message"，类似于 Java 的 Map.getOrDefault()
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)  # 读取收件箱并格式化为 JSON
        if tool_name == "shutdown_response":
            # 处理关闭请求的响应
            req_id = args["request_id"]
            with _tracker_lock:  # 使用线程锁保护共享数据
                if req_id in shutdown_requests:  # 检查字典中是否存在某个 key，类似于 Java 的 Map.containsKey()
                    # 三元表达式：value_if_true if condition else value_if_false，类似于 Java 的 condition ? a : b
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),  # 发送关闭响应给 lead
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            # 队友提交计划给 lead 审批
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]  # 生成 8 字符的短 UUID，str() 转为字符串，[:8] 截取前 8 位
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)  # 认领任务
        return f"Unknown tool: {tool_name}"  # 未知工具返回错误信息

    def _teammate_tools(self) -> list:
        # 定义队友可用的工具列表 —— 返回 Claude API 所需的工具定义格式
        # 每个工具包含 name（名称）、description（描述）、input_schema（JSON Schema 格式的参数定义）
        # 这些基础工具与 s02 相同
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             # list(VALID_MSG_TYPES) 将 set 转为 list，因为 JSON Schema 的 enum 需要数组格式
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},  # 无参数的工具
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},  # idle 工具无需参数
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    def list_all(self) -> str:
        # 列出所有队友及其状态
        if not self.config["members"]:  # 空列表为 False
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]  # 用列表收集字符串行，然后用 join 拼接
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)  # str.join() 类似于 Java 的 String.join("\n", lines)

    def member_names(self) -> list:
        # 返回所有队友名称列表
        # 列表推导式（list comprehension）是 Python 特有的语法，类似于 Java Stream 的 map + collect
        # [表达式 for 变量 in 可迭代对象] 等价于 Java 的 list.stream().map(m -> m.name).collect(Collectors.toList())
        return [m["name"] for m in self.config["members"]]


# 创建全局 TeammateManager 实例（单例），类似于 Java 的 private static final TEAM = new TeammateManager(TEAM_DIR)
TEAM = TeammateManager(TEAM_DIR)


# ============================================================
# 基础工具实现（与 s02 相同）—— 底层文件和 shell 操作函数
# 这些函数以 _ 开头表示模块私有（Python 没有真正的 private 访问修饰符）
# ============================================================
def _safe_path(p: str) -> Path:
    # 安全路径检查 —— 防止路径遍历攻击（Path Traversal）
    # resolve() 解析为绝对路径并消除 .. 等符号链接，类似于 Java 的 Path.toAbsolutePath().normalize()
    path = (WORKDIR / p).resolve()
    # is_relative_to() 检查路径是否在 WORKDIR 内，防止访问工作目录之外的文件
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")  # 抛出异常，类似于 Java 的 throw new IllegalArgumentException()
    return path


def _run_bash(command: str) -> str:
    # 执行 shell 命令（带安全检查）
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]  # 危险命令黑名单
    # any() 函数：如果任何一个元素为 True 则返回 True，类似于 Java Stream 的 anyMatch()
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # subprocess.run() 执行外部命令，类似于 Java 的 ProcessBuilder
        # shell=True 表示通过 shell 执行（允许管道等 shell 特性）
        # capture_output=True 捕获 stdout 和 stderr
        # text=True 将输出解码为字符串（而非字节）
        # timeout=120 设置超时 120 秒
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()  # 合并标准输出和错误输出
        return out[:50000] if out else "(no output)"  # 截取前 50000 字符防止输出过大
    except subprocess.TimeoutExpired:  # 捕获超时异常，类似于 Java 的 catch (TimeoutException e)
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    # 读取文件内容
    # limit: int = None 是可选参数，None 类似于 Java 的 null
    try:
        lines = _safe_path(path).read_text().splitlines()  # 读取文件并按行分割
        if limit and limit < len(lines):  # 如果指定了行数限制且文件行数超过限制
            # 列表切片 lines[:limit] 取前 limit 行，类似于 Java 的 List.subList(0, limit)
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]  # 末尾添加省略提示
        return "\n".join(lines)[:50000]  # 用换行符拼接并截断
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    # 写入文件内容
    try:
        fp = _safe_path(path)  # 安全路径检查
        fp.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        fp.write_text(content)  # 写入内容
        return f"Wrote {len(content)} bytes"  # len() 获取长度，对于字符串是字符数，类似于 Java 的 String.length()
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    # 编辑文件：精确替换文本
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:  # 检查旧文本是否存在（使用 in 运算符），类似于 Java 的 String.contains()
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))  # replace() 替换文本，第三个参数 1 表示只替换第一个匹配
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# Lead（领队）专用的协议处理函数
# Lead 是整个团队的管理者，可以发送关闭请求、审批计划等
# ============================================================
def handle_shutdown_request(teammate: str) -> str:
    # Lead 请求队友关闭
    req_id = str(uuid.uuid4())[:8]  # 生成 8 字符短 UUID 作为请求 ID
    with _tracker_lock:  # 加锁保护共享字典
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}  # 记录请求状态
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",  # 向队友发送关闭请求消息
        "shutdown_request", {"request_id": req_id},  # extra 参数传递请求 ID
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    # Lead 审批队友提交的计划
    # approve: bool 是布尔类型，Python 的 True/False 类似于 Java 的 true/false（注意大小写差异）
    with _tracker_lock:
        req = plan_requests.get(request_id)  # 获取计划请求
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"  # 三元表达式设置状态
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    # 查询关闭请求的状态
    with _tracker_lock:
        # dict.get(key, default) 如果 key 不存在则返回 default，类似于 Java 的 Map.getOrDefault()
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# ============================================================
# Lead 工具调度表 —— 使用策略模式的字典分发
# 设计模式：类似于 Java 中 Map<String, Function<Map, String>> 的策略模式
# 每个键值对将工具名映射到对应的处理函数（lambda 匿名函数）
# lambda **kw: ... 是 Python 的匿名函数（Lambda 表达式），类似于 Java 的 (Map kw) -> expression
# **kw 表示接收任意关键字参数并打包为字典，类似于 Java 的可变参数 Map<String, Object>
# ============================================================
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),  # 执行 shell 命令
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),  # 读取文件，limit 为可选参数
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),  # 写入文件
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),  # 编辑文件
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),  # 生成队友
    "list_teammates":    lambda **kw: TEAM.list_all(),  # 列出所有队友（忽略参数）
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),  # 发送消息
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),  # 读取 Lead 收件箱
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),  # 广播消息
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),  # 请求队友关闭
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),  # 查询关闭状态
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),  # 审批计划
    "idle":              lambda **kw: "Lead does not idle.",  # Lead 不需要空闲（直接返回提示信息）
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),  # Lead 自己认领任务
}

# ============================================================
# Lead 可用的工具定义列表（14 个工具）—— 与 TOOL_HANDLERS 字典一一对应
# 每个工具定义符合 Claude API 的工具格式要求
# input_schema 使用 JSON Schema 格式描述工具参数
# 与 s02 的基础工具相同，但增加了团队管理相关工具
# ============================================================
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},  # 空对象表示无参数
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check shutdown request status.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


# ============================================================
# agent_loop: Lead 的主循环 —— 与 TeammateManager._loop() 结构类似但更简单
# Lead 只有一个工作阶段（没有空闲轮询），因为它由人类用户直接交互驱动
# ============================================================
def agent_loop(messages: list):
    while True:  # 无限循环，直到 LLM 不再请求工具
        # 每轮开始检查 Lead 的收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:  # 如果有新消息
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
                # 使用 XML 标签 <inbox>...</inbox> 包裹收件箱消息，便于 LLM 区分不同类型的信息
            })
        # 调用 Claude API
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,  # Lead 的系统提示词
            messages=messages,  # 完整的对话历史
            tools=TOOLS,  # Lead 可用的 14 个工具
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})  # 记录 LLM 响应
        if response.stop_reason != "tool_use":  # LLM 不再请求工具，退出循环
            return
        # 处理工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)  # 从字典中获取对应的处理函数
                try:
                    # handler(**block.input) 使用 ** 解包字典为关键字参数
                    # **dict 是 Python 的字典解包语法，类似于 Java 中将 Map 的键值对展开为方法参数
                    # 例如 handler(**{"command": "ls"}) 等价于 handler(command="ls")
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])  # 打印工具输出（截取前 200 字符）
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })
        messages.append({"role": "user", "content": results})  # 将工具结果发回 LLM


# ============================================================
# 程序入口 —— if __name__ == "__main__" 是 Python 的惯用写法
# __name__ 是 Python 内置变量，当文件被直接运行时值为 "__main__"，被 import 时值为模块名
# 这类似于 Java 中 public static void main(String[] args) 是程序入口
# 但 Python 中被 import 的模块不会执行此代码块（避免副作用）
# ============================================================
if __name__ == "__main__":
    history = []  # 对话历史列表
    while True:  # 主交互循环
        try:
            # input() 读取用户输入，类似于 Java 的 Scanner.nextLine()
            # "\033[36m" 和 "\033[0m" 是 ANSI 转义码，用于设置终端文字颜色（青色），类似于 Java 终端颜色控制
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):  # 捕获 Ctrl+D (EOF) 和 Ctrl+C (中断)
            # 元组 (EOFError, KeyboardInterrupt) 表示捕获多种异常类型，类似于 Java 的 catch (EOFError | KeyboardInterrupt e)
            break
        if query.strip().lower() in ("q", "exit", ""):
            # .strip() 去除首尾空白，.lower() 转为小写
            # in 操作符检查元素是否在集合中，类似于 Java 的 Set.contains()
            break  # 退出程序
        # 以下 if 语句实现斜杠命令（slash commands），类似于 CLI 工具的特殊命令
        if query.strip() == "/team":
            print(TEAM.list_all())  # 列出所有队友
            continue  # 继续下一次循环（不发送给 LLM）
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))  # 查看 Lead 收件箱
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)  # 确保任务目录存在
            for f in sorted(TASKS_DIR.glob("task_*.json")):  # 遍历所有任务文件
                t = json.loads(f.read_text())
                # 嵌套字典推导任务状态标记，类似于 Java 中使用 Map.getOrDefault()
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                # f-string 内部可使用表达式，包括三元表达式
                owner = f" @{t['owner']}" if t.get("owner") else ""  # 如果有 owner 则显示
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        # 将用户输入添加到对话历史并调用 agent_loop
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 执行 Lead 的 Agent 循环（可能触发多轮工具调用）
        # 从对话历史中提取最后的响应内容并显示
        response_content = history[-1]["content"]  # history[-1] 负数索引，表示最后一个元素，类似于 Java 的 list.get(list.size() - 1)
        if isinstance(response_content, list):  # isinstance() 类型检查，类似于 Java 的 instanceof
            for block in response_content:
                # hasattr() 检查对象是否有指定属性，类似于 Java 的反射 Field 检查
                if hasattr(block, "text"):  # 只处理文本块
                    print(block.text)  # 打印 LLM 的文本响应
        print()  # 打印空行分隔
