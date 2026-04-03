#!/usr/bin/env python3
# 【解释】这是一个 shebang（沙班）行。在 Unix/Linux 系统中，当文件被标记为可执行时，
# 这行告诉操作系统用 python3 解释器来执行此脚本。
# 类似于 Java 中没有直接对应的概念，因为 Java 代码是由 JVM 来运行的，
# 但可以理解为一种"指定运行时"的声明。

# Harness: protocols -- structured handshakes between models.
# 【解释】这是给框架（Harness）看的标记，说明这个脚本使用了 protocols（协议）功能。

"""
s10_team_protocols.py - Team Protocols

Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern. Builds on s09's team messaging.

    Shutdown FSM: pending -> approved | rejected

    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops

    Plan approval FSM: pending -> approved | rejected

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    Trackers: {request_id: {"target|from": name, "status": "pending|..."}}

Key insight: "Same request_id correlation pattern, two domains."
"""
# 【解释】上面的三引号字符串（三个双引号 """..."""）是 Python 的多行字符串。
# 当它紧接在模块定义的开头时，就变成了"文档字符串"（docstring）。
# 在 Java 中类似 Javadoc 注释（/** ... */），但 docstring 是实际的对象属性，
# 可以在运行时通过模块名.__doc__ 访问，而 Javadoc 不会编译到 class 文件中。
# docstring 中描述了两种协议的 FSM（有限状态机）流程图和请求追踪器的数据结构。

# ==================== 导入部分 ====================
# 【解释】Python 的 import 语句类似 Java 的 import，但更灵活。
# Java 中是 import com.example.Package.Class;
# Python 中可以是 import module（导入整个模块）或 from module import name（导入特定成员）。

import json
# 【解释】导入 Python 标准库的 json 模块，用于 JSON 序列化/反序列化。
# 类似于 Java 中使用 com.google.gson.Gson 或 org.json.JSONObject。
# 但 json 是 Python 内置模块，无需额外依赖。
import os
# 【解释】导入 os 模块，提供操作系统相关的功能（环境变量、文件路径等）。
# 类似于 Java 的 System.getenv() 和 java.io.File 的组合。

import subprocess
# 【解释】导入 subprocess 模块，用于创建子进程执行 shell 命令。
# 类似于 Java 的 ProcessBuilder 或 Runtime.exec()。

import threading
# 【解释】导入 threading 模块，提供线程支持。
# 类似于 Java 的 java.lang.Thread 和 java.util.concurrent 包。
# 注意：Python 有 GIL（全局解释器锁），多线程在 CPU 密集型任务中不会真正并行，
# 但对于 I/O 密集型任务（如网络请求、文件操作）仍然有效。

import time
# 【解释】导入 time 模块，提供时间相关函数（时间戳、睡眠等）。
# 类似于 Java 的 System.currentTimeMillis() 和 Thread.sleep()。

import uuid
# 【解释】导入 uuid 模块，用于生成唯一标识符（UUID）。
# 类似于 Java 的 java.util.UUID.randomUUID()。

from pathlib import Path
# 【解释】从 pathlib 模块导入 Path 类。这是 Python 3.4+ 推荐的路径操作方式。
# Path 类似于 Java 的 java.nio.file.Path，提供面向对象的路径操作。
# "from X import Y" 语法类似 Java 的 "import X.Y;" 或 static import。

from anthropic import Anthropic
# 【解释】从 anthropic 第三方包导入 Anthropic 类，这是 Anthropic API 的 Python SDK 客户端。
# 类似于 Java 中 import com.anthropic.sdk.Anthropic; 需要先 pip install anthropic。

from dotenv import load_dotenv
# 【解释】从 dotenv 包导入 load_dotenv 函数，用于从 .env 文件加载环境变量。
# 类似于 Java 中使用 io.github.cdimascio.dotenv.Dotenv 类。

# ==================== 环境配置 ====================
load_dotenv(override=True)
# 【解释】调用 load_dotenv 加载 .env 文件中的环境变量到 os.environ 中。
# override=True 表示如果环境变量已存在则覆盖。类似 Java dotenv 库的 configure() 方法。

if os.getenv("ANTHROPIC_BASE_URL"):
# 【解释】os.getenv() 获取环境变量的值，如果不存在则返回 None（不会抛异常）。
# 类似于 Java 的 System.getenv("KEY")，但 Java 不存在时也返回 null。

    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    # 【解释】os.environ 是一个类字典对象，存储了所有环境变量，类似 Java 的 System.getenv()。
    # pop(key, default) 方法移除并返回指定 key，如果 key 不存在则返回 default（这里是 None）。
    # 类似于 Java 中 System.getProperties().remove("KEY")，但更安全（不会抛异常）。

# ==================== 全局常量 ====================
# 【解释】Python 没有像 Java 那样的 final 关键字来定义常量。
# 约定俗成地，全大写命名的变量被视为常量（如 WORKDIR、MODEL）。
# 但 Python 并不强制不可修改，这是一种命名约定（类似 Java 中 private static final 但没有编译器检查）。

WORKDIR = Path.cwd()
# 【解释】Path.cwd() 返回当前工作目录（Current Working Directory）。
# 类似于 Java 的 Path.of("").toAbsolutePath() 或 System.getProperty("user.dir")。
# WORKDIR 是一个 Path 对象，支持 / 运算符进行路径拼接（见下面）。

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 【解释】创建 Anthropic API 客户端实例。
# base_url 参数通过 os.getenv() 从环境变量获取 API 基础 URL。
# 类似于 Java 中 new AnthropicClient.Builder().baseUrl(env.get("BASE_URL")).build()。

MODEL = os.environ["MODEL_ID"]
# 【解释】从环境变量获取模型 ID。注意：os.environ["KEY"] 与 os.getenv("KEY") 不同，
# 前者在 key 不存在时会抛出 KeyError 异常（类似 Java 中直接 map.get(key) 返回 null，
# 但 map[key] 则抛出异常 —— 实际上 Java 的 Map 没有这种语法，Python dict 可以用 [] 访问）。

TEAM_DIR = WORKDIR / ".team"
# 【解释】Path 对象重载了 __truediv__（/）运算符，用于路径拼接。
# WORKDIR / ".team" 等价于 WORKDIR.joinpath(".team")。
# 这是一个非常优雅的语法，类似 Java 中 path.resolve(".team")。

INBOX_DIR = TEAM_DIR / "inbox"
# 【解释】继续使用 / 运算符拼接路径：.team/inbox。类似 Java 中 teamDir.resolve("inbox")。

# ==================== 系统提示词 ====================
SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."
# 【解释】f-string（格式化字符串），以 f 开头的字符串中，花括号 {} 内的表达式会被求值并插入。
# 类似 Java 的 String.format("...%s...", value) 或 Java 15+ 的文本块 + String.formatted()。
# 但 f-string 更强大：{} 内可以是任意 Python 表达式，如 {1 + 2}、{obj.method()} 等。

# ==================== 消息类型白名单 ====================
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}
# 【解释】这里用花括号 {} 创建的是一个 set（集合），不是 dict（字典）。
# 判断依据：元素之间用逗号分隔，没有冒号 :。set 类似于 Java 的 java.util.HashSet<String>，
# 存储不重复的元素。这里用 set 而不是 list，是因为集合的 in 操作是 O(1) 时间复杂度，
# 而列表的 in 操作是 O(n)。

# ==================== 请求追踪器（Request Trackers） ====================
# 【解释】用 request_id 来关联请求和响应，这是一种"关联 ID"（Correlation ID）模式。
# 在分布式系统中很常见，类似 Java 中的 traceId。

shutdown_requests = {}
# 【解释】创建一个空字典（dict），Python 的 dict 类似于 Java 的 HashMap<String, Map<String, Object>>。
# 用于追踪关闭请求的状态：{request_id: {"target": 队友名, "status": "pending/approved/rejected"}}

plan_requests = {}
# 【解释】创建一个空字典，用于追踪计划审批请求的状态：
# {request_id: {"from": 提交者, "plan": 计划文本, "status": "pending/approved/rejected"}}

_tracker_lock = threading.Lock()
# 【解释】创建一个线程锁（互斥锁），用于保护上面两个字典的并发访问。
# 类似于 Java 的 ReentrantLock 或 synchronized 块。
# 在 Python 中使用 with lock: 语法来获取和释放锁，类似 Java 的 try-finally 模式。
# 命名以下划线 _ 开头，表示这是一个"私有"变量（Python 的命名约定，不强制）。


# ==================== MessageBus 类：JSONL 格式的消息收件箱 ====================
# 【解释】这是整个消息系统的核心类，负责在团队成员之间传递消息。
# 消息以 JSONL（JSON Lines）格式存储，每行一个 JSON 对象。
# 类似于 Java 中的一个 MessageQueue 或 MessageBus 服务类。
# Java 中通常会用接口 + 实现类来定义，Python 中直接定义类即可。

class MessageBus:
    # 【解释】定义一个类，类似于 Java 的 class MessageBus { ... }。
    # Python 的类定义使用 class 关键字，没有 public/private 的访问修饰符。
    # 类名使用大驼峰命名法（PascalCase），与 Java 一致。

    def __init__(self, inbox_dir: Path):
    # 【解释】__init__ 是 Python 的构造方法，类似 Java 的构造函数。
    # 但注意：Python 的 __init__ 不是真正创建对象的方法，__new__ 才是。
    # __init__ 接收已创建的对象并进行初始化。self 类似 Java 的 this。
    # inbox_dir: Path 是类型注解（Type Annotation），表示参数期望是 Path 类型。
    # 类型注解在 Python 中是可选的，运行时不强制检查，仅用于 IDE 提示和静态分析。
    # 类似于 Java 的参数类型声明，但 Java 的类型是强制的。

        self.dir = inbox_dir
        # 【解释】self.dir 是实例属性，类似 Java 中 this.dir = inbox_dir。
        # Python 中实例属性不需要提前声明，直接赋值即可创建。
        # 在 Java 中需要在类中先声明 private Path dir; 然后在构造方法中赋值。

        self.dir.mkdir(parents=True, exist_ok=True)
        # 【解释】mkdir 创建目录。
        # parents=True：类似 Java 中 Files.createDirectories()，会创建所有父目录。
        # exist_ok=True：如果目录已存在不报错。类似 Java 中先 !exists() 再 createDirectory()。

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
    # 【解释】定义一个实例方法。在 Java 中等价于 public String send(String sender, String to, ...)。
    # 注意 Python 方法第一个参数始终是 self（类似 Java 的 this），这是 Python 的语法要求。
    # msg_type: str = "message"：带默认值的参数，类似 Java 中没有直接等价物。
    #   Java 中需要用方法重载来模拟默认参数：send(sender, to, content) 和 send(sender, to, content, msgType)。
    # extra: dict = None：参数默认值设为 None（Python 的 null），这是一种常见的安全模式。
    #   因为 Python 的可变默认参数（如 extra: dict = {}）有陷阱——所有调用共享同一个对象！
    #   所以用 None 作为默认值，然后在方法体内判断，这是 Python 的最佳实践。
    # -> str：返回类型注解，表示方法返回 str 类型。类似 Java 的返回类型声明，但非强制。

        if msg_type not in VALID_MSG_TYPES:
        # 【解释】检查消息类型是否在白名单集合中。集合的 in 操作是 O(1)。
        # 类似 Java 中 VALID_MSG_TYPES.contains(msgType)。

            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
            # 【解释】f-string 格式化字符串，{} 中直接嵌入变量。
            # 类似 Java 的 "Error: Invalid type '" + msgType + "'. Valid: " + VALID_MSG_TYPES。

        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        # 【解释】创建一个字典（dict），类似 Java 中 new HashMap<String, Object>() 并 put 多个键值对。
        # dict 是 Python 最常用的数据结构之一，类似 Java 的 HashMap + LinkedHashMap（保持插入顺序）。
        # time.time() 返回当前时间戳（浮点数秒），类似 Java 的 System.currentTimeMillis() / 1000.0。

        if extra:
        # 【解释】Python 的"真值测试"（truthiness）：None、空列表、空字典、0、"" 都被视为 False。
        # 这里 extra 默认是 None，所以 if None 为 False，不会执行。
        # 如果调用者传了非空字典，则为 True。比 Java 的 extra != null 更简洁。

            msg.update(extra)
            # 【解释】dict.update() 方法将另一个字典的键值对合并到当前字典中（覆盖同名的键）。
            # 类似 Java 中 map.putAll(otherMap)。

        inbox_path = self.dir / f"{to}.jsonl"
        # 【解释】用 / 运算符拼接路径，f"{to}.jsonl" 是 f-string，将收件人名字作为文件名。
        # 例如收件人是 "alice"，则路径为 inbox_dir/alice.jsonl。

        with open(inbox_path, "a") as f:
        # 【解释】上下文管理器（Context Manager），是 Python 的关键语法特性。
        # with 语句类似于 Java 7+ 的 try-with-resources：
        #   try (FileWriter f = new FileWriter(path, true)) { ... }
        # "a" 模式表示追加写入（append），类似 Java 中 new FileWriter(path, true)。
        # with 块结束时自动调用 f.close()，即使发生异常也会关闭文件。

            f.write(json.dumps(msg) + "\n")
            # 【解释】json.dumps() 将 Python 对象序列化为 JSON 字符串。
            # 类似 Java 中 new Gson().toJson(msg)。注意 Python 是 dumps（带 s = string），
            # json.dump()（不带 s）则是直接写入文件对象。

        return f"Sent {msg_type} to {to}"
        # 【解释】返回操作结果的描述字符串。

    def read_inbox(self, name: str) -> list:
    # 【解释】读取并清空指定成员的收件箱。
    # 参数 name: str 是类型注解，-> list 是返回类型注解（list 等价于 Java 的 List<Object>）。
    # 注意：Python 的 list 不指定元素类型，类似 Java 的 List<?>（原始类型）。

        inbox_path = self.dir / f"{name}.jsonl"
        # 【解释】构建收件箱文件路径。

        if not inbox_path.exists():
        # 【解释】检查文件是否存在。类似 Java 中 !Files.exists(path)。

            return []
            # 【解释】返回空列表。Python 中 [] 是空列表字面量，类似 Java 的 List.of()。

        messages = []
        # 【解释】创建一个空列表，类似 Java 的 new ArrayList<>()。

        for line in inbox_path.read_text().strip().splitlines():
        # 【解释】这是一个方法链式调用（method chaining）：
        #   read_text()：读取文件全部内容为字符串，类似 Java 中 Files.readString(path)。
        #   .strip()：去除首尾空白字符（包括换行符），类似 Java 中 String.trim()。
        #   .splitlines()：按行分割字符串，返回列表，类似 Java 中 String.split("\\n")。
        # for ... in ...：Python 的 for-each 循环，类似 Java 的 for (String line : lines)。

            if line:
            # 【解释】Python 的真值测试：空字符串 "" 为 False，非空字符串为 True。
            # 用于跳过空行。比 Java 的 !line.isEmpty() 更简洁。

                messages.append(json.loads(line))
                # 【解释】json.loads() 将 JSON 字符串反序列化为 Python 对象（dict/list）。
                # 类似 Java 中 new Gson().fromJson(line, Object.class)。
                # list.append() 方法在末尾添加元素，类似 Java 中 list.add()。

        inbox_path.write_text("")
        # 【解释】将文件内容清空（写入空字符串）。
        # 这实现了"读取后清空"的语义，类似 Java 中"消费消息队列"的概念。

        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
    # 【解释】广播消息：向除发送者之外的所有队友发送消息。
    # 参数 teammates: list 是一个字符串列表，类似 Java 的 List<String>。

        count = 0
        # 【解释】计数器，记录发送消息的数量。

        for name in teammates:
        # 【解释】遍历队友列表。类似 Java 的 for (String name : teammates)。

            if name != sender:
            # 【解释】跳过发送者自己，不给自己发广播。

                self.send(sender, name, content, "broadcast")
                # 【解释】调用自身的 send 方法，消息类型固定为 "broadcast"。

                count += 1
                # 【解释】Python 没有count++ 运算符！必须写成 count += 1 或 count = count + 1。
                # 这是 Python 与 Java/JavaScript 的一个常见差异。

        return f"Broadcast to {count} teammates"


# ==================== 创建全局消息总线实例 ====================
BUS = MessageBus(INBOX_DIR)
# 【解释】创建 MessageBus 的全局单例实例，命名为 BUS（全大写表示常量）。
# 在 Java 中通常会用单例模式（Singleton Pattern）或依赖注入（DI）来管理。
# Python 中直接在模块级别创建全局变量即可，简单直接。


# ==================== TeammateManager 类：队友管理器 ====================
# 【解释】管理团队成员的生命周期：创建（spawn）、配置持久化、线程管理。
# 包含关闭协议和计划审批协议的处理逻辑。
# 类似于 Java 中的一个 TeamService 或 TeamManager 服务类。

class TeammateManager:

    def __init__(self, team_dir: Path):
    # 【解释】构造方法，接收团队目录路径。

        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        # 【解释】创建团队目录（如果不存在）。

        self.config_path = self.dir / "config.json"
        # 【解释】配置文件路径：.team/config.json，用于持久化团队成员信息。

        self.config = self._load_config()
        # 【解释】调用私有方法 _load_config() 加载配置。
        # Python 中方法名以单下划线 _ 开头表示"内部使用"（类似 Java 的 private 约定）。
        # 以双下划线 __ 开头则触发名称改写（name mangling），更严格的"私有"。

        self.threads = {}
        # 【解释】创建空字典，用于存储每个队友的线程对象。
        # 结构：{队友名: threading.Thread 对象}。

    def _load_config(self) -> dict:
    # 【解释】加载配置文件，返回字典。私有方法（下划线开头）。
    # -> dict：返回类型注解，表示返回一个字典。

        if self.config_path.exists():
        # 【解释】如果配置文件存在，则读取并解析 JSON。
            return json.loads(self.config_path.read_text())
            # 【解释】read_text() 读取文件全部内容为字符串，json.loads() 解析为字典。
            # 类似 Java 中 new Gson().fromJson(Files.readString(path), Map.class)。

        return {"team_name": "default", "members": []}
        # 【解释】如果配置文件不存在，返回默认配置。
        # {"team_name": "default", "members": []} 是一个字典字面量，
        # members 是一个空列表，类似 Java 中 new HashMap(){{ put("team_name","default"); put("members", new ArrayList<>()); }}

    def _save_config(self):
    # 【解释】将当前配置保存到文件。私有方法。

        self.config_path.write_text(json.dumps(self.config, indent=2))
        # 【解释】json.dumps(self.config, indent=2) 将字典序列化为格式化的 JSON 字符串（缩进 2 空格）。
        # 类似 Java 中 new GsonBuilder().setPrettyPrinting().create().toJson(config)。
        # write_text() 将字符串写入文件，类似 Java 中 Files.writeString(path, content)。

    def _find_member(self, name: str) -> dict:
    # 【解释】在成员列表中查找指定名称的成员，返回成员字典或 None。
    # 类似 Java 中 Optional<Member> findByName(String name)。

        for m in self.config["members"]:
        # 【解释】遍历成员列表。self.config["members"] 是一个字典的列表（List<Map>）。
        # Python 用 [] 从字典中取值，类似 Java 中 map.get("members")。

            if m["name"] == name:
            # 【解释】比较成员名是否匹配。Python 中 == 用于值比较（类似 Java 的 .equals()）。
            # 注意：Python 的 == 不会比较对象引用，而是比较值。Java 中字符串比较需要用 .equals()。

                return m
                # 【解释】找到后直接返回该成员字典。

        return None
        # 【解释】遍历完没找到，返回 None（Python 的 null）。

    def spawn(self, name: str, role: str, prompt: str) -> str:
    # 【解释】创建（或重启）一个队友。这是 TeammateManager 的核心公共方法。
    # name: 队友名称，role: 角色描述，prompt: 初始提示词。
    # 返回操作结果字符串。

        member = self._find_member(name)
        # 【解释】先查找是否已存在同名成员。

        if member:
        # 【解释】如果成员已存在...

            if member["status"] not in ("idle", "shutdown"):
            # 【解释】检查成员状态。not in 用于检查元素是否不在元组中。
            # ("idle", "shutdown") 是一个元组（tuple），用圆括号表示，类似 Java 的 List.of("idle", "shutdown")。
            # tuple 类似于 Java 的不可变 List，一旦创建就不能修改。

                return f"Error: '{name}' is currently {member['status']}"
                # 【解释】如果成员正在工作，返回错误信息。

            member["status"] = "working"
            # 【解释】更新成员状态为"工作中"。字典的值可以直接修改。

            member["role"] = role
            # 【解释】更新角色。
        else:
        # 【解释】如果成员不存在，创建新成员。

            member = {"name": name, "role": role, "status": "working"}
            # 【解释】创建成员字典，包含名称、角色和状态。

            self.config["members"].append(member)
            # 【解释】将新成员添加到配置的成员列表中。
            # list.append() 类似 Java 中 list.add()。

        self._save_config()
        # 【解释】保存配置到文件，确保持久化。

        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        # 【解释】创建一个新线程来运行队友的主循环。
        # threading.Thread 类似 Java 的 new Thread(runnable)。
        # target：指定线程要执行的函数（类似 Java 的 Runnable）。
        # args：传递给 target 函数的位置参数，必须是元组。
        #   (name, role, prompt) 是一个元组（tuple），作为参数传递给 _teammate_loop。
        # daemon=True：设置为守护线程，类似 Java 的 thread.setDaemon(true)。
        #   守护线程会在主线程结束时自动终止，不会阻止程序退出。

        self.threads[name] = thread
        # 【解释】将线程对象保存到字典中，方便后续管理。

        thread.start()
        # 【解释】启动线程。类似 Java 的 thread.start()。
        # 注意：Python 中启动线程不会自动运行 target 函数，必须显式调用 start()。

        return f"Spawned '{name}' (role: {role})"
        # 【解释】返回成功信息。

    def _teammate_loop(self, name: str, role: str, prompt: str):
    # 【解释】队友的主循环方法，在独立线程中运行。这是整个队友 Agent 的核心逻辑。
    # 每个队友是一个持续运行的循环：检查收件箱 -> 调用 API -> 处理工具调用 -> 重复。
    # 类似于 Java 中的一个 while(true) 循环 + Runnable 的 run() 方法。

        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        # 【解释】用圆括号 () 将多个 f-string 拼接成一个长字符串。
        # Python 中相邻的字符串字面量会自动拼接（不需要 + 运算符），圆括号允许跨行。
        # 类似 Java 中多行字符串拼接：String s = "line1 " + "line2 " + "line3";
        # 或者 Java 15+ 的文本块（Text Block）："""..."""。

        messages = [{"role": "user", "content": prompt}]
        # 【解释】创建消息历史列表，初始包含用户提示词。
        # 列表中有一个字典，类似 Java 中 List.of(Map.of("role", "user", "content", prompt))。
        # 这是 Anthropic API 要求的消息格式，每条消息包含 role 和 content。

        tools = self._teammate_tools()
        # 【解释】获取队友可用的工具定义列表。

        should_exit = False
        # 【解释】退出标志，用于控制循环终止（收到关闭批准后退出）。

        for _ in range(50):
        # 【解释】最多循环 50 次，防止无限循环。
        # _ 是 Python 的惯用法，表示"循环变量不使用"（类似 Java 中 for (int i = 0; ...) 但不用 i）。
        # range(50) 生成 0 到 49 的整数序列，类似 Java 的 IntStream.range(0, 50)。

            inbox = BUS.read_inbox(name)
            # 【解释】读取并清空队友的收件箱，获取所有新消息。

            for msg in inbox:
            # 【解释】遍历收件箱中的每条消息。

                messages.append({"role": "user", "content": json.dumps(msg)})
                # 【解释】将消息作为 user 消息追加到对话历史中。
                # json.dumps(msg) 将消息字典转为 JSON 字符串。

            if should_exit:
            # 【解释】如果上一轮已经收到关闭批准，退出循环。

                break
                # 【解释】break 跳出 for 循环，类似 Java 的 break。

            try:
            # 【解释】try-except 是 Python 的异常处理机制，类似 Java 的 try-catch。

                response = client.messages.create(
                # 【解释】调用 Anthropic API 创建消息。这是一个 SDK 调用。
                # 类似 Java 中 client.messages().create(request)。

                    model=MODEL,
                    # 【解释】使用哪个 AI 模型。

                    system=sys_prompt,
                    # 【解释】系统提示词，定义 AI 的角色和行为。

                    messages=messages,
                    # 【解释】对话历史，包含之前的消息。

                    tools=tools,
                    # 【解释】可用的工具定义列表。

                    max_tokens=8000,
                    # 【解释】最大生成 token 数，类似 Java 中设置 maxTokens 参数。
                )
            except Exception:
            # 【解释】捕获所有异常。except Exception 类似 Java 的 catch (Exception e)。
            # 注意：Python 可以只写 except Exception: 不绑定变量，Java 中必须写 catch (Exception e)。

                break
                # 【解释】API 调用失败，跳出循环结束队友。

            messages.append({"role": "assistant", "content": response.content})
            # 【解释】将 AI 的响应追加到消息历史中。response.content 是内容块列表。

            if response.stop_reason != "tool_use":
            # 【解释】stop_reason 表示响应停止的原因。
            # "tool_use" 表示 AI 想要调用工具；其他值（如 "end_turn"）表示 AI 不想继续了。

                break
                # 【解释】AI 没有调用工具，循环结束。

            results = []
            # 【解释】创建空列表，用于收集工具调用的结果。

            for block in response.content:
            # 【解释】遍历响应的内容块。response.content 是一个列表，每个块可能是文本或工具调用。

                if block.type == "tool_use":
                # 【解释】检查是否是工具调用块。block.type 是属性访问（类似 Java 的 getType()）。

                    output = self._exec(name, block.name, block.input)
                    # 【解释】执行工具调用。_exec 是内部方法，负责分发和处理各种工具。

                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    # 【解释】打印工具调用信息。
                    # str(output) 将对象转为字符串，类似 Java 的 String.valueOf(output) 或 toString()。
                    # [:120] 是切片（slice）语法，取前 120 个字符。
                    # Python 的切片非常灵活：str[start:end:step]，类似 Java 中 str.substring(0, 120)。

                    results.append({
                        # 【解释】构建工具结果，追加到结果列表。

                        "type": "tool_result",
                        # 【解释】结果类型标识。

                        "tool_use_id": block.id,
                        # 【解释】关联到具体的工具调用请求（类似 Java 中的 requestId）。

                        "content": str(output),
                        # 【解释】工具的返回结果，转为字符串。
                    })

                    if block.name == "shutdown_response" and block.input.get("approve"):
                    # 【解释】检查队友是否批准了关闭请求。
                    # block.input 是一个字典，block.input.get("approve") 安全地获取值（不存在返回 None）。
                    # dict.get(key) 类似 Java 中 map.get(key)（不存在的 key 返回 null 而不抛异常）。

                        should_exit = True
                        # 【解释】设置退出标志，下一轮循环将退出。

            messages.append({"role": "user", "content": results})
            # 【解释】将工具结果作为 user 消息追加到历史中（Anthropic API 的要求格式）。

        # 【解释】循环结束后的清理工作。

        member = self._find_member(name)
        # 【解释】查找成员记录。

        if member:
        # 【解释】如果成员存在（理论上应该存在）...

            member["status"] = "shutdown" if should_exit else "idle"
            # 【解释】Python 的三元表达式（条件表达式）：
            #   值1 if 条件 else 值2
            # 类似 Java 的三元运算符：should_exit ? "shutdown" : "idle"
            # 根据退出原因更新成员状态。

            self._save_config()
            # 【解释】保存配置，确保持久化。

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
    # 【解释】工具调度方法（Tool Dispatcher），根据工具名称分发到对应的处理函数。
    # sender: 调用工具的队友名，tool_name: 工具名称，args: 工具参数字典。
    # 类似于 Java 中的 Map<String, Function> 或 Strategy 模式的调度器。

        # these base tools are unchanged from s02
        # 【解释】以下基础工具与 s02 脚本相同，没有变化。

        if tool_name == "bash":
        # 【解释】if-elif 是 Python 的条件分支，类似 Java 的 if-else if。
        # 但 Python 用 elif 而不是 else if（注意缩进！）。

            return _run_bash(args["command"])
            # 【解释】args["command"] 从参数字典中获取值。
            # 注意：如果 key 不存在会抛出 KeyError。更安全的方式是 args.get("command")。

        if tool_name == "read_file":
            return _run_read(args["path"])

        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])

        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])

        if tool_name == "send_message":
        # 【解释】发送消息给其他队友。

            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
            # 【解释】args.get("msg_type", "message") 安全获取 msg_type，默认值为 "message"。
            # 类似 Java 中 map.getOrDefault("msgType", "message")。

        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
            # 【解释】读取收件箱并格式化为 JSON 字符串。indent=2 表示缩进 2 空格。

        if tool_name == "shutdown_response":
        # 【解释】关闭协议：队友响应 Lead 的关闭请求。

            req_id = args["request_id"]
            # 【解释】获取关联 ID。

            approve = args["approve"]
            # 【解释】获取是否批准（布尔值）。

            with _tracker_lock:
            # 【解释】上下文管理器用于线程锁。
            # with _tracker_lock: 等价于 Java 中：
            #   _tracker_lock.lock();
            #   try { ... } finally { _tracker_lock.unlock(); }
            # 这是 Python 中使用锁的推荐方式，确保异常时也能释放锁。

                if req_id in shutdown_requests:
                # 【解释】检查请求 ID 是否存在于关闭请求追踪器中。
                # dict 的 in 操作是 O(1)，类似 Java 中 map.containsKey(key)。

                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
                    # 【解释】更新请求状态。三元表达式：如果 approve 为 True 则 "approved"，否则 "rejected"。

            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            # 【解释】向 Lead 发送关闭响应消息。
            # args.get("reason", "") 获取关闭原因，默认为空字符串。

            return f"Shutdown {'approved' if approve else 'rejected'}"

        if tool_name == "plan_approval":
        # 【解释】计划审批协议：队友向 Lead 提交计划等待审批。

            plan_text = args.get("plan", "")
            # 【解释】获取计划文本，默认为空字符串。

            req_id = str(uuid.uuid4())[:8]
            # 【解释】生成一个 8 字符的短 UUID 作为请求 ID。
            # uuid.uuid4() 生成随机 UUID，str() 转为字符串（如 "a1b2c3d4-e5f6-..."）。
            # [:8] 切片取前 8 个字符（去掉连字符后的前 8 位）。
            # 类似 Java 中 UUID.randomUUID().toString().replace("-","").substring(0, 8)。

            with _tracker_lock:
            # 【解释】加锁保护共享字典。

                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
                # 【解释】在追踪器中注册新请求，状态为 "pending"（等待审批）。

            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            # 【解释】向 Lead 发送计划审批请求。

            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

        return f"Unknown tool: {tool_name}"
        # 【解释】未知工具，返回错误信息。这是默认的"兜底"处理。

    def _teammate_tools(self) -> list:
    # 【解释】定义队友可用的工具列表。返回一个字典列表（List<Map<String, Object>>）。
    # 每个字典描述一个工具的名称、描述和输入参数 schema。
    # 这遵循 Anthropic API 的 tool 定义格式。

        # these base tools are unchanged from s02
        # 【解释】以下基础工具定义与 s02 脚本相同。

        return [
            # 【解释】Python 的列表字面量用方括号 []，类似 Java 中 List.of(...) 或 Arrays.asList(...)。

            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            # 【解释】bash 工具定义。字典的值可以嵌套字典。
            # input_schema 定义了工具的参数结构，类似 Java 中的 JSON Schema 或 OpenAPI 规范。

            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            # 【解释】read_file 工具，读取文件内容。

            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            # 【解释】write_file 工具，写入文件。

            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            # 【解释】edit_file 工具，替换文件中的文本。

            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            # 【解释】send_message 工具。
            # list(VALID_MSG_TYPES) 将集合转为列表，因为 JSON Schema 的 enum 需要数组格式。
            # 类似 Java 中 new ArrayList<>(VALID_MSG_TYPES)（Set 转 List）。

            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
            # 【解释】read_inbox 工具，无参数。

            {"name": "shutdown_response", "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            # 【解释】shutdown_response 工具，关闭协议的核心。
            # approve 参数是 boolean 类型，对应 Python 的 True/False。

            {"name": "plan_approval", "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            # 【解释】plan_approval 工具，计划审批协议的核心。
        ]

    def list_all(self) -> str:
    # 【解释】列出所有团队成员的信息，返回格式化的字符串。

        if not self.config["members"]:
        # 【解释】如果成员列表为空（空列表 [] 为 False）。
        # 类似 Java 中 if (config.getMembers().isEmpty())。

            return "No teammates."
            # 【解释】返回提示信息。

        lines = [f"Team: {self.config['team_name']}"]
        # 【解释】创建包含团队名称的列表。列表的第一个元素是标题行。

        for m in self.config["members"]:
        # 【解释】遍历所有成员。

            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
            # 【解释】将成员信息格式化为字符串，追加到列表中。
            # f-string 中可以嵌入多个表达式。类似 Java 中的 String.format()。

        return "\n".join(lines)
        # 【解释】str.join(iterable) 是 Python 中非常常用的方法：用指定分隔符连接列表中的字符串。
        # "\n".join(lines) 将列表中的字符串用换行符连接起来。
        # 注意：这是字符串的方法，不是列表的方法！类似 Java 中 String.join("\n", lines)。

    def member_names(self) -> list:
    # 【解释】获取所有成员的名称列表。

        return [m["name"] for m in self.config["members"]]
        # 【解释】这是 Python 的"列表推导式"（List Comprehension），是一种非常强大的语法。
        # [表达式 for 变量 in 可迭代对象] 会生成一个新列表。
        # 等价于 Java 中的 stream().map().toList() 或 for 循环收集：
        #   List<String> names = new ArrayList<>();
        #   for (Map<String, String> m : config.getMembers()) { names.add(m.get("name")); }
        #   return names;
        # 列表推导式更简洁、更 Pythonic。


# ==================== 创建全局队友管理器实例 ====================
TEAM = TeammateManager(TEAM_DIR)
# 【解释】创建 TeammateManager 的全局单例实例，命名为 TEAM。
# Python 中直接在模块级别创建对象，所有 import 此模块的代码共享同一个实例。
# 类似 Java 中 Spring 的 @Singleton Bean，但更简单直接。


# ==================== 基础工具实现函数 ====================
# 【解释】这些是独立函数（不属于任何类），实现具体的工具功能。
# 在 Java 中通常会用一个 ToolService 类的静态方法来组织。
# Python 中模块级别的函数就是"顶层函数"，不需要类也能存在。
# 函数名以 _ 开头表示"私有"（模块内部使用）。

def _safe_path(p: str) -> Path:
# 【解释】安全路径解析函数，防止路径遍历攻击（Path Traversal）。
# 参数 p: str 是用户提供的路径，返回值是安全解析后的 Path 对象。
# 类似 Java 中对用户输入进行 sanitize 的工具方法。

    path = (WORKDIR / p).resolve()
    # 【解释】resolve() 解析路径中的所有符号链接和相对引用（如 ".."、"."），返回绝对路径。
    # 类似 Java 中 path.toAbsolutePath().normalize()。

    if not path.is_relative_to(WORKDIR):
    # 【解释】检查解析后的路径是否在工作目录范围内。
    # is_relative_to() 判断当前路径是否以指定路径为前缀。
    # 如果用户传入 "../../../etc/passwd"，解析后会指向系统目录，不包含在 WORKDIR 中。

        raise ValueError(f"Path escapes workspace: {p}")
        # 【解释】raise 抛出异常，类似 Java 的 throw new IllegalArgumentException("...")。
        # ValueError 是 Python 内置异常类型，类似 Java 的 IllegalArgumentException。

    return path


def _run_bash(command: str) -> str:
# 【解释】执行 shell 命令并返回输出。包含危险命令检查。

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    # 【解释】危险命令黑名单，是一个字符串列表（类似 Java 的 List<String>）。

    if any(d in command for d in dangerous):
    # 【解释】any() 是 Python 内置函数：如果可迭代对象中有任何元素为 True，则返回 True。
    # 这里的 "for d in dangerous" 是生成器表达式（Generator Expression），
    # 类似 Java 中 Stream.anyMatch()：dangerous.stream().anyMatch(d -> command.contains(d))。
    # any() 相当于 Java Stream 的 anyMatch，all() 相当于 allMatch，none() 在 Python 中用 not any()。

        return "Error: Dangerous command blocked"
        # 【解释】拦截危险命令，返回错误信息。

    try:
    # 【解释】异常处理开始。

        r = subprocess.run(
        # 【解释】subprocess.run() 执行 shell 命令，类似 Java 中 ProcessBuilder.start().waitFor()。

            command, shell=True, cwd=WORKDIR,
            # 【解释】shell=True 表示通过 shell 执行命令（如 bash -c "command"）。
            # cwd=WORKDIR 设置工作目录。类似 Java 中 ProcessBuilder.directory(workDir)。

            capture_output=True, text=True, timeout=120,
            # 【解释】capture_output=True：捕获标准输出和标准错误。
            # 类似 Java 中 process.getInputStream() 和 process.getErrorStream()。
            # text=True：以文本模式（而非字节模式）返回输出。
            # timeout=120：超时时间 120 秒。类似 Java 中 Process.waitFor(120, TimeUnit.SECONDS)。
        )
        # 注意：subprocess.run() 的参数都是关键字参数（keyword arguments），类似 Java 中 Builder 模式。

        out = (r.stdout + r.stderr).strip()
        # 【解释】r.stdout 是标准输出，r.stderr 是标准错误。都是字符串（因为 text=True）。
        # + 用于字符串拼接（但效率不如 f-string 或 join），.strip() 去除首尾空白。

        return out[:50000] if out else "(no output)"
        # 【解释】三元表达式：如果有输出则截取前 50000 个字符，否则返回 "(no output)"。
        # Python 的字符串切片越界不会报错，如果 out 长度不足 50000，out[:50000] 返回整个字符串。
        # 类似 Java 中需要 Math.min(out.length(), 50000) 来避免 StringIndexOutOfBoundsException。

    except subprocess.TimeoutExpired:
    # 【解释】捕获特定的超时异常，而不是笼统的 Exception。
    # 类似 Java 中 catch (TimeoutException e) 或 catch (ExecutionException e)。

        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
# 【解释】读取文件内容。limit 参数可选，用于限制读取的行数。
# limit: int = None：默认值为 None，表示不限制。注意用 None 而不是 0（0 也是假值但有不同含义）。

    try:
        lines = _safe_path(path).read_text().splitlines()
        # 【解释】链式调用：_safe_path(path) 安全解析路径 -> read_text() 读取全文 -> splitlines() 按行分割。
        # splitlines() 返回一个列表，类似 Java 中 String.split("\\r?\\n")。

        if limit and limit < len(lines):
        # 【解释】如果指定了 limit 且小于总行数...
        # limit and ...：短路求值。如果 limit 是 None（默认值），则 limit 为 False，不会继续判断。
        # 这是 Python 中常见的"有值才检查"的简洁写法。

            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
            # 【解释】只取前 limit 行，然后在末尾追加一个提示行。
            # lines[:limit] 是切片，取前 limit 个元素。
            # + 运算符连接两个列表，类似 Java 中 list1.addAll(list2)（但创建新列表）。
            # f"... ({len(lines) - limit} more)" 提示还有多少行被省略。

        return "\n".join(lines)[:50000]
        # 【解释】将行列表用换行符连接，截取前 50000 个字符。

    except Exception as e:
    # 【解释】捕获所有异常，将异常对象绑定到变量 e。
    # 类似 Java 中 catch (Exception e)。

        return f"Error: {e}"
        # 【解释】f-string 中 {e} 会调用 e 的 __str__() 方法，类似 Java 中 e.getMessage() 或 e.toString()。


def _run_write(path: str, content: str) -> str:
# 【解释】将内容写入文件。

    try:
        fp = _safe_path(path)
        # 【解释】安全解析路径。

        fp.parent.mkdir(parents=True, exist_ok=True)
        # 【解释】fp.parent 获取文件的父目录（Path 对象）。
        # mkdir(parents=True, exist_ok=True)：创建所有不存在的父目录。
        # 类似 Java 中 Files.createDirectories(fp.getParent())。

        fp.write_text(content)
        # 【解释】将字符串内容写入文件。类似 Java 中 Files.writeString(fp, content)。

        return f"Wrote {len(content)} bytes"
        # 【解释】len(content) 返回字符串长度（字节数取决于编码，这里 write_text 默认用 UTF-8）。
        # 类似 Java 中 content.getBytes(StandardCharsets.UTF_8).length。

    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
# 【解释】编辑文件：将文件中的 old_text 替换为 new_text（只替换第一个匹配项）。

    try:
        fp = _safe_path(path)
        # 【解释】安全解析路径。

        c = fp.read_text()
        # 【解释】读取文件全部内容。

        if old_text not in c:
        # 【解释】检查 old_text 是否存在于文件内容中。
        # Python 的 in 运算符用于字符串包含检查，类似 Java 中 c.contains(oldText)。

            return f"Error: Text not found in {path}"
            # 【解释】如果找不到要替换的文本，返回错误。

        fp.write_text(c.replace(old_text, new_text, 1))
        # 【解释】str.replace(old, new, count) 方法替换字符串。
        # 第三个参数 1 表示只替换第一次出现的位置。
        # 类似 Java 中没有直接等价的方法，需要用 replaceFirst(oldText, newText)，
        # 但 Java 的 replaceFirst 使用正则表达式，而 Python 的 replace 使用纯文本。

        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


# ==================== Lead 专用的协议处理函数 ====================
# 【解释】这些函数处理 Lead（团队领导）角色的协议逻辑。

def handle_shutdown_request(teammate: str) -> str:
# 【解释】处理关闭请求：Lead 向指定队友发送关闭请求。
# 这是关闭协议的第一步（FSM: pending）。
# 类似 Java 中的一个 Service 方法。

    req_id = str(uuid.uuid4())[:8]
    # 【解释】生成 8 字符的短 UUID 作为关联 ID。

    with _tracker_lock:
    # 【解释】获取线程锁，保护共享字典。

        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
        # 【解释】在追踪器中注册关闭请求，初始状态为 "pending"。

    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    # 【解释】通过消息总线向队友发送关闭请求消息。

    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"
    # 【解释】返回操作结果，包含请求 ID 和当前状态。


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
# 【解释】处理计划审批：Lead 审批队友提交的计划。
# approve: bool 是 Python 的布尔类型（True/False），对应 Java 的 boolean。
# feedback: str = "" 带默认值的参数，类似 Java 中需要重载方法来模拟。

    with _tracker_lock:
        req = plan_requests.get(request_id)
        # 【解释】加锁后获取计划请求。dict.get(key) 不存在时返回 None（不抛异常）。

    if not req:
    # 【解释】如果请求不存在（req 是 None），返回错误。

        return f"Error: Unknown plan request_id '{request_id}'"

    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
        # 【解释】加锁后更新请求状态。
        # 注意：这里两次加锁而不是一次，是因为第一次需要读取判断，第二次需要修改。
        # 类似 Java 中对 HashMap 的读-判断-写操作，如果不加锁可能有竞态条件。

    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    # 【解释】向提交计划的队友发送审批结果。

    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
# 【解释】查询关闭请求的状态。Lead 用此工具查看队友是否已响应关闭请求。

    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))
        # 【解释】dict.get(key, default) 如果 key 不存在则返回默认值。
        # 类似 Java 中 map.getOrDefault(key, defaultValue)。


# ==================== Lead 工具调度表（Tool Dispatch Table） ====================
# 【解释】使用字典 + lambda 表达式实现工具调度。
# 这是 Python 中一种简洁的"策略模式"实现方式，比 Java 中需要定义接口 + 多个实现类更简洁。
# Java 对比：
#   Map<String, Function<Map, String>> handlers = new HashMap<>();
#   handlers.put("bash", kw -> runBash(kw.get("command")));
#   ...

TOOL_HANDLERS = {
    # 【解释】字典的值全部是 lambda 表达式（匿名函数）。

    "bash":              lambda **kw: _run_bash(kw["command"]),
    # 【解释】lambda 是 Python 的匿名函数（类似 Java 的 lambda 表达式）。
    # lambda **kw: ... 中的 **kw 表示接收任意数量的关键字参数，打包成一个字典。
    # 类似 Java 中 (Map<String, Object> kw) -> runBash((String) kw.get("command"))。
    # ** 是 Python 的可变关键字参数语法，详见下方补充说明。
    #
    # 【补充说明 - Python 可变参数语法】：
    #   def f(*args)：*args 收集所有位置参数为元组，类似 Java 的 varargs String... args
    #   def f(**kwargs)：**kwargs 收集所有关键字参数为字典，类似 Java 中接收一个 Map<String, Object>
    #   lambda **kw：同 **kwargs 的语法用于匿名函数

    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    # 【解释】kw.get("limit") 可能返回 None，_run_read 的 limit 参数默认就是 None，所以这里安全。

    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),

    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    # 【解释】调用 TEAM 全局实例的 spawn 方法来创建队友。

    "list_teammates":    lambda **kw: TEAM.list_all(),
    # 【解释】这个 lambda 忽略了 kw 参数（因为 list_teammates 工具不需要参数）。
    # Python 中 lambda 可以接收不使用的参数，不会有编译错误（不像 Java 的 lambda 严格匹配参数）。

    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),

    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    # 【解释】读取 Lead 的收件箱。

    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    # 【解释】向所有队友广播消息。

    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    # 【解释】调用关闭协议处理函数。

    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    # 【解释】Lead 端的 shutdown_response 工具是"查询状态"，而不是"响应请求"。
    # 注意区分：队友端的 shutdown_response 是"响应关闭请求"，Lead 端是"查看响应状态"。
    # 同名工具在不同角色有不同语义，这是一种有趣的设计。

    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    # 【解释】Lead 端的 plan_approval 工具是"审批计划"。
    # 注意区分：队友端的 plan_approval 是"提交计划"，Lead 端是"审批计划"。
    # 同样是同名工具，不同角色不同语义。
}
# 【解释】这个字典实现了 12 个工具的调度。
# 在 agent_loop 中通过 TOOL_HANDLERS.get(tool_name) 来查找并调用对应的处理函数。
# 这种模式比 if-elif 链更清晰，也更易于扩展（只需添加新的键值对）。

# ==================== Lead 工具定义列表 ====================
# 【解释】Lead（团队领导）可以使用的 12 个工具的定义。
# 这些定义遵循 Anthropic API 的工具格式规范（Tool Use）。
# 每个工具包含：name（名称）、description（描述）、input_schema（JSON Schema 格式的参数定义）。

# these base tools are unchanged from s02
# 【解释】以下基础工具定义与 s02 脚本相同。

TOOLS = [
# 【解释】TOOLS 是一个列表常量（全大写），包含 12 个工具定义字典。

    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    # 【解释】bash 工具：执行 shell 命令。required 字段指定必填参数列表。

    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    # 【解释】read_file 工具：读取文件。limit 是可选参数（不在 required 列表中），用于限制行数。
    # 注意：队友版的 read_file 没有 limit 参数，Lead 版有，这是因为 Lead 需要更精细的控制。

    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},

    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    {"name": "spawn_teammate", "description": "Spawn a persistent teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    # 【解释】spawn_teammate 工具：创建一个队友 Agent。这是 Lead 独有的工具，队友不能创建队友。
    # 三个参数都是必填的：名称、角色描述、初始提示词。

    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    # 【解释】list_teammates 工具：列出所有队友。无参数（properties 为空）。

    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},

    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    # 【解释】读取 Lead 自己的收件箱。

    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    # 【解释】broadcast 工具：向所有队友广播消息。Lead 独有。

    {"name": "shutdown_request", "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    # 【解释】shutdown_request 工具：关闭协议 - Lead 发起关闭请求。
    # Lead 独有的工具，用于请求队友关闭。返回 request_id 用于后续查询状态。

    {"name": "shutdown_response", "description": "Check the status of a shutdown request by request_id.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    # 【解释】shutdown_response 工具（Lead 版）：查询关闭请求的状态。
    # 注意与队友版的同名工具区别：队友版是"响应关闭请求"，Lead 版是"查看响应状态"。

    {"name": "plan_approval", "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    # 【解释】plan_approval 工具（Lead 版）：审批队友提交的计划。
    # approve 是布尔类型，feedback 是可选的反馈文本。
    # 同样注意与队友版的区别：队友版是"提交计划"，Lead 版是"审批计划"。
]
# 【解释】这个列表在 agent_loop 中传给 Anthropic API，让 AI 知道它可以调用哪些工具。


# ==================== Lead 的 Agent 主循环 ====================
def agent_loop(messages: list):
# 【解释】Lead Agent 的主循环函数，与队友的 _teammate_loop 结构类似但更简单。
# 参数 messages: list 是对话历史（引用类型，函数内修改会影响外部）。
# 注意：Python 的列表是可变对象，作为参数传递时是"引用传递"（类似 Java 中传 List 引用），
# 函数内对列表的修改（如 append）会影响调用者。但重新赋值（messages = [...]）不会影响。

    while True:
    # 【解释】无限循环，直到 AI 不再调用工具时通过 return 退出。
    # 类似 Java 中 while (true) { ... }

        inbox = BUS.read_inbox("lead")
        # 【解释】读取 Lead 的收件箱。注意：read_inbox 会清空收件箱（消费消息）。

        if inbox:
        # 【解释】如果收件箱有新消息（非空列表为 True）。

            messages.append({
            # 【解释】将收件箱内容包装成一条 user 消息追加到对话历史。
            # 使用 <inbox> 标签包裹，帮助 AI 区分用户输入和系统消息。

                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
                # 【解释】f-string 嵌套：外层是 f-string，内部 json.dumps() 生成 JSON 字符串。
                # indent=2 使 JSON 格式化输出，方便 AI 阅读理解。
            })

        response = client.messages.create(
        # 【解释】调用 Anthropic API，发送消息并获取 AI 响应。

            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        messages.append({"role": "assistant", "content": response.content})
        # 【解释】将 AI 的响应追加到历史。response.content 是一个列表（ContentBlock 数组）。

        if response.stop_reason != "tool_use":
        # 【解释】如果 AI 不想调用工具（即"思考完毕，给出最终回答"），退出循环。

            return
            # 【解释】return 无返回值，函数返回 None（类似 Java 中 return; 或 void 方法的 return）。

        results = []
        # 【解释】用于收集本轮所有工具调用的结果。

        for block in response.content:
        # 【解释】遍历响应的所有内容块（可能包含文本块和工具调用块）。

            if block.type == "tool_use":
            # 【解释】只处理工具调用块。

                handler = TOOL_HANDLERS.get(block.name)
                # 【解释】从调度表中查找工具处理函数。
                # dict.get(key) 不存在返回 None，不会抛异常。

                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    # 【解释】三元表达式：如果找到了 handler，调用它并传入参数；否则返回错误。
                    # **block.input：将字典解包为关键字参数传递。
                    # 例如 block.input = {"command": "ls"}，则 **block.input 等价于 handler(command="ls")。
                    # 这是 Python 的字典解包语法，类似 Java 中没有直接对应的概念，
                    # 但可以理解为将 Map 的内容展开为方法参数。
                    # 等价 Java 写法（伪代码）：handler.invokeAll(block.input)

                except Exception as e:
                # 【解释】捕获工具执行过程中的异常。

                    output = f"Error: {e}"

                print(f"> {block.name}:")
                print(str(output)[:200])
                # 【解释】打印工具调用信息到控制台。

                results.append({
                # 【解释】构建工具结果块，追加到结果列表。

                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })

        messages.append({"role": "user", "content": results})
        # 【解释】将工具结果作为 user 消息追加到历史（Anthropic API 要求的格式）。
        # 注意：content 可以是字符串或列表（这里的 results 是列表）。


# ==================== 程序入口 ====================
if __name__ == "__main__":
# 【解释】这是 Python 的程序入口写法。
# __name__ 是 Python 内置变量：当文件被直接运行时，__name__ 的值是 "__main__"；
# 当文件被 import 到其他模块时，__name__ 的值是模块名。
# 所以 if __name__ == "__main__": 中的代码只在直接运行时执行，被 import 时不会执行。
# 类似 Java 中的 public static void main(String[] args) 方法。
# 区别：Java 的 main 方法是独立的，Python 的做法是把入口代码放在条件块中。

    history = []
    # 【解释】创建空列表，存储与 Lead Agent 的对话历史。
    # 在整个交互会话中持续累积，不会被清空（除非重启程序）。

    while True:
    # 【解释】主循环：持续接收用户输入，调用 agent_loop 处理。

        try:
        # 【解释】try-except 包裹输入操作，处理用户中断（Ctrl+C / Ctrl+D）。

            query = input("\033[36ms10 >> \033[0m")
            # 【解释】input() 函数从标准输入读取一行用户输入（类似 Java 的 Scanner.nextLine()）。
            # \033[36m 和 \033[0m 是 ANSI 转义码：
            #   \033[36m：设置文本颜色为青色（cyan）
            #   \033[0m：重置所有文本样式
            # 这些转义码在终端中会显示为彩色提示符 "s10 >> "。
            # 类似 Java 中 System.out.print("\u001B[36ms10 >> \u001B[0m")。

        except (EOFError, KeyboardInterrupt):
        # 【解释】捕获两种异常：
        #   EOFError：文件结束（Ctrl+D），类似 Java 中 NoSuchElementException。
        #   KeyboardInterrupt：用户中断（Ctrl+C），类似 Java 中没有直接对应（Java 会直接终止）。
        # (EOFError, KeyboardInterrupt)：元组语法，表示捕获多种异常。
        # 类似 Java 中 catch (EOFError | KeyboardInterrupt e)（Java 的多异常捕获）。

            break
            # 【解释】用户中断，退出循环。

        if query.strip().lower() in ("q", "exit", ""):
        # 【解释】检查用户是否输入了退出命令。
        # .strip()：去除首尾空白（类似 Java 的 trim()）。
        # .lower()：转为小写（类似 Java 的 toLowerCase()）。
        # in ("q", "exit", "")：检查是否在元组中。元组是不可变序列。

            break
            # 【解释】匹配退出命令，退出循环。

        if query.strip() == "/team":
        # 【解释】斜杠命令：列出团队成员。

            print(TEAM.list_all())
            # 【解释】print() 是 Python 的输出函数，类似 Java 的 System.out.println()。
            # 注意：print 会自动在末尾添加换行符，而 Java 的 System.out.print() 不会。

            continue
            # 【解释】continue 跳过本次循环的剩余部分，直接进入下一轮。
            # 类似 Java 中 for/while 循环中的 continue。

        if query.strip() == "/inbox":
        # 【解释】斜杠命令：查看 Lead 的收件箱。

            print(json.dumps(BUS.read_inbox("lead"), indent=2))

            continue

        history.append({"role": "user", "content": query})
        # 【解释】将用户输入追加到对话历史中。

        agent_loop(history)
        # 【解释】调用 Lead Agent 的主循环处理当前对话。
        # agent_loop 可能会多次调用 API（因为 AI 可能连续调用多个工具），直到 AI 给出最终回答。

        response_content = history[-1]["content"]
        # 【解释】history[-1] 获取列表最后一个元素（Python 支持负索引）。
        # -1 表示最后一个元素，-2 表示倒数第二个，以此类推。
        # 类似 Java 中 history.get(history.size() - 1)。
        # 获取 AI 最后一次响应的内容。

        if isinstance(response_content, list):
        # 【解释】isinstance() 是 Python 的类型检查函数。
        # 检查 response_content 是否是 list 类型（或其子类）。
        # 类似 Java 中 responseContent instanceof List。
        # AI 的响应可能是文本字符串，也可能是内容块列表（包含文本块和工具调用块）。

            for block in response_content:
            # 【解释】遍历响应的内容块。

                if hasattr(block, "text"):
                # 【解释】hasattr() 检查对象是否有指定的属性。
                # 类似 Java 中通过反射检查：block.getClass().getDeclaredField("text") != null。
                # 这里检查内容块是否包含 text 属性（文本块有，工具调用块没有）。

                    print(block.text)
                    # 【解释】输出文本内容。

        print()
        # 【解释】输出一个空行，分隔不同轮次的对话。
        # 类似 Java 中 System.out.println()（println 自带换行）。
