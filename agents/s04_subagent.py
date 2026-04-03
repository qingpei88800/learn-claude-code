#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

# -- 导入标准库（同 s01~s03 的公共基础设施部分）--
import os                                        # 同前：系统环境变量操作，等价于 Java 的 System.getenv() 等
import subprocess                                # 同前：执行外部命令，等价于 Java 的 ProcessBuilder / Runtime.exec()
from pathlib import Path                         # 同前：路径操作，对比 Java 的 java.nio.file.Path

# -- 导入第三方库（同前）--
from anthropic import Anthropic                  # 同前：Anthropic SDK 客户端
from dotenv import load_dotenv                   # 同前：从 .env 文件加载环境变量

# -- 环境变量初始化（同前）--
load_dotenv(override=True)                       # 同前：加载 .env，override=True 表示覆盖已有变量

# -- 防止代理冲突（同前）--
if os.getenv("ANTHROPIC_BASE_URL"):              # 同前：如果设置了自定义 API 地址
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)  # 同前：移除可能冲突的认证 token

# -- 全局常量（同前）--
WORKDIR = Path.cwd()                             # 同前：当前工作目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))  # 同前：API 客户端
MODEL = os.environ["MODEL_ID"]                   # 同前：模型名称

# s04 新增：父代理的 System prompt，增加了 task 工具的使用指引
# f-string 格式化，同前
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."

# s04 新增：子代理的 System prompt，与父代理完全不同
# 子代理只知道"完成给定任务并总结发现"，不知道有 task 工具的存在
# 这体现了"最小权限原则"——子代理不需要知道自己是被别人创建的
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child（同前）--
def safe_path(p: str) -> Path:
    # 路径安全校验，防止路径穿越攻击，同前
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    # 执行 shell 命令，同前
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    # s04 原文新增了 OSError 捕获，相比 s01~s03 更加健壮
    # (FileNotFoundError, OSError) 多异常捕获，等价于 Java 的 catch (FileNotFoundException | OSError e)
    except (FileNotFoundError, OSError) as e:
        return "Error: {e}"

def run_read(path: str, limit: int = None) -> str:
    # 读取文件内容，同前
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    # 写入文件，同前
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    # 编辑文件，精确替换文本，同前
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- TOOL_HANDLERS: 工具调度映射表（同前基础部分）--
# dict + lambda 实现策略模式，同前
# 注意：这里只有基础的 4 个工具（bash, read_file, write_file, edit_file）
# 没有包含 task 工具——task 工具的调度逻辑在 agent_loop 中单独处理
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# ============================================================
# CHILD_TOOLS：子代理可用的工具列表（s04 新增核心概念之一）
# ============================================================
# 子代理只能使用 4 个基础工具（bash, read_file, write_file, edit_file）
# 关键点：**不包含 task 工具**——禁止递归创建子代理
# 这是一种"安全限制"设计：
#   - 如果允许递归创建子代理，可能会出现无限嵌套（代理创建子代理，子代理又创建子代理...）
#   - 类比 Java 中的线程池限制递归创建线程，防止 StackOverflowError
#   - 在操作系统概念中，类似 fork bomb 的防范
# 每个元素是一个 dict，定义工具的 name、description、input_schema（JSON Schema 格式）
# LLM 根据这些定义决定何时调用哪个工具
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


# ============================================================
# run_subagent：子代理执行函数（s04 新增的核心函数）
# ============================================================
# 这是整个 s04 最关键的函数，实现了"上下文隔离"的核心思想：
#   1. 创建一个全新的消息列表 sub_messages（不继承父代理的任何历史）
#   2. 在这个干净的消息列表上运行独立的 agent 循环
#   3. 子代理可以自由使用文件系统（与父代理共享 WORKDIR）
#   4. 执行完毕后，只返回最终文本摘要给父代理
#   5. sub_messages 在函数返回后即被垃圾回收，上下文完全丢弃
#
# 对比 Java 世界中的概念：
#   - 类似于创建一个独立的线程或进程来执行子任务
#   - 类似于 Java 的 ExecutorService.submit(Callable) —— 提交任务，获取结果
#   - 不同之处：Java 线程共享堆内存，而子代理的消息列表是纯局部变量，天然隔离
#
# 参数：
#   prompt: str - 父代理传递给子代理的任务描述
# 返回：
#   str - 子代理执行后的文本摘要
def run_subagent(prompt: str) -> str:
    # ================================================================
    # 第一步：创建全新的消息列表 —— 这就是"上下文隔离"的起点
    # ================================================================
    # sub_messages 是一个局部变量，与父代理的 messages 完全独立
    # 初始状态只有一条 user 消息（父代理传来的任务描述）
    # 对比 Java：这类似于创建一个新的 ArrayList<>()，而不是复制父线程的数据
    # 子代理的消息列表：
    #   [0] {"role": "user", "content": prompt}  <-- 唯一的消息，干净的起点
    # 对比父代理的消息列表可能已经有几十条消息（多轮对话 + 多次工具调用）
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context

    # ================================================================
    # 第二步：子代理循环 —— 与父代理的 agent_loop 结构类似，但完全独立
    # ================================================================
    # for _ in range(30): 限制子代理最多执行 30 轮
    # _ 是 Python 的惯例"丢弃变量"（discard variable），
    # 表示这个循环变量不会被使用，仅用于控制循环次数
    # 类比 Java 中: for (int _ = 0; _ < 30; _++) { ... }
    # 在 Java 中通常用 for (int i = 0; i < 30; i++)，即使 i 没被用到
    # Python 用 _ 约定表示"我知道这个变量没用，只是需要一个循环计数器"
    # 30 轮是安全上限，防止子代理陷入无限循环（类似 Java 中的超时机制）
    for _ in range(30):  # safety limit

        # 调用 Claude API，注意这里的参数与父代理不同：
        #   - system=SUBAGENT_SYSTEM: 使用子代理专属的系统提示
        #     父代理知道可以用 task 工具，子代理不知道
        #   - messages=sub_messages: 使用子代理自己的消息列表（与父代理隔离）
        #   - tools=CHILD_TOOLS: 只有基础工具，没有 task 工具（禁止递归）
        # 这些参数差异确保了子代理"活在另一个世界"里
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )

        # 将模型的回复追加到子代理的消息列表
        # 注意：这里操作的是 sub_messages，不是父代理的 messages
        # 两个列表在内存中是完全不同的对象，互不影响
        # 类比 Java：两个不同的线程各自操作自己的局部变量 List<Message>
        sub_messages.append({"role": "assistant", "content": response.content})

        # 如果模型没有调用工具（stop_reason != "tool_use"），说明子任务完成
        # 跳出循环，准备返回结果
        if response.stop_reason != "tool_use":
            break

        # ================================================================
        # 第三步：执行子代理的工具调用（与父代理逻辑相同，但操作的是 sub_messages）
        # ================================================================
        # 收集本轮所有工具调用的结果
        results = []
        # 遍历模型返回的内容块列表，同前
        for block in response.content:
            if block.type == "tool_use":
                # 从 TOOL_HANDLERS 映射表查找对应的处理函数，同前
                # 子代理只能调用基础工具（bash/read_file/write_file/edit_file）
                # 因为 CHILD_TOOLS 中没有 task 工具，所以 block.name 不可能是 "task"
                handler = TOOL_HANDLERS.get(block.name)
                # 执行工具调用并获取结果，同前
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                # 将结果截断到 50000 字符，防止返回内容过大
                # str(output) 确保输出是字符串类型，再切片截断
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        # 将工具结果作为"用户消息"追加到子代理的消息列表
        # 注意：sub_messages 在这里持续增长，但函数返回后就整个被丢弃
        sub_messages.append({"role": "user", "content": results})

    # ================================================================
    # 第四步：提取最终文本摘要 —— 子代理只返回文本，不返回整个消息历史
    # ================================================================
    # 这是上下文隔离的关键设计：只返回摘要，不返回中间过程
    # 类比 Java：Callable<V> 的 call() 方法返回 V 结果，不返回执行过程的日志
    #
    # "".join(b.text for b in response.content if hasattr(b, "text"))
    # 这是一个生成器表达式（generator expression），嵌套在 str.join() 中
    # 逐步解析：
    #   1. response.content 是一个列表，包含 TextBlock 和 ToolUseBlock 等不同类型的块
    #   2. for b in response.content: 遍历每个内容块
    #   3. if hasattr(b, "text"): 过滤出有 text 属性的块（即 TextBlock）
    #      hasattr(b, "text") 是 Python 的运行时属性检查
    #      类比 Java：可以用 b instanceof TextBlock 检查类型，或用反射 b.getClass().getDeclaredField("text")
    #      Python 的 hasattr 比 Java 的 instanceof 更灵活——不要求知道具体类型，只关心"有没有这个属性"
    #   4. b.text: 取出文本内容
    #   5. "".join(...): 将所有文本片段拼接成一个字符串
    #      等价于 Java 的 String.join("", textBlocks.stream().map(b -> b.text).collect(Collectors.toList()))
    #
    # ... or "(no summary)": 如果模型没有返回任何文本（比如只返回了工具调用就达到了轮次上限），
    # 则返回 "(no summary)" 作为兜底值
    # Python 的 or 运算符在布尔上下文中：如果左边为空字符串（falsy），则返回右边的值
    # 类比 Java：text.isEmpty() ? "(no summary)" : text
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"


# ============================================================
# PARENT_TOOLS：父代理可用的工具列表（s04 新增核心概念之二）
# ============================================================
# 父代理的工具 = 子代理工具（CHILD_TOOLS）+ task 工具
# CHILD_TOOLS + [...] 是 Python 列表拼接，等价于 Java 的：
#   List<Map<String, Object>> parentTools = new ArrayList<>(childTools);
#   parentTools.add(taskTool);
# PARENT_TOOLS 包含 5 个工具，CHILD_TOOLS 包含 4 个工具
# 差异只有一个：父代理多了一个 task 工具，用于派生子代理
PARENT_TOOLS = CHILD_TOOLS + [
    # task 工具：这是 s04 新增的"子代理派发器"
    # 当 LLM 调用这个工具时，不是执行一个简单的函数，而是启动一个全新的 agent 上下文
    # 类比 Java 中的设计模式：
    #   - 命令模式（Command Pattern）：将"创建子代理"封装为一个可调用的命令
    #   - 工厂方法（Factory Method）：task 工具是一个工厂，创建出子代理实例
    # input_schema 定义：
    #   prompt（必填）: 传递给子代理的任务描述
    #   description（可选）: 任务的简短描述，用于日志/调试输出
    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},
]


# ============================================================
# agent_loop：父代理的主循环（与 s02/s03 结构类似，新增 task 工具分发）
# ============================================================
def agent_loop(messages: list):
    # while True: 无限循环，等价于 Java 的 while (true)，同前
    while True:
        # 调用 Claude API
        # 与子代理的关键差异：父代理使用 SYSTEM（不是 SUBAGENT_SYSTEM）、PARENT_TOOLS（不是 CHILD_TOOLS）
        # 父代理知道自己可以派生子代理（因为有 task 工具），子代理不知道
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,
        )
        # 将模型回复追加到消息历史，同前
        messages.append({"role": "assistant", "content": response.content})
        # 模型不调用工具时退出循环，同前
        if response.stop_reason != "tool_use":
            return
        # 收集工具调用结果，同前
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # ====================================================
                # s04 新增：task 工具的特殊分发逻辑
                # ====================================================
                # 如果模型调用了 task 工具，进入子代理分支
                # 其他工具（bash/read_file/write_file/edit_file）走原有逻辑
                if block.name == "task":
                    # block.input.get("description", "subtask") 获取任务描述，默认值 "subtask"
                    # block.input.get("prompt", "") 获取任务内容
                    # .get() 方法带默认值，key 不存在时不会抛异常，同前
                    desc = block.input.get("description", "subtask")
                    prompt = block.input.get("prompt", "")
                    # 打印任务派发信息，便于调试观察
                    # f-string 中 {prompt[:80]} 截取前 80 字符显示，防止长 prompt 淹没终端
                    print(f"> task ({desc}): {prompt[:80]}")
                    # ====================================================
                    # 核心：调用 run_subagent() 创建子代理执行子任务
                    # ====================================================
                    # run_subagent(prompt) 会：
                    #   1. 创建一个全新的消息列表 sub_messages = [prompt]
                    #   2. 在 sub_messages 上运行独立的 agent 循环（最多 30 轮）
                    #   3. 子代理可以使用 bash/read_file/write_file/edit_file 工具
                    #   4. 子代理不能使用 task 工具（禁止递归创建子代理）
                    #   5. 执行完毕后，只返回最终文本摘要
                    #   6. sub_messages 被丢弃（Python 垃圾回收自动回收内存）
                    #
                    # 这个调用是同步阻塞的（在当前线程中执行）：
                    #   - 父代理会等待子代理执行完毕后才继续
                    #   - 类比 Java 中 future.get() 阻塞等待 Callable 执行结果
                    #   - 后续的 s09 会展示并行子代理（类似 Java 的 CompletableFuture）
                    #
                    # output 就是子代理返回的文本摘要
                    # 父代理不会看到子代理的中间过程（消息列表、工具调用细节等）
                    # 这就是"上下文隔离"的价值——保护父代理的上下文不被子任务的细节污染
                    output = run_subagent(prompt)
                else:
                    # 非_task 工具：走原有的 TOOL_HANDLERS 分发逻辑，同前
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                # 打印工具结果的前 200 字符，同前
                print(f"  {str(output)[:200]}")
                # 将工具结果封装为 tool_result 追加到 results，同前
                # 注意：无论是 task 工具还是普通工具，结果格式都是一样的
                # 父代理无法从 tool_result 中区分哪些是子代理返回的，哪些是普通工具返回的
                # 这种统一的结果格式简化了父代理的处理逻辑
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        # 将工具结果追加到父代理的消息历史，同前
        # 如果其中有 task 工具的结果，那只是一段文本摘要（比如"文件包含3个函数..."）
        # 而不是子代理的完整消息历史——这是关键的"信息压缩"设计
        messages.append({"role": "user", "content": results})


# -- 主入口 REPL（同前）--
# if __name__ == "__main__": 是 Python 的惯用写法，等价于 Java 的 public static void main(String[] args)
if __name__ == "__main__":
    history = []   # 父代理的消息历史，贯穿整个对话过程
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")     # 提示符改为 s04 >>，同前
        except (EOFError, KeyboardInterrupt):           # 同前：捕获 Ctrl+D/Ctrl+C
            break
        if query.strip().lower() in ("q", "exit", ""):  # 同前：退出条件
            break
        history.append({"role": "user", "content": query})  # 同前：追加用户消息
        agent_loop(history)                                 # 同前：进入 agent 循环
        response_content = history[-1]["content"]           # 同前：取最后一条回复
        if isinstance(response_content, list):             # 同前：检查是否为列表
            for block in response_content:
                if hasattr(block, "text"):                 # 同前：检查是否有 text 属性
                    print(block.text)
        print()
