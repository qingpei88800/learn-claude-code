#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

# ============================================================
# 导入部分：同 s01，额外引入了 Path 和 subprocess
# ============================================================
import os                                      # 同 s01：系统环境变量操作，等价于 Java 的 System.getenv() 等
import subprocess                              # 同 s01：执行外部命令，等价于 Java 的 ProcessBuilder / Runtime.exec()
from pathlib import Path                       # Python 路径操作库，对比 Java 的 java.nio.file.Path
                                                # Python 的 Path 是一个纯数据类（不可变对象），可以直接用 / 拼接路径
                                                # Java 的 Path 是接口，路径拼接用 resolve() 或 resolveSibling()

from anthropic import Anthropic                # 同 s01：Anthropic SDK 客户端
from dotenv import load_dotenv                 # 同 s01：从 .env 文件加载环境变量

# 以下三段初始化逻辑同 s01，简略注释
load_dotenv(override=True)                     # 同 s01：加载 .env，override=True 表示覆盖已有变量
                                                # Java 中通常用 System.setProperty() 或 Spring 的 @PropertySource

if os.getenv("ANTHROPIC_BASE_URL"):            # 同 s01：如果设置了自定义 API 地址
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)  # 同 s01：移除可能冲突的认证 token

# ---- 以下开始是 s02 的新增逻辑 ----

# 获取当前工作目录。Path.cwd() 等价于 Java 的 Paths.get(".").toAbsolutePath()
# 不同之处：Java 的 Path 接口不提供获取 cwd 的静态方法，需要 Paths.get(".").toAbsolutePath()
# 而 Python 的 Path 类直接提供 cwd() 类方法
WORKDIR = Path.cwd()

# 同 s01：创建 Anthropic API 客户端，支持自定义 base_url（用于代理/私有部署）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 同 s01：从环境变量读取模型名称。os.environ["MODEL_ID"] 在 key 不存在时会抛 KeyError
# 对比 Java：如果用 Map.get() 不存在返回 null，而 Python 的 dict["key"] 不存在则抛 KeyError
MODEL = os.environ["MODEL_ID"]

# f-string 格式化字符串，等价于 Java 的 "You are a coding agent at " + WORKDIR + "..."
# f"..." 是 Python 3.6+ 的特性，{} 中可以直接放变量表达式
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ============================================================
# safe_path：路径安全校验（s02 新增）
# ============================================================
def safe_path(p: str) -> Path:
    # 将用户输入的字符串路径 p 与工作目录 WORKDIR 拼接，并解析为绝对路径
    # WORKDIR / p 使用了 Path 的 __truediv__ 运算符重载（operator overloading）
    # 在 Java 中，Path 拼接用 path.resolve(p)，不能直接用 / 运算符（Java 不支持运算符重载）
    # .resolve() 会解析所有 . 和 .. 符号，返回规范化绝对路径，类似于 Java 的 Path.toRealPath()
    # 但 .resolve() 不会检查路径是否实际存在，而 Java 的 toRealPath() 会
    path = (WORKDIR / p).resolve()

    # 检查 path 是否在 WORKDIR 范围内（防止路径穿越攻击，如 "../../../etc/passwd"）
    # is_relative_to() 是 Python 3.9+ 的方法，检查 self 是否是 other 的子路径
    # Java 中没有直接等价的方法，需要自行实现：path.startsWith(WORKDIR)
    if not path.is_relative_to(WORKDIR):
        # 抛出 ValueError，类似 Java 的 throw new IllegalArgumentException(...)
        # Python 中 ValueError 通常用于参数值不合法的场景
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# ============================================================
# run_bash：执行 Shell 命令（同 s01，仅将 cwd 从 os.getcwd() 改为 WORKDIR）
# ============================================================
def run_bash(command: str) -> str:
    # 危险命令黑名单。any() 是 Python 内置函数，等价于 Java Stream 的 anyMatch()
    # 这里用生成器表达式配合 any() 实现"只要有一个匹配就返回 True"
    # Java 写法：Arrays.stream(dangerous).anyMatch(d -> command.contains(d))
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 同 s01：subprocess.run() 执行 shell 命令
        # shell=True: 通过 /bin/sh 执行命令（类似 Java 的 new ProcessBuilder("sh", "-c", command)）
        # capture_output=True: 同时捕获 stdout 和 stderr（类似 Java 的 ProcessBuilder.redirectErrorStream(true)，但这里 stderr 是单独捕获的）
        # text=True: 将输出作为字符串而非字节返回（类似 Java 中用 new String(process.getInputStream().readAllBytes())）
        # timeout=120: 超时时间 120 秒，超时抛 subprocess.TimeoutExpired 异常
        # cwd=WORKDIR: 在指定目录执行命令（Java 中 ProcessBuilder.directory(new File(WORKDIR))）
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        # 拼接 stdout 和 stderr，并去除首尾空白（.strip() 类似 Java 的 String.trim()）
        out = (r.stdout + r.stderr).strip()
        # Python 三元表达式：a if condition else b
        # out[:50000] 是字符串切片，取前 50000 个字符（Java 中是 out.substring(0, Math.min(50000, out.length()))）
        # Python 切片越界不会报错，会自动截断；Java 的 substring 越界会抛 IndexOutOfBoundsException
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# ============================================================
# run_read：读取文件内容（s02 新增工具）
# ============================================================
def run_read(path: str, limit: int = None) -> str:
    # limit: int = None 表示默认值为 None（Java 方法中用 int limit = 0 或 Optional<Integer>）
    # Python 的 None 等价于 Java 的 null
    try:
        # safe_path(path) 先做路径安全校验
        # .read_text() 读取文件全部内容为字符串，等价于 Java 11+ 的 Files.readString(path)
        # 自动以 UTF-8 编码读取，Java 需要指定 Files.readString(path, StandardCharsets.UTF_8)
        text = safe_path(path).read_text()

        # .splitlines() 按换行符分割为列表，类似 Java 的 text.split("\\R")
        # 但 splitlines() 更智能，能处理 \n、\r\n、\r 等不同换行符
        # 返回值类型是 list[str]（Python 3.9+ 泛型语法），等价于 Java 的 List<String>
        lines = text.splitlines()

        # limit and limit < len(lines)：Python 的短路求值
        # 如果 limit 是 None（falsy），整个表达式为 False，不会执行 len(lines)
        # 注意：limit 不能是 0，因为 0 也是 falsy，但 limit=0 在业务上没有意义
        if limit and limit < len(lines):
            # 列表拼接：lines[:limit] 取前 limit 行（切片）
            # + [f"..."] 追加一个提示字符串
            # f"... ({len(lines) - limit} more lines)" 格式化字符串
            # 等价于 Java 的 String.format("... (%d more lines)", lines.size() - limit)
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]

        # "\n".join(lines) 将列表用换行符连接为字符串，等价于 Java 的 String.join("\n", lines)
        # [:50000] 截断输出，防止返回内容过大
        return "\n".join(lines)[:50000]
    except Exception as e:
        # 捕获所有异常，返回错误信息。Java 中通常用 catch (Exception e) 达到相同效果
        # 但 Java 最佳实践建议捕获更具体的异常类型
        return f"Error: {e}"


# ============================================================
# run_write：写入文件内容（s02 新增工具）
# ============================================================
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)    # 路径安全校验

        # fp.parent 获取父目录（Path 对象），等价于 Java 的 path.getParent()
        # .mkdir(parents=True, exist_ok=True) 创建目录
        #   parents=True: 类似 Java 的 Files.createDirectories()，会创建所有不存在的父目录
        #                  如果 parents=False，则等价于 Java 的 Files.createDirectory()，父目录不存在会抛异常
        #   exist_ok=True: 目录已存在时不抛异常，类似 Java 的 Files.createDirectories() 的默认行为
        #                   如果 exist_ok=False，目录已存在会抛 FileExistsError
        fp.parent.mkdir(parents=True, exist_ok=True)

        # .write_text(content) 将字符串写入文件（UTF-8 编码），等价于 Java 的 Files.writeString(path, content)
        # 如果文件不存在会自动创建，已存在则覆盖（类似 Java 的 StandardOpenOption.CREATE, TRUNCATE_EXISTING）
        fp.write_text(content)

        # len(content) 返回字符串长度（字符数），等价于 Java 的 content.length()
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# run_edit：精确替换文件中的文本片段（s02 新增工具）
# ============================================================
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)            # 路径安全校验
        content = fp.read_text()         # 读取文件全部内容，等价于 Files.readString()

        # `in` 运算符检查子字符串是否存在，等价于 Java 的 content.contains(old_text)
        if old_text not in content:
            return f"Error: Text not found in {path}"

        # str.replace(old, new, count) 的第三个参数是替换次数
        # content.replace(old_text, new_text, 1) 表示只替换第一个匹配项
        # 对比 Java：String.replace() 会替换所有匹配，没有限制次数的参数
        # Java 中要实现"只替换第一个"需要用 Pattern + Matcher 或 substring 手动定位
        # 这是 Python 字符串处理比 Java 更方便的一个例子
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# TOOL_HANDLERS：工具分发映射表（s02 核心：调度器模式）
# 这是整个 s02 最重要的设计：用 dict + lambda 实现"策略模式 + 简单工厂"
# ============================================================
# dict 字面量创建映射表，key 是工具名（str），value 是 lambda 函数
# 这本质上是一个"策略模式"（Strategy Pattern）的极简实现：
#   Java 中通常定义一个 interface ToolHandler { String execute(Map<String,Object> params); }
#   然后为每个工具实现一个类，再通过 Map<String, ToolHandler> 注册
#   Python 用 lambda 一行搞定，无需定义接口和实现类
TOOL_HANDLERS = {
    # lambda **kw: ... 中的 **kw 是"关键字参数解包"（keyword arguments unpacking）
    # 等价于 Java 方法签名中的 (Map<String, Object> params)
    # **kw 会将传入的关键字参数收集成一个 dict
    # 例如调用 handler(command="ls") 时，kw 就是 {"command": "ls"}
    # kw["command"] 通过 dict 的 [] 运算符取值（类似 Java Map 的 map.get(key)，但 key 不存在时抛 KeyError 而非返回 null）
    "bash":       lambda **kw: run_bash(kw["command"]),

    # kw.get("limit") 使用 dict 的 .get() 方法，key 不存在时返回 None（而非抛异常）
    # 对比 Java：Map.get(key) 在 key 不存在时返回 null，行为类似
    # 而 kw["limit"] 在 key 不存在时会抛 KeyError，类似 Java 中直接访问 Map 中不存在的 key 会抛 NullPointerException
    # 这里用 .get() 是因为 limit 是可选参数，可能不存在于调用参数中
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),

    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),

    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# TOOLS: 工具定义列表，传递给 Anthropic API，告诉模型有哪些工具可用
# 每个 dict 描述一个工具的名称、说明和输入参数的 JSON Schema
# 这个列表会序列化为 JSON 传给 API，模型根据这些定义决定调用哪个工具、传什么参数
# list 中嵌套 dict，等价于 Java 的 List<Map<String, Object>>
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     # input_schema 遵循 JSON Schema 规范，定义工具输入的结构
     # type: "object" 表示参数是一个 JSON 对象
     # properties: 定义各字段的名称和类型
     # required: 声明哪些字段是必填的
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


# ============================================================
# agent_loop：Agent 主循环（与 s01 结构相同，区别在于工具分发逻辑）
# ============================================================
def agent_loop(messages: list):
    # while True: 无限循环，等价于 Java 的 while (true)
    while True:
        # 调用 Anthropic API，传入消息历史、系统提示、工具定义
        # 与 s01 的区别：s01 的 TOOLS 只有一个 bash 工具，s02 有 4 个工具
        # API 调用是同步阻塞调用，等价于 Java 的同步 HTTP 请求
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,    # max_tokens 限制模型单次回复的最大 token 数
        )
        # 将模型的回复追加到消息历史中
        # messages 是 list，.append() 在末尾添加元素，等价于 Java 的 list.add()
        # 注意：Python 的 list 是可变序列，在函数内部修改列表会影响到调用者（因为 list 是引用传递）
        # 这在 Java 中也是一样的——传递的是引用的副本
        messages.append({"role": "assistant", "content": response.content})

        # stop_reason 表示模型为什么停止生成
        # "tool_use" 表示模型要调用工具，需要继续循环
        # "end_turn" 表示模型认为任务完成，可以退出循环
        if response.stop_reason != "tool_use":
            return  # 函数无返回值，等价于 Java 的 return;（void 返回）

        # 收集所有工具调用的执行结果
        results = []

        # 遍历模型返回的内容块列表
        # response.content 是一个 list，每个元素可能是 TextBlock 或 ToolUseBlock
        # Python 的 for ... in 语法等价于 Java 的 for (var block : response.content)
        for block in response.content:
            if block.type == "tool_use":
                # TOOL_HANDLERS.get(block.name) 从映射表查找对应的处理函数
                # dict.get(key) 在 key 不存在时返回 None，不会抛异常
                # 对比 Java 的 Map.get(key)，行为相同——不存在返回 null
                # 如果用 dict[key]，key 不存在会抛 KeyError（类似 Java 中对 null 调用方法）
                handler = TOOL_HANDLERS.get(block.name)

                # handler(**block.input) 是关键：
                #   1. block.input 是一个 dict，例如 {"command": "ls -la"}
                #   2. ** 运算符将 dict "解包"为关键字参数
                #      等价于 handler(command="ls -la")
                #      类似 Java 中用反射将 Map<String,Object> 的键值对映射为方法参数
                #   3. 这种"字典解包"是 Python 的核心特性之一，Java 没有直接等价物
                # if handler else ...: 如果 handler 是 None（即未知工具），返回错误信息
                # 这是 Python 的条件表达式，等价于 Java 的 ternary: handler != null ? handler.invoke(kw) : "error"
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"

                # 打印工具调用信息，[:200] 截断显示（防止终端被大量输出淹没）
                print(f"> {block.name}:")
                print(output[:200])

                # 将工具执行结果封装为 tool_result 格式，追加到 results 列表
                # tool_use_id 必须与模型的 tool_use 请求一一对应（API 协议要求）
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        # 将所有工具结果作为"用户消息"追加到历史
        # API 协议约定：工具结果必须以 role="user" 的消息形式返回
        # content 是一个 list（多个 tool_result），而非单个字符串
        messages.append({"role": "user", "content": results})


# ============================================================
# 主入口：交互式 REPL 循环（同 s01，仅提示符改为 s02 >>）
# ============================================================
# if __name__ == "__main__": 是 Python 的惯用写法
# 等价于 Java 的 public static void main(String[] args)
# 当文件被直接执行时 __name__ 为 "__main__"；被其他文件 import 时 __name__ 为模块名
if __name__ == "__main__":
    history = []   # 消息历史列表，贯穿整个对话过程
    while True:
        try:
            # input() 从标准输入读取一行，等价于 Java 的 Scanner.nextLine() 或 BufferedReader.readLine()
            # \033[36m 和 \033[0m 是 ANSI 转义码：设置文字颜色为青色 / 恢复默认颜色
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # 捕获 Ctrl+D（EOFError）和 Ctrl+C（KeyboardInterrupt）
            # (EOFError, KeyboardInterrupt) 是元组语法，表示捕获多种异常类型
            # 等价于 Java 的 catch (EOFError | KeyboardInterrupt e)（Java 7+ multi-catch）
            break
        # .strip() 去除首尾空白，.lower() 转为小写
        # in ("q", "exit", "") 检查值是否在元组中，等价于 Java 的 Set.of("q", "exit", "").contains(value)
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户输入追加到消息历史
        history.append({"role": "user", "content": query})

        # 进入 agent 循环（模型可能会调用多次工具）
        agent_loop(history)

        # agent_loop 返回后，history 的最后一条就是模型的最终回复
        # history[-1] 用负索引取最后一个元素，等价于 Java 的 history.get(history.size() - 1)
        # 这是 Python 列表的特性：负索引从末尾开始计数，-1 是最后一个，-2 是倒数第二个
        response_content = history[-1]["content"]

        # isinstance() 检查对象类型，等价于 Java 的 instanceof 关键字
        # 这里检查 response_content 是否是 list 类型
        # 需要检查的原因：模型最终回复可能是 list（包含多个 content block），也可能是单个字符串
        if isinstance(response_content, list):
            for block in response_content:
                # hasattr(block, "text") 检查对象是否有 text 属性
                # 类似 Java 的反射：block.getClass().getDeclaredField("text") != null
                # 但 hasattr 更通用，不仅限于字段，也适用于任何属性
                if hasattr(block, "text"):
                    print(block.text)
        print()
