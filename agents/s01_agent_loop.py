#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.

# ─── shebang 行 ─────────────────────────────────────────────────────────────
# 类似 Java 没有的概念。告诉 Unix/Linux 系统用 env 程序在 PATH 中搜索 python3
# 解释器来执行此脚本。Java 直接由 JVM 运行 .class 文件，不需要这一行。
# 如果在 Windows 上运行则忽略此行。在 macOS/Linux 上还需 chmod +x 赋予执行权限。

"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

# ─── 模块导入 ────────────────────────────────────────────────────────────────
# Python 的 import 类似 Java 的 import，但更灵活：
# - Java: import java.util.Map;   → 导入特定类，必须用全限定名或类名访问
# - Python: import os             → 导入整个模块，通过 os.xxx 访问模块成员
# - Python: from os import getenv → 只导入模块中的特定函数，可以直接用 getenv()
# Python 的模块就是一个 .py 文件，对应 Java 中一个 .java 文件（编译后是 .class）
import os                  # 操作系统接口模块：环境变量、文件路径、进程管理等。类似 Java 的 System.getenv() + java.io.File + java.lang.Process 的集合
import subprocess          # 子进程管理模块：用于执行外部 shell 命令。类似 Java 的 ProcessBuilder

# ─── 条件导入与 try/except ────────────────────────────────────────────────────
# Python 的 try/except 等同于 Java 的 try/catch，但语法不同：
# - Java:  try { ... } catch (ImportException e) { ... }
# - Python: try: ... except ImportError: ...
# 注意：Python 的 except 不需要括号包裹异常类型，冒号代替大括号
try:
    # readline 是 Python 标准库中用于增强命令行输入的模块（类似 Java 没有
    # 直接等价物，但功能类似于 JLine 库）。它提供历史记录、行编辑等功能。
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    # macOS 使用 libedit 替代 GNU readline，默认不支持 UTF-8 的退格键处理
    # parse_and_bind() 用于向 readline 发送配置指令，类似 Java 中没有直接等价的 API
    readline.parse_and_bind('set bind-tty-special-chars off')   # 禁用特殊字符绑定，避免干扰 UTF-8 输入
    readline.parse_and_bind('set input-meta on')                # 开启输入元键（Meta/Alt），允许输入非 ASCII 字符
    readline.parse_and_bind('set output-meta on')               # 开启输出元键，正确显示非 ASCII 字符
    readline.parse_and_bind('set convert-meta off')             # 关闭元键转换，保留原始字节
    readline.parse_and_bind('set enable-meta-keybindings on')   # 启用 Meta 键绑定，使 Alt+Backspace 等组合键正常工作
except ImportError:
    # pass 类似 Java 的空 catch 块 catch (ImportException e) { /* 忽略 */ }
    # 如果 readline 不可用（某些极简 Python 安装），就跳过，不影响核心功能
    pass

# ─── from ... import ... 语法 ────────────────────────────────────────────────
# from anthropic import Anthropic  → 类似 Java 的 import anthropic.Anthropic;
# 区别：Python 导入后直接用 Anthropic，Java 也一样直接用类名
# Anthropic 是 Anthropic 公司官方 SDK 的客户端类，类似 Java 中 OkHttp 的 OkHttpClient
from anthropic import Anthropic
# load_dotenv 是 python-dotenv 库的函数，用于从 .env 文件加载环境变量
# 类似 Java 中 Spring Boot 的 @PropertySource 或直接读取 application.properties
from dotenv import load_dotenv

# ─── 环境变量与配置 ──────────────────────────────────────────────────────────
# 加载 .env 文件中的变量到 os.environ（进程环境变量字典）
# override=True 表示 .env 中的值会覆盖已有的系统环境变量
# 这与 Java 的 System.getenv() 不同——Java 的 getenv() 是只读的，而 Python 的
# os.environ 是一个可变的 dict（字典），可以直接读写
load_dotenv(override=True)

# os.getenv("KEY") 类似 Java 的 System.getenv("KEY")，返回字符串或 None（null）
# 如果设置了自定义的 API 基础 URL（例如代理或本地部署），则需要额外处理认证 token
if os.getenv("ANTHROPIC_BASE_URL"):
    # os.environ.pop(KEY, None) 类似 Java Map 的 remove()，从环境变量字典中
    # 删除指定 key。第二个参数 None 表示 key 不存在时不抛异常（类似 Java Map.remove() 返回 null）
    # 为什么这样做？自定义 BASE_URL 时，ANTHROPIC_AUTH_TOKEN 可能导致认证冲突
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ─── 客户端初始化与全局常量 ──────────────────────────────────────────────────
# 创建 Anthropic API 客户端实例。类似 Java 中 new OkHttpClient.Builder().baseUrl(url).build()
# base_url 参数允许自定义 API 地址（用于代理或自部署场景）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# os.environ["MODEL_ID"] 通过字典下标语法访问环境变量
# 与 os.getenv() 的区别：如果 MODEL_ID 不存在，os.environ[] 会抛出 KeyError 异常
# 而 os.getenv() 只返回 None。这里用 [] 是因为 MODEL_ID 是必需配置，缺失时应立即报错
# 类似 Java 中：String model = Optional.ofNullable(System.getenv("MODEL_ID")).orElseThrow()
# Python 中全大写的变量名约定表示"常量"，类似 Java 的 static final
# 但 Python 没有真正的常量机制，全大写只是命名约定
MODEL = os.environ["MODEL_ID"]

# ─── f-string 格式化字符串 ───────────────────────────────────────────────────
# f"..." 是 Python 的 f-string（格式化字符串字面量），Python 3.6+ 引入
# 花括号 {} 内的表达式会被求值并嵌入字符串
# 对比 Java:
#   Python: f"agent at {os.getcwd()}"
#   Java:   String.format("agent at %s", System.getProperty("user.dir"))
#   Java:   "agent at " + System.getProperty("user.dir")
#   Java 15+: "agent at %s".formatted(System.getProperty("user.dir"))
# os.getcwd() 返回当前工作目录，类似 Java 的 System.getProperty("user.dir")
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ─── dict 字典（Python 的核心数据结构） ─────────────────────────────────────
# TOOLS 是一个 list（列表），包含一个 dict（字典）
# Python 的 dict 类似 Java 的 HashMap<String, Object>，用 {} 创建
# Python 的 list 类似 Java 的 List<Object>，用 [] 创建
# key 必须是不可变类型（str, int, tuple 等），value 可以是任意类型
# 这是 Anthropic API 要求的 tool 定义格式（JSON Schema），告诉模型有哪些工具可用
TOOLS = [{  # [...] 创建一个 list，里面有一个 dict 元素。对比 Java: List.of(Map.of(...))
    "name": "bash",           # 工具名称，模型调用时会引用这个名字
    "description": "Run a shell command.",  # 工具描述，帮助模型理解何时使用此工具
    "input_schema": {         # 工具的输入参数 schema，遵循 JSON Schema 规范
        "type": "object",     # 参数是一个对象（类似 Java 的 POJO）
        "properties": {       # 对象的属性定义（类似 Java POJO 的字段）
            "command": {"type": "string"},  # command 字段，类型为 string
        },
        "required": ["command"],  # 必填字段列表
    },
}]


# ─── 函数定义 def ────────────────────────────────────────────────────────────
# Python 用 def 关键字定义函数，对比 Java：
#   Python: def run_bash(command: str) -> str:
#   Java:   public static String runBash(String command)
# 区别：
# 1. Python 的类型注解 (str) -> str 是可选的，仅用于静态检查，运行时不强制
#    Java 的类型声明是强制的，编译器会检查
# 2. Python 的 str 是内置类型；Java 的 String 是 java.lang.String 类
# 3. Python 默认所有函数都是"public"的，没有访问修饰符
# 4. Python 不需要 class 包裹函数（支持模块级函数），Java 的方法必须在类中
def run_bash(command: str) -> str:  # -> str 是返回值类型注解（可选），类似 Java 的返回类型 String
    # 定义危险命令列表，用 [] 创建 list，类似 Java 的 List.of("rm -rf /", "sudo", ...)
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # any() 是 Python 内置函数，类似 Java Stream 的 anyMatch()
    # any(d in command for d in dangerous) 使用生成器表达式（generator expression）
    # 等价于 Java: dangerous.stream().anyMatch(command::contains)
    # "x in y" 检查子字符串是否存在，类似 Java 的 y.contains(x)
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"  # 直接返回字符串，Python 不需要 new String()

    # try/except 对比 Java 的 try/catch/finally
    # Python 没有 finally 块（除非显式写），异常处理更简洁
    try:
        # subprocess.run() 在子进程中执行 shell 命令，类似 Java 的 ProcessBuilder：
        #   Java: new ProcessBuilder("sh", "-c", command).directory(cwd).start()
        # 参数说明：
        #   shell=True          → 通过系统 shell 执行（/bin/sh），支持管道、重定向等
        #   cwd=os.getcwd()     → 设置工作目录，类似 ProcessBuilder.directory()
        #   capture_output=True → 捕获 stdout 和 stderr，类似 Java 读取 process.getInputStream()
        #   text=True           → 将输出作为字符串返回（而非 bytes），类似 Java 的 new String(bytes, UTF_8)
        #   timeout=120         → 超时 120 秒，类似 Process.waitFor(120, TimeUnit.SECONDS)
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        # r.stdout 和 r.stderr 都是字符串，+ 用于字符串拼接（类似 Java）
        # .strip() 去除首尾空白，类似 Java 的 String.strip()
        out = (r.stdout + r.stderr).strip()
        # Python 的三元表达式：A if condition else B
        # 对比 Java 的三元运算符：condition ? A : B（顺序不同！）
        # out[:50000] 是字符串切片，取前 50000 个字符，类似 Java 的 out.substring(0, 50000)
        # 但 Python 切片不会越界，如果 out 长度不足 50000 则返回整个字符串
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 捕获超时异常，类似 Java 的 catch (TimeoutException e)
        # 注意 Python 的 except 后面跟的是异常类实例，不需要 as e（除非要用异常对象）
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        # 多异常捕获：(A, B) 表示同时捕获两种异常，类似 Java 的 catch (FileNotFoundException | OSError e)
        # as e 将异常对象赋值给变量 e，类似 Java 的 catch (Exception e)
        # f"Error: {e}" 中 {e} 会调用 e.__str__()，类似 Java 的 "Error: " + e.getMessage()
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
# agent_loop 函数实现了 AI Agent 的核心循环模式
# messages: list 类型注解，表示参数是一个列表
# 对比 Java：List<Map<String, Object>> messages（Python 不需要泛型语法）
# Python 是动态类型语言，类型注解只是提示，运行时不检查
def agent_loop(messages: list):
    # while True 是 Python 的无限循环写法，类似 Java 的 while (true)
    # 为什么不用 for？因为循环次数未知，取决于模型的 stop_reason
    while True:
        # ─── API 调用 ────────────────────────────────────────────────────
        # client.messages.create() 调用 Anthropic Messages API
        # 类似 Java 中用 HttpClient 发送 POST 请求到 https://api.anthropic.com/v1/messages
        # 但 SDK 封装了认证、序列化、重试等细节
        # 参数说明：
        #   model=MODEL     → 使用的模型 ID，如 "claude-sonnet-4-20250514"
        #   system=SYSTEM   → 系统提示词，设定 AI 的角色和行为准则
        #   messages=messages → 对话历史列表，API 要求的格式
        #   tools=TOOLS     → 可用工具列表，模型可以决定调用哪个工具
        #   max_tokens=8000 → 模型单次响应的最大 token 数
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # Append assistant turn
        # messages 是一个 list，.append() 在末尾添加元素，类似 Java 的 List.add()
        # 这里将模型响应添加到对话历史中，{"role": "assistant", ...} 是一个 dict
        # response.content 是模型返回的内容块列表（可能包含文本和工具调用）
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        # 检查模型是否决定停止。stop_reason 有多种值：
        #   "tool_use" → 模型请求调用工具，循环继续
        #   "end_turn" → 模型认为对话完成，循环结束
        #   "max_tokens" → 达到 token 上限，需要处理
        if response.stop_reason != "tool_use":
            return  # Python 的 return 不带值表示返回 None（类似 Java 的 return; 或 return null）
                      # 函数隐式返回 None，这是 Python 的惯例

        # Execute each tool call, collect results
        # 初始化空列表，类似 Java 的 List<Object> results = new ArrayList<>()
        results = []
        # for ... in ... 是 Python 的迭代循环，类似 Java 的 for-each：
        #   Python: for block in response.content:
        #   Java:   for (ContentBlock block : response.getContent())
        # response.content 是一个列表，包含多种类型的块：文本块、工具调用块等
        for block in response.content:
            # block.type 是属性访问，类似 Java 的 block.getType()
            # Python 不需要 getter/setter，直接访问属性
            if block.type == "tool_use":
                # ─── ANSI 颜色转义码 ────────────────────────────────────
                # \033 是八进制的 ESC 字符（ASCII 27），类似 Java 的 "\033" 或 "\u001b"
                # \033[33m 设置前景色为黄色（ANSI SGR 格式）
                # \033[0m 重置所有样式
                # Java 中也使用相同的 ANSI 码，但通常通过 JAnsi 等库封装
                # 这些码只在支持 ANSI 的终端中生效（大多数现代终端都支持）
                print(f"\033[33m$ {block.input['command']}\033[0m")
                # block.input 是一个 dict，block.input['command'] 用下标访问
                # 等价于 Java 中 block.getInput().get("command")
                # 注意：dict 的下标访问 key 不存在时会抛出 KeyError
                output = run_bash(block.input["command"])
                # 只打印前 200 字符到终端，避免刷屏。完整的 output 会发送给模型
                print(output[:200])
                # 将工具执行结果追加到 results 列表
                # 结果格式必须符合 Anthropic API 的 tool_result 规范：
                #   type: "tool_result"          → 固定值，标识这是一个工具结果
                #   tool_use_id: block.id         → 关联到对应的工具调用请求
                #   content: output               → 工具执行的输出内容
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})

        # 将工具结果作为"用户消息"追加到对话历史
        # 为什么 role 是 "user"？因为 Anthropic API 的协议要求：
        # 交替的 user/assistant 消息，tool_result 必须放在 user 消息中
        # content 是 results 列表（可以包含多个工具结果）
        messages.append({"role": "user", "content": results})
        # 循环回到 while True，将更新后的 messages（包含工具结果）再次发送给模型
        # 模型看到工具结果后，可以继续调用工具或给出最终回答


# ─── 程序入口 ────────────────────────────────────────────────────────────────
# if __name__ == "__main__": 是 Python 的惯用入口模式
# 对比 Java：
#   Java: public static void main(String[] args) — 入口方法
#   Python: 没有专用入口方法，整个文件从上到下执行
# __name__ 是 Python 内置变量：
#   - 直接运行此文件时，__name__ == "__main__"
#   - 被其他模块 import 时，__name__ == "s01_agent_loop"
# 这个 if 块确保以下代码只在直接运行时执行，被导入时不会执行
# 类似 Java 中 main 方法的隔离效果
if __name__ == "__main__":
    # history 是对话历史列表，初始为空
    # 整个程序运行期间，history 会不断累积 user 和 assistant 的消息
    # 类似 Java 的 List<Map<String, Object>> history = new ArrayList<>()
    history = []
    # 外层循环：读取用户输入，直到用户退出
    while True:
        try:
            # input() 读取一行用户输入，类似 Java 的 Scanner.nextLine() 或 BufferedReader.readLine()
            # \033[36m 是 ANSI 青色（cyan）前景色，\033[0m 重置
            # 所以提示符 "s01 >> " 会以青色显示
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # 多异常捕获：(EOFError, KeyboardInterrupt) 用元组包裹
            # EOFError: 用户按 Ctrl+D（Unix）或 Ctrl+Z（Windows），表示输入结束
            # KeyboardInterrupt: 用户按 Ctrl+C，中断程序
            # 类似 Java 中 catch (Exception e) 但更精确
            break  # 跳出 while True 循环，程序结束

        # .strip() 去除首尾空白，.lower() 转为小写
        # in ("q", "exit", "") 检查值是否在元组中
        # () 创建的元组 (tuple) 是不可变序列，类似 Java 中没有直接等价物（可用 List.of()）
        # 为什么用 tuple 而不是 list？因为这里只是做成员检查，tuple 更轻量且语义上表示"固定集合"
        if query.strip().lower() in ("q", "exit", ""):
            break  # 用户输入 q、exit 或空行时退出

        # 将用户消息追加到对话历史
        history.append({"role": "user", "content": query})
        # 调用 agent_loop 执行核心循环
        # 注意：agent_loop 会修改 history（因为是可变对象，按引用传递）
        # 这与 Java 一样：传递的是引用的副本，方法内可以修改对象内容
        agent_loop(history)

        # 获取最后一条消息的内容（即模型的最终响应）
        # history[-1] 用负索引访问列表最后一个元素，这是 Python 的特色语法
        # Java 中没有负索引，需要 history.get(history.size() - 1)
        # [-1] = 最后一个, [-2] = 倒数第二个，依此类推
        response_content = history[-1]["content"]

        # isinstance() 检查对象是否是某个类型的实例，类似 Java 的 instanceof 关键字：
        #   Python: isinstance(obj, list)
        #   Java:   obj instanceof List
        # 区别：Python 的 isinstance() 是函数调用，Java 的 instanceof 是运算符
        # 为什么需要检查？因为模型的 content 可能是字符串（纯文本响应）
        # 也可能是列表（包含文本块和工具调用块），需要区分处理
        if isinstance(response_content, list):
            for block in response_content:
                # hasattr() 是 Python 的内省（反射）函数，检查对象是否具有指定属性
                # 类似 Java 的反射：
                #   Python: hasattr(block, "text")
                #   Java:   block.getClass().getDeclaredField("text") != null（但更复杂）
                #   Java:   或者 block instanceof TextBlock（如果有多态类型系统）
                # 为什么用 hasattr 而不是 isinstance？因为 content 列表中混合了
                # 不同类型的块（TextBlock, ToolUseBlock 等），hasattr 更灵活
                if hasattr(block, "text"):
                    print(block.text)
        print()  # 打印空行，分隔对话轮次
