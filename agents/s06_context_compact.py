#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
# Shebang 行: 指定用 python3 解释器执行。Java 没有等价物，Java 通过 JVM 启动。
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:
    三层压缩管线，让 Agent 可以无限运行而不超出上下文窗口。

    Every turn:
    每一轮（Agent 调用工具 -> 获得结果 -> 再次思考）都会经过压缩检查。

    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace non-read_file tool_result content older than last 3
      with "[Previous: used {tool_name}]"
      第一层：微压缩。每次调用 LLM 前静默执行。
      把超过最近 3 个的工具结果内容替换为占位符。
      注意：read_file 的结果不会被压缩，因为它们是参考材料。
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                  第二层：自动压缩。当估算 token 超过 50000 阈值时触发。
                  先把完整对话保存到 .transcripts/ 目录，然后让 LLM 摘要，
                  最后用一条摘要消息替换所有历史消息。
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.
                  第三层：手动压缩。模型主动调用 compact 工具时触发。
                  逻辑与 auto_compact 相同，但由 Agent 自主决定何时压缩。

Key insight: "The agent can forget strategically and keep working forever."
核心洞察：「Agent 可以策略性地遗忘，从而永远工作下去。」
类比 Java：这就像 JVM 的 GC 机制 —— 程序不需要关心内存回收，
压缩管线自动管理上下文窗口，让 Agent 可以无限运行。
"""

# ============================================================
# 导入模块
# Python 的 import 类似 Java 的 import，但 Python 没有 package-private 可见性。
# ============================================================
import json                                          # JSON 序列化/反序列化，类似 Java 的 com.fasterxml.jackson.databind.ObjectMapper
import os                                            # 操作系统接口，类似 Java 的 java.lang.System + java.io.File
import subprocess                                    # 子进程管理，类似 Java 的 ProcessBuilder / Runtime.exec()
import time                                          # 时间工具，类似 Java 的 java.time.Instant
from pathlib import Path                             # 路径操作，类似 Java 的 java.nio.file.Path

from anthropic import Anthropic                      # Anthropic SDK 客户端，类似 Java 的 HttpClient 封装
from dotenv import load_dotenv                       # 从 .env 文件加载环境变量，类似 Java 的 spring-dotenv

load_dotenv(override=True)
# 加载 .env 文件中的环境变量。override=True 表示覆盖已存在的变量。
# Java 中通常通过 SpringApplication.run() 或 @PropertySource 自动加载。

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
# 如果设置了自定义 API 地址（如代理），则移除 AUTH_TOKEN 避免认证冲突。
# os.getenv() 类似 Java 的 System.getenv()，返回 Optional<String>（Python 中为 None）。
# os.environ.pop() 类似 Java 的 System.getProperties().remove()。

WORKDIR = Path.cwd()
# 获取当前工作目录。Path.cwd() 类似 Java 的 Path.of("").toAbsolutePath()。
# Python 顶层变量类似 Java 的 static final 字段。

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 创建 Anthropic API 客户端实例。类似 Java 中 new HttpClient() 并配置 baseUrl。

MODEL = os.environ["MODEL_ID"]
# 从环境变量获取模型 ID。os.environ["KEY"] 类似 Java 的 System.getenv("KEY")，
# 但如果 key 不存在会抛出 KeyError，类似 Java 的 Objects.requireNonNull()。

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."
# f-string 是 Python 的格式化字符串，类似 Java 的 String.format() 但更简洁。
# Java 写法: "You are a coding agent at " + WORKDIR + ". Use tools to solve tasks."

# ============================================================
# 压缩配置常量
# Python 中模块级常量通常用全大写命名，类似 Java 的 static final。
# ============================================================
THRESHOLD = 50000
# 自动压缩的 token 阈值。当估算 token 数超过此值时触发 auto_compact。
# Claude 的上下文窗口约为 200K tokens，这里设置 50K 作为安全线。

TRANSCRIPT_DIR = WORKDIR / ".transcripts"
# 对话记录保存目录。Path 的 "/" 运算符重载用于拼接路径，
# 类似 Java 的 Path.resolve(".transcripts")。

KEEP_RECENT = 3
# micro_compact 时保留最近的 N 个工具结果不压缩。

PRESERVE_RESULT_TOOLS = {"read_file"}
# 集合（set），包含不应被 micro_compact 压缩的工具名。
# Python 的 set 类似 Java 的 java.util.HashSet。
# read_file 的结果保留，因为它们是参考材料；压缩后会迫使 Agent 重新读取文件。


# ============================================================
# 辅助函数：估算 token 数
# ============================================================
def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    # 粗略估算 token 数：约 4 个字符 = 1 个 token。
    # 这是一种启发式方法，实际应使用 tokenizer，但这里为了简单。
    # Java 中可能使用: messageList.toString().length() / 4
    return len(str(messages)) // 4
    # str(messages) 将整个消息列表转为字符串。
    # // 是 Python 的整除运算符，类似 Java 的整数除法（自动截断）。


# ============================================================
# 第一层压缩：micro_compact（微压缩）
# 核心思想：每次调用 LLM 前，把旧的工具结果内容替换为简短占位符。
# 这是最轻量的压缩，静默执行，对 Agent 透明。
#
# 类比 Java：
#   想象一个 List<Message>，其中某些 Message 包含大量工具输出。
#   micro_compact 会遍历这些消息，把旧的输出内容替换为 "[Previous: used xxx]"。
#   类似于 Java 中对大对象做 toString() 时只返回摘要信息。
# ============================================================
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    # 收集所有 tool_result 条目的位置信息（消息索引、部分索引、结果字典）
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        # enumerate() 同时返回索引和值，类似 Java 的 fori 循环 + list.get(i)。
        # Java 写法: for (int msgIdx = 0; msgIdx < messages.size(); msgIdx++) { ... }
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            # Claude API 中，tool_result 消息的 role 是 "user"，content 是一个列表。
            # isinstance() 是 Python 的类型检查，类似 Java 的 instanceof。
            # msg.get("content") 类似 Java 的 map.get()，键不存在返回 None 而不抛异常。
            for part_idx, part in enumerate(msg["content"]):
                # 遍历 user 消息的 content 列表中的每个部分。
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    # 找到 type 为 "tool_result" 的部分。
                    # 在 Claude API 中，工具调用结果以 tool_result 类型嵌入 user 消息中。
                    tool_results.append((msg_idx, part_idx, part))
                    # 收集三元组：(消息在 messages 中的索引, 部分在 content 中的索引, 结果字典)
    if len(tool_results) <= KEEP_RECENT:
        # 如果工具结果数量不超过保留阈值（3），无需压缩，直接返回。
        return messages

    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    # 通过匹配 prior assistant 消息中的 tool_use_id 来查找每个结果对应的工具名。
    # 这是因为 tool_result 中只有 tool_use_id，没有工具名本身。
    tool_name_map = {}
    # 空字典，类似 Java 的 new HashMap<String, String>()。
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            # 获取 assistant 消息的 content，默认为空列表。
            if isinstance(content, list):
                for block in content:
                    # 遍历 assistant 消息的每个内容块。
                    if hasattr(block, "type") and block.type == "tool_use":
                        # hasattr() 检查对象是否有某属性，类似 Java 的反射 field != null。
                        # Anthropic SDK 返回的 content 块是对象（不是纯 dict），所以用 hasattr。
                        # block.type == "tool_use" 表示这是一个工具调用块。
                        tool_name_map[block.id] = block.name
                        # 建立 tool_use_id -> tool_name 的映射。
                        # 后面用这个映射来查出 tool_result 对应的是哪个工具。

    # Clear old results (keep last KEEP_RECENT). Preserve read_file outputs because
    # they are reference material; compacting them forces the agent to re-read files.
    # 清除旧的工具结果（保留最近 KEEP_RECENT 个）。
    # read_file 的输出被保留，因为它们是参考材料；压缩它们会迫使 Agent 重新读取文件。
    to_clear = tool_results[:-KEEP_RECENT]
    # 切片操作 [:-3] 表示去掉最后 3 个元素，保留前面所有。
    # 类似 Java 的 list.subList(0, list.size() - 3)。
    for _, _, result in to_clear:
        # 遍历需要清除的结果。用 _ 表示忽略不需要的变量，类似 Java 中不使用的循环变量。
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            # 跳过内容不是字符串或长度 <= 100 的结果 —— 太短了不值得压缩。
            continue
            # continue 类似 Java 的 continue，跳过当前循环迭代。
        tool_id = result.get("tool_use_id", "")
        # 获取工具调用 ID，用于查找对应的工具名。
        tool_name = tool_name_map.get(tool_id, "unknown")
        # 从映射中查找工具名，找不到则用 "unknown"。类似 Java 的 map.getOrDefault()。
        if tool_name in PRESERVE_RESULT_TOOLS:
            # 如果是 read_file 的结果，跳过压缩。
            # Python 的 in 操作符检查集合成员，类似 Java 的 set.contains()。
            continue
        result["content"] = f"[Previous: used {tool_name}]"
        # 关键操作：将工具结果内容替换为简短占位符。
        # 这是原地修改（因为 result 是 dict 的引用），不需要返回新列表。
        # Java 中需要 result.put("content", "[Previous: used " + toolName + "]")。
    return messages
    # 返回修改后的 messages。因为 micro_compact 是原地修改 dict 引用，
    # 所以返回值和传入的是同一个列表对象。


# ============================================================
# 第二层压缩：auto_compact（自动压缩）
# 核心思想：当上下文接近窗口上限时，让 LLM 自己总结对话，然后替换全部历史。
#
# 类比 Java：
#   想象一个 StringBuilder 积累了大量数据，快到内存上限时，
#   先把完整内容持久化到文件（类似 Java 的 Files.write()），
#   然后用一个简短摘要替换 StringBuilder 的全部内容。
# ============================================================
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    # 将完整对话记录保存到磁盘，确保信息不丢失。

    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    # 创建 .transcripts 目录，exist_ok=True 类似 Java 的 Files.createDirectories()。
    # 如果目录已存在不会报错。

    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    # 构造文件路径。int(time.time()) 返回当前 Unix 时间戳（秒），
    # 类似 Java 的 (int)(System.currentTimeMillis() / 1000)。
    # .jsonl 是 JSON Lines 格式 —— 每行一个 JSON 对象。

    with open(transcript_path, "w") as f:
        # with 语句是 Python 的上下文管理器，类似 Java 的 try-with-resources。
        # open(path, "w") 以写模式打开文件，类似 Java 的 new FileWriter(path)。
        # with 块结束后文件自动关闭，类似 Java 中 try() 块结束自动 close()。
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
            # json.dumps() 将 Python 对象序列化为 JSON 字符串，
            # 类似 Java 的 objectMapper.writeValueAsString(msg)。
            # default=str 是关键参数：当遇到不可序列化的对象（如 datetime、Path 等）时，
            # 调用 str() 转换为字符串，避免 TypeError。
            # Java 中通常需要自定义 Serializer 或配置 ObjectMapper 的 FAIL_ON_UNKNOWN_PROPERTIES。
            # 每条消息占一行（JSONL 格式），便于后续逐行读取。

    print(f"[transcript saved: {transcript_path}]")
    # 打印保存路径，方便调试。print() 类似 Java 的 System.out.println()。

    # Ask LLM to summarize
    # 让 LLM 总结对话，提取关键信息。

    conversation_text = json.dumps(messages, default=str)[-80000:]
    # 将所有消息序列化为 JSON 字符串，然后截取最后 80000 个字符。
    # 截取尾部是因为最近的对话通常最重要。
    # [-80000:] 是 Python 的负索引切片，类似 Java 的
    # str.substring(Math.max(0, str.length() - 80000))。

    response = client.messages.create(
        # 调用 Claude API 进行总结。类似 Java 中发送 HTTP POST 请求。
        model=MODEL,
        # 使用配置的模型。
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        # 把对话原文作为用户消息发送给 LLM，让它总结。
        # 提示词要求包含：1) 完成了什么 2) 当前状态 3) 关键决策。
        max_tokens=2000,
        # 限制总结的长度为 2000 tokens，避免总结本身占用太多空间。
    )

    summary = next((block.text for block in response.content if hasattr(block, "text")), "")
    # 从 LLM 响应中提取文本内容。这里用到了几个 Python 特性：
    #
    # 1. 生成器表达式 (block.text for block in ...)
    #    类似 Java Stream: response.content.stream()
    #        .filter(block -> block instanceof TextBlock)
    #        .map(block -> ((TextBlock) block).getText())
    #
    # 2. next(generator, default)
    #    获取生成器的第一个值，如果生成器为空则返回默认值 ""。
    #    类似 Java 的: stream.findFirst().orElse("")
    #    或者传统迭代器: if (iterator.hasNext()) return iterator.next(); else return "";
    #
    # 3. hasattr(block, "text")
    #    检查 content 块是否有 text 属性。
    #    Claude API 响应的 content 可能包含 TextBlock 和 ToolUseBlock 等不同类型。

    if not summary:
        # 如果 LLM 没有返回有效总结（罕见情况），使用默认文本。
        # Python 中空字符串 "" 是 falsy 值，所以 not summary 为 True。
        summary = "No summary generated."

    # Replace all messages with compressed summary
    # 用压缩后的摘要替换所有消息。
    # 这就是「策略性遗忘」的核心 —— 丢弃全部历史，只保留摘要。
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]
    # 返回一个只包含一条消息的新列表。
    # 这条消息告知 LLM：「对话已压缩，原始记录在 transcript_path，以下是摘要」。
    # 关键：返回的是新列表，不是原地修改。
    # 调用方需要用 messages[:] = auto_compact(messages) 来替换原列表的内容。


# ============================================================
# 工具实现
# 以下函数都是 Agent 可以调用的工具的具体实现。
# ============================================================

def safe_path(p: str) -> Path:
    # 同前：安全路径解析，防止路径遍历攻击。
    path = (WORKDIR / p).resolve()
    # resolve() 解析为绝对路径并消除 .. 等，类似 Java 的 Path.toAbsolutePath().normalize()。
    if not path.is_relative_to(WORKDIR):
        # 检查路径是否在工作目录内，防止路径遍历（如 ../../../etc/passwd）。
        # is_relative_to() 是 Python 3.9+ 的方法，类似 Java 的 path.startsWith(WORKDIR)。
        raise ValueError(f"Path escapes workspace: {p}")
        # 抛出 ValueError，类似 Java 的 throw new IllegalArgumentException()。
    return path

def run_bash(command: str) -> str:
    # 同前：执行 shell 命令，带危险命令过滤和超时保护。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        # any() 类似 Java Stream 的 anyMatch()：
        # dangerous.stream().anyMatch(d -> command.contains(d))
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        # subprocess.run() 执行命令，类似 Java 的 ProcessBuilder.start()。
        # shell=True 通过 shell 执行，text=True 返回字符串而非字节。
        # timeout=120 设置 120 秒超时，类似 Java 的 Process.waitFor(120, TimeUnit.SECONDS)。
        out = (r.stdout + r.stderr).strip()
        # 合并标准输出和标准错误。strip() 去除首尾空白，类似 Java 的 String.trim()。
        return out[:50000] if out else "(no output)"
        # 截断输出到 50000 字符，防止工具结果过大。
        # [:50000] 是切片，类似 Java 的 str.substring(0, Math.min(50000, str.length()))。
    except subprocess.TimeoutExpired:
        # 捕获超时异常，类似 Java 的 catch (TimeoutException e)。
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    # 同前：读取文件内容，支持行数限制。
    try:
        lines = safe_path(path).read_text().splitlines()
        # read_text() 读取整个文件为字符串，类似 Java 的 Files.readString(path)。
        # splitlines() 按行分割，类似 Java 的 String.split("\\n")。
        if limit and limit < len(lines):
            # limit 默认为 None，Python 中 None 是 falsy 值。
            # Java 中需要: if (limit != null && limit < lines.size())
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
            # 截取前 limit 行，并追加提示信息。
            # 列表拼接 lines[:limit] + [...] 类似 Java 的
            #   new ArrayList<>(lines.subList(0, limit)) {{ add("..."); }}。
        return "\n".join(lines)[:50000]
        # "\n".join() 用换行符合并列表，类似 Java 的 String.join("\n", lines)。
    except Exception as e:
        # 捕获所有异常，返回错误信息。类似 Java 的 catch (Exception e)。
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    # 同前：写入文件内容。
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        # 创建父目录，类似 Java 的 Files.createDirectories(fp.getParent())。
        fp.write_text(content)
        # 写入文件，类似 Java 的 Files.writeString(fp, content)。
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    # 同前：精确替换文件中的文本。
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            # 检查旧文本是否存在，类似 Java 的 content.contains(oldText)。
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        # str.replace(old, new, count) 替换，count=1 只替换第一个匹配。
        # 类似 Java 没有直接等价的方法，需要手动用 indexOf + substring 实现。
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# 工具注册表
# Python 的 dict 类似 Java 的 Map<String, BiFunction<String, Object, String>>。
# lambda **kw: ... 是 Python 的关键字参数 lambda，类似 Java 的 (Map<String, Object> kw) -> ...
# ============================================================
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    # **kw 收集所有关键字参数为字典，类似 Java 的 Map<String, Object> params。
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    # kw.get("limit") 如果不存在返回 None，类似 Java 的 map.getOrDefault("limit", null)。
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
    # compact 工具不需要实际执行，只需要标记「手动压缩请求」。
    # 实际的压缩逻辑在 agent_loop 中处理。
}

TOOLS = [
    # 工具定义列表，告诉 Claude 有哪些工具可用。
    # 每个工具的定义遵循 Anthropic API 的 JSON Schema 格式。
    # 类似 Java 中定义一个 ToolDescriptor 类的 List<ToolDescriptor>。
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     # s06 新增：手动压缩工具。Agent 可以在感到上下文拥挤时主动调用此工具。
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


# ============================================================
# Agent 主循环
# 核心循环：调用 LLM -> 执行工具 -> 将结果返回 LLM -> 重复。
# 与前几版的区别：每轮都执行 micro_compact，并检查是否需要 auto_compact。
# ============================================================
def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        # 第一层：在每次 LLM 调用前执行微压缩。
        micro_compact(messages)
        # micro_compact 是原地修改 messages 中的 dict 内容，返回值是同一个列表。
        # 类似 Java 中对一个 List<Message> 内部的 Message 对象做修改。

        # Layer 2: auto_compact if token estimate exceeds threshold
        # 第二层：如果 token 估算超过阈值，执行自动压缩。
        if estimate_tokens(messages) > THRESHOLD:
            # 检查当前消息的估算 token 数是否超过阈值。
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
            # ★★★ 关键语法：messages[:] = ... ★★★
            # 这是 Python 的「切片赋值」，用右边的新列表替换左边列表的全部元素。
            # 等价于：messages.clear(); messages.addAll(auto_compact(messages))
            #
            # 为什么不直接 messages = auto_compact(messages)？
            # 因为那只会修改局部变量 messages 的引用，不会影响调用方的列表！
            # 类比 Java：
            #   messages = newList;        // 只改变局部引用，外部的 list 不变
            #   messages.clear();          // 清空原列表
            #   messages.addAll(newList);  // 填充新内容 —— 这才是原地修改
            #
            # Python 的切片赋值 messages[:] = new_list 就是上述 Java 操作的简写。
            # auto_compact 返回一个只包含一条摘要消息的新列表。

        response = client.messages.create(
            # 调用 Claude API。messages 可能已经被压缩过了。
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        # 将 LLM 的响应追加到消息列表。append() 类似 Java 的 list.add()。

        if response.stop_reason != "tool_use":
            # 如果 LLM 不需要调用工具（直接给出文字回答），则退出循环。
            # stop_reason 可能是 "end_turn"（正常结束）或 "tool_use"（需要调用工具）。
            return

        results = []
        manual_compact = False
        # manual_compact 标志：标记本轮是否调用了 compact 工具。
        for block in response.content:
            # 遍历 LLM 响应中的每个内容块。
            # response.content 是一个列表，可能包含 TextBlock 和 ToolUseBlock。
            if block.type == "tool_use":
                # 如果是工具调用块。
                if block.name == "compact":
                    # Layer 3: 如果调用的是 compact 工具，设置手动压缩标志。
                    # 不需要真正执行 compact，因为压缩逻辑在下面统一处理。
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    # 从工具注册表中查找处理函数。类似 Java 的 handlerMap.get(block.getName())。
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                        # handler(**block.input) 使用关键字参数解包调用。
                        # ** 操作符将字典解包为关键字参数，类似 Java 的方法参数传递。
                        # 如果找不到 handler，返回错误信息。
                        # Java 中等价：handler != null ? handler.apply(block.getInput()) : "Unknown tool"
                    except Exception as e:
                        # 捕获工具执行异常，避免 Agent 因单个工具失败而崩溃。
                        output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                # 打印工具名和输出（截断到 200 字符），方便调试。
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                # 构造 tool_result，追加到结果列表。
                # tool_use_id 用于将结果与原始工具调用关联。

        messages.append({"role": "user", "content": results})
        # 将所有工具结果作为 user 消息追加到对话中。
        # Claude API 要求 tool_result 的 role 必须是 "user"。

        # Layer 3: manual compact triggered by the compact tool
        # 第三层：如果 Agent 调用了 compact 工具，执行手动压缩。
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            # 再次使用切片赋值替换全部消息。
            # 与 auto_compact（第二层）使用完全相同的压缩逻辑。
            # 区别在于触发时机：auto_compact 是被动触发（token 超限），
            # manual_compact 是主动触发（Agent 自主判断需要压缩）。
            return
            # 压缩后直接返回，让外层循环重新开始（用压缩后的消息继续对话）。


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # Python 的入口判断。__name__ 是模块属性，直接运行时为 "__main__"。
    # 类似 Java 的 public static void main(String[] args)。
    history = []
    # 对话历史列表，跨多个 agent_loop 调用保持。
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
            # input() 读取用户输入，类似 Java 的 Scanner.nextLine()。
            # \033[36m 是 ANSI 转义码，设置终端文字为青色（cyan）。
            # \033[0m 重置颜色。类似 Java 的 ANSI_COLOR_CYAN + "s06 >> " + ANSI_RESET。
        except (EOFError, KeyboardInterrupt):
            # 捕获 Ctrl+D（EOF）和 Ctrl+C（中断），优雅退出。
            # 类似 Java 的 catch (Exception e) 但更精确。
            break
        if query.strip().lower() in ("q", "exit", ""):
            # 如果用户输入 q、exit 或空行，退出。
            # .strip().lower() 链式调用：先去空白再转小写。
            # in ("q", "exit", "") 检查是否在元组中，类似 Java 的 Set.of("q", "exit", "").contains()。
            break
        history.append({"role": "user", "content": query})
        # 将用户输入追加到对话历史。
        agent_loop(history)
        # 调用 Agent 循环。history 会被 agent_loop 中的压缩机制修改。
        response_content = history[-1]["content"]
        # history[-1] 获取最后一个元素（负索引），类似 Java 的 list.get(list.size() - 1)。
        if isinstance(response_content, list):
            # 如果最后一轮响应是列表（可能包含多个内容块）。
            for block in response_content:
                if hasattr(block, "text"):
                    # hasattr() 检查是否有 text 属性（过滤 TextBlock）。
                    print(block.text)
        print()
        # 打印空行，分隔对话。
