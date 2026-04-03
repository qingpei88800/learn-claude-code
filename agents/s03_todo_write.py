#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
"""

# -- 导入标准库（同 s01/s02 的公共基础设施部分）--
import os
import subprocess
from pathlib import Path

# -- 导入第三方库（同 s01/s02）--
from anthropic import Anthropic
from dotenv import load_dotenv

# -- 环境变量初始化（同 s01/s02）--
load_dotenv(override=True)

# -- 防止代理冲突（同 s01/s02）--
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# -- 全局常量（同 s02）--
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# -- System prompt，新增了 todo 使用指引 --
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# ============================================================
# -- TodoManager: s03 新增的核心类 --
# 让 LLM 通过 tool call 自行维护任务列表，
# 类似 Java 中的 TaskService / TaskManager 服务类
# ============================================================
# Python 的 class 定义：
#   - 不需要像 Java 那样声明访问修饰符（public/private/protected）
#   - 没有 extends 关键字，默认继承 object
#   - 所有方法都是公开的（Python 用 _ 前缀表示"约定私有"，但不强制）
class TodoManager:
    # __init__ 是 Python 的构造方法，等价于 Java 的构造函数
    # Python 没有 this 关键字，用 self 代替，且 self 必须显式写在参数列表第一个位置
    # Java 写法: public TodoManager() { this.items = new ArrayList<>(); }
    def __init__(self):
        # self.items 是实例属性，等价于 Java 的成员变量
        # Java 写法: private List<Map<String, String>> items = new ArrayList<>();
        # Python 的列表 [] 等价于 Java 的 ArrayList（动态数组）
        self.items = []

    # 实例方法，第一个参数永远是 self（类似 Java 的 this，但必须显式声明）
    # 参数类型注解 `items: list` -> 返回值注解 `-> str`
    # 类型注解在 Python 中是可选的，仅用于 IDE 提示和静态检查，运行时不强制
    # Java 写法: public String update(List<Map<String, Object>> items) { ... }
    def update(self, items: list) -> str:
        # len(items) 等价于 Java 的 items.size()
        # raise ValueError() 等价于 Java 的 throw new IllegalArgumentException()
        # Python 用 raise 抛异常，Java 用 throw
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")  # 限制最大数量，防止滥用
        # 创建一个新列表来存放校验后的任务项
        validated = []
        # 记录当前处于 in_progress 状态的任务数量
        in_progress_count = 0
        # enumerate(items) 同时获取索引 i 和值 item
        # 等价于 Java 的 for (int i = 0; i < items.size(); i++) { Map item = items.get(i); }
        for i, item in enumerate(items):
            # item.get("text", "") 是 dict 的 get 方法，带默认值
            # 等价于 Java: item.getOrDefault("text", "")
            # 如果 key 不存在，返回默认值 "" 而不是抛异常
            # str() 将值转为字符串，.strip() 去除首尾空白（类似 Java 的 String.trim()）
            text = str(item.get("text", "")).strip()
            # 字符串方法链调用：str() -> .lower()
            # 和 Java 的链式调用一样：value.toString().toLowerCase()
            # 默认值 "pending" 确保即使 LLM 没传 status，也能正常工作
            status = str(item.get("status", "pending")).lower()
            # id 也提供默认值：用当前索引 + 1（从 1 开始计数）
            # str(i + 1) 先转字符串再传入 get 作为默认值
            item_id = str(item.get("id", str(i + 1)))
            # not text 等价于 Java 的 text == null || text.isEmpty()
            # Python 中空字符串、None、0、空列表等都被视为 "falsy"
            if not text:
                # f-string 格式化字符串，等价于 Java 的 String.format("Item %s: text required", item_id)
                raise ValueError(f"Item {item_id}: text required")
            # status not in (...) 检查值是否在元组中
            # 等价于 Java: !Set.of("pending", "in_progress", "completed").contains(status)
            # 这里用的是元组 tuple () 而不是列表 []，元组是不可变的，性能略好
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            # append 等价于 Java 的 ArrayList.add()
            # 这里构建了一个新的 dict，确保只保留 id/text/status 三个字段
            validated.append({"id": item_id, "text": text, "status": status})
        # 业务约束：同一时刻只能有一个任务处于 "进行中" 状态
        # 防止 LLM 同时标记多个任务为 in_progress，保持执行焦点
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        # 将校验后的列表赋值给实例属性 self.items
        # Python 中赋值即引用，这里 validated 是一个新列表，所以是安全的
        self.items = validated
        # 调用实例方法 self.render() 将任务列表渲染为文本返回给 LLM
        return self.render()

    # render 方法：将内部状态渲染为人类可读的文本
    # 返回值是 str，LLM 会收到这个渲染结果作为 tool_result
    # Java 写法: public String render() { ... }
    def render(self) -> str:
        # not self.items 在 Python 中表示列表为空
        # 等价于 Java 的 this.items.isEmpty()
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            # dict 作为查找表使用：用 item["status"] 作为 key 取出对应的标记符号
            # Java 写法: Map.of("pending", "[ ]", "in_progress", "[>]", "completed", "[x]").get(item.get("status"))
            # [>] 表示当前正在进行中的任务，类似进度条的概念
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            # f-string 中直接嵌入变量和表达式，比 Java 的 String.format() 更直观
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        # 列表推导式（generator expression）配合 sum() 使用
        # sum(1 for t in self.items if t["status"] == "completed")
        # 等价于 Java Stream: items.stream().filter(t -> "completed".equals(t.get("status"))).count()
        # 这里的 for ... if 是一个生成器表达式，每匹配一次就产生一个 1，sum 求和即得总数
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        # "\n".join(lines) 将字符串列表用换行符连接成一个字符串
        # 等价于 Java 的 String.join("\n", lines)
        return "\n".join(lines)


# 在模块级别创建 TodoManager 的单例实例
# Python 中直接赋值即可创建全局对象，等价于 Java 的 TodoManager TODO = new TodoManager();
# 整个模块共享同一个 TODO 实例
TODO = TodoManager()


# -- Tool implementations（同 s02，新增了 todo 工具）--
# safe_path: 路径安全检查，防止路径穿越攻击（同 s02）
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

# run_bash: 执行 shell 命令（同 s01/s02）
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 三元表达式（条件表达式）：out[:50000] if out else "(no output)"
        # 等价于 Java 的三元运算符: out.isEmpty() ? "(no output)" : out.substring(0, Math.min(50000, out.length()))
        # 语法: 值A if 条件 else 值B -- 注意和 Java 的 条件 ? 值A : 值B 顺序不同
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

# run_read: 读取文件（同 s02）
def run_read(path: str, limit: int = None) -> str:
    # limit: int = None 表示参数有默认值 None，等价于 Java 的方法重载或 @Nullable Integer limit
    # Python 没有 null 关键字，用 None 代替
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

# run_write: 写入文件（同 s02）
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

# run_edit: 编辑文件，精确替换文本（同 s02）
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- TOOL_HANDLERS: 工具调度映射表（同 s02，新增 todo 条目）--
# Python 的 dict 可以存储 lambda 作为值，实现简单的策略模式
# 等价于 Java 中用 Map<String, Function<Map<String, Object>, String>> 实现
# lambda **kw: ... 中的 **kw 表示接收所有关键字参数并打包成 dict
# 等价于 Java 的 lambda Map<String, Object> kw -> ...
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),  # s03 新增: 调用 TodoManager 实例
}

# -- TOOLS: 工具定义列表（同 s02，新增 todo 工具定义）--
# 每个工具是一个 dict，描述工具名、描述、输入 schema（JSON Schema 格式）
# LLM 会根据这些定义决定何时调用哪个工具
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # s03 新增: todo 工具，让 LLM 可以管理任务列表
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# ============================================================
# -- Agent loop with nag reminder injection --
# s03 的核心改动：在 agent_loop 中新增了 nag reminder 机制
# 当 LLM 连续多轮没有调用 todo 工具时，自动注入提醒
# ============================================================
def agent_loop(messages: list):
    # rounds_since_todo: 距离上一次调用 todo 工具的轮次计数器
    # 用于判断 LLM 是否"忘记"更新任务列表
    # 等价于 Java 中的 int roundsSinceTodo = 0;
    rounds_since_todo = 0
    # while True: Python 的无限循环，等价于 Java 的 while (true) { ... }
    while True:
        # 调用 Claude API，同 s01/s02
        # Nag reminder 不在这里注入，而是在下面的 tool_result 中注入
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 将 assistant 的回复追加到消息历史，同 s01/s02
        messages.append({"role": "assistant", "content": response.content})
        # 如果模型没有调用工具（stop_reason != "tool_use"），说明任务完成，退出循环
        if response.stop_reason != "tool_use":
            return
        # 收集所有工具调用的结果
        results = []
        # 标记本轮是否有调用 todo 工具
        used_todo = False
        # 遍历 response.content 中的所有内容块，同 s02
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # handler(**block.input): 用 ** 解包 dict 为关键字参数传入
                    # 等价于 Java 中手动将 Map 的键值对逐个作为参数传入
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                # str(output)[:200]: 三元表达式的链式用法
                # 先转字符串，再截取前 200 字符，防止 LLM 输出过长影响调试
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                # 检查是否调用了 todo 工具
                if block.name == "todo":
                    used_todo = True
        # 三元表达式实现计数器重置或递增
        # 如果用了 todo -> 归零；否则 +1
        # 等价于 Java: roundsSinceTodo = usedTodo ? 0 : roundsSinceTodo + 1;
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        # nag reminder 机制：如果连续 3 轮没更新 todo，注入提醒
        # 这是以 "tool_result" 旁边的 "text" 类型消息注入的
        # LLM 会看到这条提醒，然后（大概率）去调用 todo 工具更新进度
        if rounds_since_todo >= 3:
            # 注意：这个 text 块伪装成 tool_result 的一部分被追加到 results 中
            # LLM 无法区分哪些是真实的 tool_result，哪些是系统注入的 text
            # 这种"在 tool_result 中夹带私货"的技巧是 agent 框架中常见的引导策略
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        # 将结果追加到消息历史，同 s01/s02
        # 但如果触发了 nag reminder，results 中会包含一个额外的 text 块
        messages.append({"role": "user", "content": results})


# -- 主入口（同 s01/s02）--
# if __name__ == "__main__": 是 Python 的惯用写法
# 等价于 Java 的 public static void main(String[] args) { ... }
# 当文件被直接运行时执行，被 import 时不执行
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        # isinstance() 检查对象类型，等价于 Java 的 instanceof
        # response_content 可能是 str（纯文本回复）或 list（包含 tool_use 块的回复）
        if isinstance(response_content, list):
            for block in response_content:
                # hasattr() 检查对象是否有指定属性，等价于 Java 的反射或 instanceof + 字段检查
                if hasattr(block, "text"):
                    print(block.text)
        print()
