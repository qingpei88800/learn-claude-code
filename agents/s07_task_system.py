#!/usr/bin/env python3
# 【Shebang 行】告诉操作系统用 python3 解释器来执行此脚本。
# 类似 Java 中虽然没有 shebang，但 Java 程序依赖 JVM 来运行，而 Python 脚本通过 shebang 指定解释器。
# Harness: persistent tasks -- goals that outlive any single conversation.
# 【持久化任务框架】—— 任务可以在多次对话之间持续存在，不会因为对话结束而丢失。

"""
s07_task_system.py - Tasks（任务系统）

【模块文档字符串】用三重双引号包裹，是 Python 的多行注释/文档字符串（docstring），
相当于 Java 中的 Javadoc 注释。Java 用 /* ... */ 包裹，Python 用三重引号 '''...''' 或 \"\"\"...\"\"\" 包裹。

Tasks persist as JSON files in .tasks/ so they survive context compression.
【核心设计思想】任务以 JSON 文件形式持久化存储在 .tasks/ 目录中，
因此可以跨越上下文压缩（context compression）而存活。
这类似于 Java 中将对象序列化为 JSON/XML 文件保存到磁盘，以便程序重启后数据不丢失。

Each task has a dependency graph (blockedBy).
【依赖图】每个任务都有一个依赖关系图，通过 blockedBy 字段表示。
相当于 Java 中的 DAG（有向无环图）任务调度，类似于 Gradle 的任务依赖系统。

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], ...}

    Dependency resolution:
    【依赖解析流程图】
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy
         【完成任务 1 后，会自动从 task 2 的 blockedBy 列表中移除 task 1 的 ID，
           这样 task 2 就不再是 "blocked" 状态，可以开始执行了】

Key insight: "State that survives compression -- because it's outside the conversation."
【关键洞察】"能够跨越上下文压缩的状态——因为它存在于对话之外。"
这意味着任务数据存储在文件系统中（类似于数据库），而不是存在内存中的对话历史里，
所以即使 AI 对话上下文被压缩/截断，任务数据也不会丢失。
"""

import json
# 【标准库导入】导入 Python 标准库的 json 模块，用于 JSON 序列化/反序列化。
# 相当于 Java 中 import org.json.*; 或使用 Jackson/Gson 库。

import os
# 【标准库导入】导入 os 模块，提供与操作系统交互的功能（环境变量、文件路径等）。
# 相当于 Java 中 java.lang.System 和 java.io.File 的组合。

import subprocess
# 【标准库导入】导入 subprocess 模块，用于创建子进程执行 shell 命令。
# 相当于 Java 中 Runtime.getRuntime().exec() 或 ProcessBuilder。

from pathlib import Path
# 【从模块导入】从 pathlib 模块导入 Path 类。Path 是 Python 3.4+ 提供的
# 面向对象的路径操作类，比传统的 os.path 更好用。
# 相当于 Java 中 java.nio.file.Path。
# Python 的 "from X import Y" 语法类似于 Java 的 import，但可以只导入模块中的特定类/函数。

from anthropic import Anthropic
# 【第三方库导入】从 anthropic SDK 导入 Anthropic 客户端类。
# 相当于 Java 中 import com.anthropic.client.Anthropic;
# 这需要先 pip install anthropic（类似 Java 的 Maven/Gradle 依赖管理）。

from dotenv import load_dotenv
# 【第三方库导入】从 python-dotenv 库导入 load_dotenv 函数。
# 用于从 .env 文件加载环境变量，类似于 Java Spring Boot 中的 application.properties。

load_dotenv(override=True)
# 【函数调用】加载 .env 文件中的环境变量。
# override=True 表示如果环境变量已存在，用 .env 中的值覆盖。
# 相当于 Java Spring Boot 中 @PropertySource 注解读取配置文件。

if os.getenv("ANTHROPIC_BASE_URL"):
    # 【条件语句】os.getenv() 获取环境变量值，如果不存在返回 None（相当于 Java 的 System.getenv()）。
    # 这行检查是否设置了自定义的 API 基础 URL（比如使用代理或私有部署）。
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    # 【字典操作】从环境变量字典中移除指定的 key。
    # os.environ 是一个类似字典（dict）的对象，存储了所有环境变量。
    # .pop(key, default) 方法：如果 key 存在则移除并返回值，否则返回 default（这里设为 None）。
    # 相当于 Java 中 System.getProperties().remove("ANTHROPIC_AUTH_TOKEN")。
    # 这是为了避免在使用自定义 API URL 时认证令牌冲突的问题。

# ========== 全局常量定义 ==========
# Python 没有真正的常量关键字（Java 的 final），但约定用全大写命名表示常量。

WORKDIR = Path.cwd()
# 【常量】WORKDIR = 当前工作目录。Path.cwd() 获取当前工作目录路径对象。
# 相当于 Java 中 Path currentDir = Path.of(".").toAbsolutePath();

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 【全局变量】创建 Anthropic API 客户端实例。
# base_url 参数可选，如果环境变量中设置了则使用自定义 URL，否则使用默认的官方 API 地址。
# 相当于 Java 中：AnthropicClient client = AnthropicClient.builder()
#     .baseUrl(System.getenv("ANTHROPIC_BASE_URL")).build();
# 注意：Python 中 class 实例化不需要 new 关键字（Java 需要 new Anthropic(...)）。

MODEL = os.environ["MODEL_ID"]
# 【常量】从环境变量获取要使用的模型 ID（如 "claude-sonnet-4-20250514"）。
# os.environ["KEY"] 使用方括号访问，如果 KEY 不存在会抛出 KeyError（类似 Java 的 HashMap）。
# 与 os.getenv() 不同，getenv() 在 key 不存在时返回 None 而不抛异常。

TASKS_DIR = WORKDIR / ".tasks"
# 【路径拼接】使用 "/" 运算符拼接路径，这是 Path 类重载了 __truediv__ 运算符。
# 在 Java 中需要用 Path.resolve()：Path tasksDir = WORKDIR.resolve(".tasks");
# Python 的 Path 类支持 "/" 运算符进行路径拼接，更加简洁直观。

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."
# 【f-string 格式化字符串】f"..." 是 Python 3.6+ 引入的格式化字符串语法。
# 花括号 {WORKDIR} 中可以嵌入任意 Python 表达式，运行时会被替换为表达式的值。
# 相当于 Java 中的 "You are a coding agent at " + WORKDIR + ". Use task tools..."
# 或 Java 15+ 的文本块："""
#   You are a coding agent at %s. Use task tools...""".formatted(WORKDIR)


# -- TaskManager: CRUD with dependency graph, persisted as JSON files --
# 【设计模式说明】TaskManager 类实现了 CRUD（Create/Read/Update/Delete）操作，
# 并管理任务之间的依赖关系图。数据以 JSON 文件形式持久化。
# 这类似于 Java 中的 Repository 模式 + Service 层的组合，
# 同时融入了图结构（依赖关系管理）的概念。

class TaskManager:
    # 【类定义】Python 使用 class 关键字定义类，与 Java 相同。
    # 注意：Python 类定义不需要访问修饰符（public/private/protected），
    # Python 通过命名约定来表示可见性：以 _ 开头的成员表示"受保护的"（约定而非强制），
    # 以 __ 开头表示"私有的"（名称修饰，name mangling）。
    # 所有方法默认是 public 的。

    def __init__(self, tasks_dir: Path):
        # 【构造方法】__init__ 是 Python 的实例初始化方法，相当于 Java 的构造函数。
        # Python 中所有实例方法的第一个参数必须是 self，它引用当前实例对象，
        # 类似于 Java 中的 this 关键字，但 Python 要求显式声明。
        # tasks_dir: Path 是类型提示（Type Hint），告诉开发者期望传入 Path 类型的参数。
        # 类型提示是可选的，Python 运行时不会强制检查（不同于 Java 的强类型系统）。

        self.dir = tasks_dir
        # 【实例属性赋值】self.dir = tasks_dir，将传入参数保存为实例属性。
        # 在 Java 中：this.dir = tasksDir;
        # Python 不需要提前声明实例属性，赋值时自动创建。

        self.dir.mkdir(exist_ok=True)
        # 【创建目录】如果 .tasks 目录不存在则创建它。
        # exist_ok=True 表示如果目录已存在不抛出异常。
        # 相当于 Java 中 Files.createDirectories(path)；

        self._next_id = self._max_id() + 1
        # 【实例属性】_next_id 记录下一个可用的任务 ID（自增序列）。
        # 以 _ 开头表示这是"受保护"的内部属性（命名约定）。
        # 通过扫描现有文件中的最大 ID + 1 来计算，确保重启后 ID 不冲突。
        # 类似于 Java 中数据库的自增主键（AUTO_INCREMENT）。

    def _max_id(self) -> int:
        # 【私有方法】以 _ 开头表示内部方法（命名约定，非强制）。
        # -> int 是返回值类型提示，告诉开发者此方法返回 int 类型。
        # 相当于 Java 中的 private int maxId()。

        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        # 【列表推导式（List Comprehension）】这是 Python 最具特色的语法之一！
        # 完整展开等价于：
        #   ids = []
        #   for f in self.dir.glob("task_*.json"):
        #       ids.append(int(f.stem.split("_")[1]))
        #
        # 逐部分解析：
        #   self.dir.glob("task_*.json") - 遍历目录下所有匹配 task_*.json 模式的文件。
        #     类似 Java 中 PathMatcher 或 Files.list() + filter。
        #   f.stem - 获取文件名（不含扩展名），例如 "task_1"。
        #     类似 Java 中 getFileName().toString().replace(".json", "")。
        #   .split("_")[1] - 按下划线分割取第二部分，即 ID 数字字符串。
        #     类似 Java 中 String.split("_")[1]。
        #   int(...) - 将字符串转换为整数。
        #     类似 Java 中 Integer.parseInt(...)。
        #
        # Java 中等价写法：
        #   List<Integer> ids = Files.list(tasksDir.toPath())
        #       .filter(p -> p.toString().matches("task_\\d+\\.json"))
        #       .map(p -> Integer.parseInt(p.getFileName().toString().split("_")[1]))
        #       .collect(Collectors.toList());

        return max(ids) if ids else 0
        # 【三元表达式】Python 的三元表达式语法是：值A if 条件 else 值B。
        # 如果 ids 列表非空，返回最大值；否则返回 0。
        # Java 中等价写法：return ids.isEmpty() ? 0 : Collections.max(ids);

    def _load(self, task_id: int) -> dict:
        # 【加载任务】从 JSON 文件中读取任务数据。
        # -> dict 表示返回一个字典（dict）类型。
        # Python 的 dict 类似于 Java 的 HashMap<String, Object> 或 JSONObject。

        path = self.dir / f"task_{task_id}.json"
        # 【路径拼接 + f-string】拼接出任务文件的完整路径，例如 ".tasks/task_1.json"。

        if not path.exists():
            # 【文件存在性检查】path.exists() 检查文件是否存在。
            # 相当于 Java 中 Files.exists(path)。

            raise ValueError(f"Task {task_id} not found")
            # 【抛出异常】Python 使用 raise 关键字抛出异常，Java 使用 throw。
            # ValueError 是 Python 内置异常类，类似于 Java 的 IllegalArgumentException。

        return json.loads(path.read_text())
        # 【读取并解析 JSON】
        # path.read_text() - 读取文件全部内容为字符串（自动处理编码）。
        #   相当于 Java 中 Files.readString(path)。
        # json.loads(s) - 将 JSON 字符串解析为 Python 字典（dict）。
        #   loads = "load string"，相当于 Java 中 new JSONObject(s) 或 Jackson 的 objectMapper.readValue()。
        #   注意：Python 的 json.loads() 返回的是 dict（字典），不是自定义对象。
        #   在 Java 中通常需要用 @Data class + Jackson 来映射为 POJO。

    def _save(self, task: dict):
        # 【保存任务】将任务字典序列化为 JSON 并写入文件。
        # task: dict 参数类型提示，表示期望传入一个字典。

        path = self.dir / f"task_{task['id']}.json"
        # 【字典访问】task['id'] 使用方括号 + key 访问字典中的值。
        # 相当于 Java 中 task.get("id") 或 task.getString("id")。

        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
        # 【序列化并写入文件】
        # json.dumps(task, ...) - 将 Python 字典序列化为 JSON 字符串。
        #   dumps = "dump string"（与 loads 对应）。
        #   indent=2 - 缩进 2 个空格，使 JSON 可读（美化输出）。
        #   ensure_ascii=False - 允许非 ASCII 字符（如中文）直接输出，而不是转义为 \uXXXX。
        #   相当于 Java 中 objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(task)。
        # path.write_text(...) - 将字符串写入文件（覆盖原有内容）。
        #   相当于 Java 中 Files.writeString(path, content)。

    def create(self, subject: str, description: str = "") -> str:
        # 【公开方法 - 创建任务】
        # subject: str - 任务标题，类型提示为字符串。
        # description: str = "" - 可选参数，默认值为空字符串。
        #   Python 支持默认参数值，类似 Java 中方法重载提供默认值。
        #   Java 中需要用方法重载实现：create(String subject) 和 create(String subject, String description)。
        #   Python 直接在参数声明中指定默认值即可，更加简洁。
        # -> str - 返回值为 JSON 字符串。

        task = {
            # 【字典字面量】用花括号 {} 创建字典（dict），类似于 Java 中：
            #   Map<String, Object> task = new LinkedHashMap<>();
            #   task.put("id", this._nextId);
            #   task.put("subject", subject);
            #   ...
            # Python 字典字面量比 Java 的 Map 构建简洁得多。

            "id": self._next_id, "subject": subject, "description": description,
            # 【字典键值对】多个键值对可以写在一行内，用逗号分隔。
            # 注意：self._next_id 不需要加分号（Python 用换行分隔语句，不需要分号）。

            "status": "pending", "blockedBy": [], "owner": "",
            # 【初始值设定】
            # status: "pending" - 初始状态为"待处理"。
            # blockedBy: [] - 空列表，表示没有阻塞依赖。
            #   [] 是 Python 的空列表字面量，相当于 Java 的 List.of() 或 new ArrayList<>()。
            # owner: "" - 空字符串，表示尚未分配负责人。
        }
        # 字典创建完毕，task 变量引用这个字典对象。

        self._save(task)
        # 【调用内部方法】将任务保存到文件系统。

        self._next_id += 1
        # 【自增 ID】递增下一个可用 ID。
        # 类似于 Java 中 this._nextId++;（Python 没有 ++ 运算符，使用 += 1）。

        return json.dumps(task, indent=2, ensure_ascii=False)
        # 【返回 JSON 字符串】将创建的任务对象序列化为格式化的 JSON 返回。

    def get(self, task_id: int) -> str:
        # 【公开方法 - 获取任务】根据任务 ID 获取任务的 JSON 表示。
        # 相当于 Java 中的 String getTask(int taskId)。

        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)
        # 【加载并序列化】_load() 返回 dict，json.dumps() 将 dict 转为 JSON 字符串。
        # Java 等价写法：return objectMapper.writeValueAsString(loadTask(taskId));

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        # 【公开方法 - 更新任务】更新任务的状态或依赖关系。
        # status: str = None - 默认值为 None（Python 的 null）。
        #   注意 Python 用 None 表示空值，Java 用 null。
        #   None 是 Python 的单例对象，类似于 Java 的 null 引用。
        # add_blocked_by: list = None - 要添加的阻塞依赖 ID 列表。
        # remove_blocked_by: list = None - 要移除的阻塞依赖 ID 列表。
        # Python 允许一行方法签名跨多行书写（用圆括号包裹），Java 也允许。
        # 在 Java 中，这样的多可选参数通常用 Builder 模式实现。

        task = self._load(task_id)
        # 【加载现有任务】先从文件中读取当前任务数据。

        if status:
            # 【真值判断】Python 中 if status: 等价于 if status is not None and status != "":
            # Python 的真值（truthiness）规则：None、空字符串 ""、0、空列表 []、空字典 {} 都为 False。
            # 这比 Java 更灵活，Java 中 if (status) 只能用于 boolean，字符串需要 if (status != null && !status.isEmpty())。

            if status not in ("pending", "in_progress", "completed"):
                # 【成员判断】not in 运算符检查值是否不在元组中。
                # ("pending", "in_progress", "completed") 是一个元组（tuple），用圆括号表示。
                # 元组是 Python 的不可变序列，类似于 Java 中 List.of("pending", "in_progress", "completed")。
                # Java 中等价写法：if (!List.of("pending", "in_progress", "completed").contains(status))

                raise ValueError(f"Invalid status: {status}")
                # 【抛出异常】使用 f-string 将变量嵌入错误消息。

            task["status"] = status
            # 【更新字典值】直接通过 key 赋值更新字典中的值。
            # 相当于 Java 中 task.put("status", status);。

            if status == "completed":
                # 【状态联动】当任务标记为"已完成"时，需要清理依赖关系。
                self._clear_dependency(task_id)
                # 【级联更新】清除其他任务对此任务的依赖。
                # 这是一个重要的设计：完成任务会自动解除下游任务的阻塞状态。
                # 类似于 Java 中观察者模式（Observer Pattern）或事件驱动机制。

        if add_blocked_by:
            # 【添加依赖】如果传入了 add_blocked_by 列表。

            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
            # 【列表操作 + 去重】
            # task["blockedBy"] + add_blocked_by - 两个列表拼接，产生新列表。
            #   Python 的 + 运算符可以拼接列表，相当于 Java 中 Stream.concat(list1.stream(), list2.stream()).collect()。
            # set(...) - 将列表转换为集合（set），自动去重。
            #   Python 的 set 类似于 Java 的 HashSet。
            # list(...) - 将集合转换回列表。
            # 整体效果：合并两个列表并去除重复元素。
            # Java 中等价写法：
            #   List<Integer> merged = Stream.concat(
            #       task.getBlockedBy().stream(), addBlockedBy.stream()
            #   ).distinct().collect(Collectors.toList());

        if remove_blocked_by:
            # 【移除依赖】如果传入了 remove_blocked_by 列表。

            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
            # 【列表推导式 - 过滤】从 blockedBy 列表中移除指定的 ID。
            # [x for x in task["blockedBy"] if x not in remove_blocked_by] 的含义：
            #   遍历 task["blockedBy"] 中的每个元素 x，
            #   如果 x 不在 remove_blocked_by 列表中，则保留该元素。
            # 相当于 Java 中的 Stream 过滤：
            #   task.getBlockedBy().stream()
            #       .filter(x -> !removeBlockedBy.contains(x))
            #       .collect(Collectors.toList());

        self._save(task)
        # 【持久化】将更新后的任务数据保存回文件。

        return json.dumps(task, indent=2, ensure_ascii=False)
        # 【返回更新后的 JSON】

    def _clear_dependency(self, completed_id: int):
        # 【级联清除依赖】当一个任务完成时，从所有其他任务的 blockedBy 列表中移除它。
        # 这是一个私有辅助方法（以 _ 开头）。
        # 相当于 Java 中的级联更新操作（在关系型数据库中类似 ON DELETE CASCADE）。

        """Remove completed_id from all other tasks' blockedBy lists."""
        # 【方法文档字符串】描述方法的功能，相当于 Java 的 Javadoc。

        for f in self.dir.glob("task_*.json"):
            # 【遍历文件】使用 glob 模式匹配遍历所有任务文件。
            # 相当于 Java 中 Files.list(dir).filter(p -> p matches "task_*.json").forEach(...)

            task = json.loads(f.read_text())
            # 【读取并解析】读取每个任务文件的 JSON 内容。

            if completed_id in task.get("blockedBy", []):
                # 【检查依赖关系】
                # task.get("blockedBy", []) - 字典的 get 方法，如果 key 不存在返回默认值 []。
                #   相当于 Java 中 task.getOrDefault("blockedBy", List.of())。
                # "in" 运算符 - 检查元素是否在列表中。
                #   相当于 Java 中 list.contains(completedId)。

                task["blockedBy"].remove(completed_id)
                # 【移除元素】从列表中移除指定元素（按值删除，不是按索引）。
                # 相当于 Java 中 list.remove(Integer.valueOf(completedId))。

                self._save(task)
                # 【保存修改】将被修改的任务写回文件。

    def list_all(self) -> str:
        # 【公开方法 - 列出所有任务】返回格式化的任务列表字符串。

        tasks = []
        # 【初始化空列表】用于收集所有任务对象。

        files = sorted(
            # 【排序函数调用】sorted() 是 Python 内置函数，返回排序后的新列表（不修改原列表）。
            # 相当于 Java 中 list.stream().sorted(...).collect(Collectors.toList())。

            self.dir.glob("task_*.json"),
            # 【要排序的序列】所有匹配的文件。

            key=lambda f: int(f.stem.split("_")[1])
            # 【排序键函数】key 参数接受一个函数，用于提取排序依据。
            # lambda f: ... 是 Python 的匿名函数（Lambda 表达式），相当于 Java 的 lambda。
            #   lambda f: int(f.stem.split("_")[1])
            #   等价于 Java 中：(Path f) -> Integer.parseInt(f.getFileName().toString().split("_")[1])
            # 这里按文件名中的数字 ID 排序。
        )

        for f in files:
            # 【遍历排序后的文件列表】

            tasks.append(json.loads(f.read_text()))
            # 【读取并收集】将每个文件解析为字典后添加到列表中。
            # 相当于 Java 中 tasks.add(parseJson(readText(f)));

        if not tasks:
            # 【空列表检查】如果没有任何任务。

            return "No tasks."
            # 【返回提示信息】

        lines = []
        # 【初始化空列表】用于构建输出行。

        for t in tasks:
            # 【遍历任务列表】

            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            # 【状态标记映射】
            # {"pending": "[ ]", ...} - 字典字面量，定义状态到标记的映射。
            #   类似于 Java 中 Map.of("pending", "[ ]", "in_progress", "[>]", "completed", "[x]")。
            # .get(t["status"], "[?]") - 根据状态查找标记，找不到则返回默认值 "[?]"。
            #   类似于 Java 中 map.getOrDefault(status, "[?]")。

            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            # 【条件表达式 + f-string】
            # 如果任务有阻塞依赖（列表非空），则拼接阻塞信息；否则为空字符串。
            # Python 中空列表 [] 的布尔值为 False，所以 t.get("blockedBy") 如果返回空列表则为 False。
            # f"..." 中的 {t['blockedBy']} 会将列表转换为字符串表示形式，如 "[1, 2]"。

            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
            # 【格式化输出行】拼接标记、ID、标题和阻塞信息。
            # 例如："[ ] #3: 实现登录功能 (blocked by: [1, 2])"

        return "\n".join(lines)
        # 【字符串连接】"\n".join(lines) 用换行符连接列表中的所有字符串。
        # 相当于 Java 中 String.join("\n", lines)。
        # 注意：Python 中是分隔符.join(列表)，Java 中是 String.join(分隔符, 列表)，顺序相反！


TASKS = TaskManager(TASKS_DIR)
# 【创建全局单例】在模块加载时创建 TaskManager 的全局实例。
# Python 中模块级别的变量类似于 Java 中 static final 变量。
# 这里是单例模式（Singleton Pattern）的最简实现——直接在模块级别创建实例。
# Java 中通常需要用 private static final + private 构造函数 + getInstance() 方法来实现单例。
# Python 由于没有访问修饰符，直接在模块级别创建实例就足够了。


# -- Base tool implementations --
# 【工具函数实现区】以下函数是 Agent 可以调用的各种"工具"的实现。
# 这些工具函数的设计模式类似于策略模式（Strategy Pattern），
# 每个函数封装了一个独立的工具行为，后面通过字典（TOOL_HANDLERS）进行注册和分发。

def safe_path(p: str) -> Path:
    # 【路径安全检查函数】确保文件路径不逃逸出工作目录。
    # 这是一个安全措施，防止 AI Agent 访问或修改工作目录之外的文件。
    # 类似于 Java 中的 SecurityManager 或沙箱机制。

    path = (WORKDIR / p).resolve()
    # 【路径解析】
    # WORKDIR / p - 拼接路径。
    # .resolve() - 解析为绝对路径，处理 ".." 等相对路径组件。
    #   例如："/project/../../../etc/passwd" 会被解析为 "/etc/passwd"。
    #   相当于 Java 中 path.toAbsolutePath().normalize()。

    if not path.is_relative_to(WORKDIR):
        # 【安全检查】is_relative_to() 检查路径是否在 WORKDIR 下。
        # 如果路径试图逃逸到工作目录之外（例如使用 "../../../etc/passwd"），则拒绝。
        # 这类似于 Java 中检查 path.startsWith(workDir)。

        raise ValueError(f"Path escapes workspace: {p}")
        # 【抛出异常】拒绝不安全的路径。

    return path
    # 【返回安全路径】

def run_bash(command: str) -> str:
    # 【执行 Bash 命令】封装了 subprocess 调用，带危险命令检测和超时控制。
    # 相当于 Java 中封装 ProcessBuilder 的工具方法。

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # 【危险命令黑名单】列表中存储了不允许执行的命令片段。

    if any(d in command for d in dangerous):
        # 【any() 函数】检查是否有任何一个危险命令片段出现在用户输入中。
        # any(生成器表达式) - 如果生成器中有任何一个为 True，则返回 True。
        # "d in command" - 字符串包含检查，相当于 Java 中 command.contains(d)。
        # 等价于 Java 中：dangerous.stream().anyMatch(command::contains)

        return "Error: Dangerous command blocked"
        # 【拒绝执行】返回错误信息而非抛异常，方便 Agent 处理。

    try:
        # 【try 块】与 Java 的 try 块完全相同，用于异常捕获。

        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        # 【执行子进程】
        # subprocess.run() - 执行外部命令并等待完成。
        #   相当于 Java 中 ProcessBuilder.start().waitFor()。
        # command - 要执行的命令字符串。
        # shell=True - 通过系统 shell 执行命令（允许使用管道、重定向等 shell 特性）。
        # cwd=WORKDIR - 设置工作目录（Change Working Directory）。
        #   相当于 Java 中 ProcessBuilder.directory(workDir)。
        # capture_output=True - 捕获标准输出和标准错误。
        #   相当于 Java 中 process.getInputStream() 和 process.getErrorStream()。
        # text=True - 将输出作为文本（字符串）而非字节返回，自动处理编码。
        #   相当于 Java 中使用 BufferedReader 读取字符流。
        # timeout=120 - 超时时间 120 秒，超时后抛出 subprocess.TimeoutExpired 异常。
        #   相当于 Java 中 Process.waitFor(120, TimeUnit.SECONDS)。

        out = (r.stdout + r.stderr).strip()
        # 【合并输出】将标准输出和标准错误拼接，并去除首尾空白字符。
        # r.stdout - 标准输出字符串。r.stderr - 标准错误字符串。
        # .strip() - 去除字符串首尾的空白字符（空格、换行、制表符等）。
        #   相当于 Java 中 String.trim()。

        return out[:50000] if out else "(no output)"
        # 【截断输出】out[:50000] 是 Python 的切片（slice）语法，取前 50000 个字符。
        #   Python 切片语法：sequence[start:end]，start 默认 0，end 可以省略。
        #   相当于 Java 中 out.substring(0, Math.min(50000, out.length()))。
        # 如果输出为空字符串，返回提示信息。

    except subprocess.TimeoutExpired:
        # 【捕获超时异常】类似于 Java 的 catch (TimeoutException e)。

        return "Error: Timeout (120s)"
        # 【返回超时错误信息】

def run_read(path: str, limit: int = None) -> str:
    # 【读取文件内容】安全地读取文件内容，支持行数限制。
    # limit: int = None - 可选参数，限制读取的最大行数。
    #   默认值为 None（不是 0），None 表示"无限制"。

    try:
        lines = safe_path(path).read_text().splitlines()
        # 【链式调用】
        # safe_path(path) - 先进行路径安全检查。
        # .read_text() - 读取文件全部内容为字符串。
        # .splitlines() - 按行分割为列表，去除换行符。
        #   相当于 Java 中 Files.readAllLines(path)。

        if limit and limit < len(lines):
            # 【限制行数】只有当 limit 非空（非 None）且小于总行数时才截断。
            # Python 中 None 的布尔值为 False，所以 if None: 不会执行。

            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
            # 【截断并添加提示】
            # lines[:limit] - 切片取前 limit 行。
            # + [f"... ({len(lines) - limit} more)"] - 列表拼接，添加截断提示行。
            # len(lines) - Python 内置函数，获取列表长度。相当于 Java 中 list.size()。

        return "\n".join(lines)[:50000]
        # 【合并并截断】将行列表用换行符合并，截断到 50000 字符。

    except Exception as e:
        # 【捕获所有异常】Python 的 Exception 类似于 Java 的 Exception（所有异常的基类）。
        # as e - 将异常对象赋值给变量 e。相当于 Java 的 catch (Exception e)。

        return f"Error: {e}"
        # 【返回错误信息】f-string 中嵌入异常对象会自动调用其 __str__() 方法。
        # 相当于 Java 中 "Error: " + e.getMessage()。

def run_write(path: str, content: str) -> str:
    # 【写入文件】安全地将内容写入文件。

    try:
        fp = safe_path(path)
        # 【路径安全检查】

        fp.parent.mkdir(parents=True, exist_ok=True)
        # 【创建父目录】
        # fp.parent - 获取文件的父目录路径。
        #   相当于 Java 中 path.getParent()。
        # mkdir(parents=True, exist_ok=True) - 递归创建所有缺失的父目录。
        #   parents=True - 递归创建，类似 Java 的 Files.createDirectories()。
        #   exist_ok=True - 目录已存在时不报错。
        #   相当于 Java 中 Files.createDirectories(fp.getParent())。

        fp.write_text(content)
        # 【写入文件】将字符串内容写入文件（如果文件存在则覆盖）。
        # 相当于 Java 中 Files.writeString(fp, content)。

        return f"Wrote {len(content)} bytes"
        # 【返回操作结果】

    except Exception as e:
        # 【异常捕获】

        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    # 【编辑文件】在文件中进行精确的文本替换（只替换第一个匹配）。
    # 这是一种简单的文件编辑方式，类似于 Java 中的字符串替换。

    try:
        fp = safe_path(path)
        # 【路径安全检查】

        c = fp.read_text()
        # 【读取文件内容】

        if old_text not in c:
            # 【检查目标文本是否存在】如果 old_text 不在文件内容中。

            return f"Error: Text not found in {path}"
            # 【返回错误】目标文本未找到。

        fp.write_text(c.replace(old_text, new_text, 1))
        # 【替换并写入】
        # str.replace(old, new, count) - 字符串替换方法。
        #   参数 1 表示只替换第一个匹配项。
        #   相当于 Java 中 c.replaceFirst(Pattern.quote(oldText), newText)。
        # 注意：Python 的 replace() 方法第三个参数是替换次数，
        # 而 Java 的 replaceAll() 没有"只替换第 N 个"的选项，需要用 replaceFirst()。

        return f"Edited {path}"
        # 【返回成功信息】

    except Exception as e:
        # 【异常捕获】

        return f"Error: {e}"


# ========== 工具注册与分发系统 ==========
# 【设计模式】这里使用了命令模式（Command Pattern）+ 策略模式（Strategy Pattern）的思想。
# TOOL_HANDLERS 字典充当工具注册表（Registry），将工具名称映射到对应的处理函数。
# 这类似于 Java 中的 Map<String, ToolHandler> 接口 + Lambda 实现。

TOOL_HANDLERS = {
    # 【工具处理函数映射表】字典的 key 是工具名称（字符串），value 是 Lambda 函数。
    # 这种注册式设计使得添加新工具非常简单——只需在字典中添加一行。
    # 类似于 Java Spring 中的 @Bean 注册或 Map 注入。

    "bash":        lambda **kw: run_bash(kw["command"]),
    # 【Lambda 表达式】lambda **kw: ... 是 Python 的匿名函数。
    #   **kw 是可变关键字参数（variable keyword arguments），表示接收任意数量的命名参数，
    #   并将它们打包成一个字典 kw。类似于 Java 中接收 Map<String, Object> 参数。
    #   调用时：handler(command="ls -la")  =>  kw = {"command": "ls -la"}
    #   kw["command"] - 从字典中取出 "command" 的值，传给 run_bash 函数。
    # 整体效果：当工具名为 "bash" 时，调用 run_bash(command=kw["command"])。

    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    # 【read_file 工具】调用 run_read 函数。kw.get("limit") 可能返回 None（如果参数未提供）。

    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    # 【write_file 工具】调用 run_write 函数。

    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 【edit_file 工具】调用 run_edit 函数。

    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    # 【task_create 工具】调用全局 TaskManager 实例的 create 方法。

    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    # 【task_update 工具】调用 update 方法，所有参数都是可选的（通过 kw.get() 获取）。
    # 注意：这里使用了驼峰命名 addBlockedBy/removeBlockedBy，是为了与 Claude API 的 JSON Schema 保持一致。

    "task_list":   lambda **kw: TASKS.list_all(),
    # 【task_list 工具】调用 list_all 方法，不需要额外参数。

    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
    # 【task_get 工具】根据 ID 获取任务详情。
}

TOOLS = [
    # 【工具定义列表】定义了所有可供 Claude AI 调用的工具的元数据。
    # 每个工具是一个字典，包含 name（名称）、description（描述）、input_schema（输入 JSON Schema）。
    # 这个列表会被传递给 Claude API，让 AI 知道有哪些工具可以使用以及如何调用它们。
    # 相当于 Java 中定义一组 ToolDescriptor 对象的列表。
    # 注意：TOOLS 定义了工具的"接口规范"（Schema），
    # 而 TOOL_HANDLERS 定义了工具的"实际实现"（Implementation），
    # 这种"声明与实现分离"的设计类似于 Java 中接口与实现类的关系。

    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    # 【bash 工具定义】
    # name - 工具名称，Agent 调用时使用此名称。
    # description - 工具描述，帮助 AI 理解何时使用此工具。
    # input_schema - JSON Schema 格式的输入参数定义。
    #   "type": "object" - 输入是一个对象。
    #   "properties" - 定义对象的各个属性及其类型。
    #   "required" - 必填的属性列表。
    # 这种 Schema 定义方式与 Java 中的 @JsonProperty + @NotNull 注解或 OpenAPI 规范类似。

    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    # 【read_file 工具定义】path 是必填参数，limit 是可选参数（不在 required 中）。

    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    # 【write_file 工具定义】path 和 content 都是必填参数。

    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 【edit_file 工具定义】三个参数都是必填的。

    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    # 【task_create 工具定义】只有 subject 是必填的，description 是可选的。

    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    # 【task_update 工具定义】
    # "enum": [...] - 限制字符串只能是枚举值之一（类似 Java 的 Enum 类型）。
    # "array": {"items": {"type": "integer"}} - 定义整数数组类型。
    #   相当于 Java 中 List<Integer>。

    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    # 【task_list 工具定义】没有输入参数（properties 为空对象）。

    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    # 【task_get 工具定义】需要一个整数类型的 task_id 参数。
]


def agent_loop(messages: list):
    # 【Agent 主循环函数】实现了一个完整的 ReAct（Reasoning + Acting）循环。
    # messages: list - 对话历史列表，每个元素是一个字典 {"role": "user/assistant", "content": ...}。
    # 这是 Claude API 要求的消息格式，类似于 Java 中 List<Map<String, Object>>。

    while True:
        # 【无限循环】类似于 Java 中的 while (true)。

        response = client.messages.create(
            # 【调用 Claude API】向 Claude 发送消息请求。
            # client.messages.create() 调用 Claude 的 Messages API。
            # 相当于 Java 中：Response response = client.messages().create(request);

            model=MODEL,
            # 【模型选择】使用环境变量指定的模型。

            system=SYSTEM,
            # 【系统提示词】设定 AI 的角色和行为准则。
            # 相当于 Java 中 request.setSystemPrompt(SYSTEM)。

            messages=messages,
            # 【对话历史】传入完整的对话历史，让 AI 能理解上下文。

            tools=TOOLS,
            # 【可用工具】传入工具定义列表，让 AI 知道可以调用哪些工具。

            max_tokens=8000,
            # 【最大输出 token 数】限制 AI 单次响应的最大长度，避免过长的响应。
            # 相当于 Java 中 request.setMaxTokens(8000)。
        )

        messages.append({"role": "assistant", "content": response.content})
        # 【保存助手响应】将 AI 的响应添加到对话历史中。
        # response.content 是 AI 的响应内容（可能包含文本和工具调用）。
        # 列表的 append() 方法在末尾添加元素，相当于 Java 中 list.add(element)。

        if response.stop_reason != "tool_use":
            # 【检查停止原因】stop_reason 表示 AI 停止生成的原因。
            # "tool_use" - AI 想要调用工具（还没说完，需要执行工具后继续）。
            # "end_turn" - AI 认为对话已完成，不需要再继续。
            # 如果 AI 不需要调用工具（即生成完毕），则退出循环。

            return
            # 【退出函数】return 无返回值，相当于 Java 中 return;。

        results = []
        # 【初始化工具调用结果列表】

        for block in response.content:
            # 【遍历响应块】response.content 是一个列表，可能包含多个内容块。
            # 每个块可能是文本块（TextBlock）或工具调用块（ToolUseBlock）。
            # 类似于 Java 中遍历 List<ContentBlock>。

            if block.type == "tool_use":
                # 【检查是否为工具调用块】

                handler = TOOL_HANDLERS.get(block.name)
                # 【查找处理函数】根据工具名称从注册表中查找对应的处理函数。
                # .get() 方法如果 key 不存在返回 None，而不是抛异常。
                #   相当于 Java 中 map.get(block.getName())。

                try:
                    # 【异常捕获块】

                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    # 【执行工具调用】
                    # handler(**block.input) - 解包字典作为关键字参数调用函数。
                    #   ** 运算符将字典解包为关键字参数（keyword arguments）。
                    #   例如：block.input = {"command": "ls"}，则 **block.input 展开为 command="ls"。
                    #   最终等价于：handler(command="ls")。
                    #   这类似于 Java 中通过反射将 Map 转换为方法参数。
                    # if handler else ... - Python 的三元表达式，如果 handler 为 None 则返回错误信息。

                except Exception as e:
                    # 【捕获工具执行异常】

                    output = f"Error: {e}"
                    # 【返回错误信息】将异常信息作为工具输出返回给 AI。

                print(f"> {block.name}:")
                # 【打印工具调用信息】在终端显示正在执行的工具名称。
                # f"> {block.name}:" - 格式化输出，> 表示这是工具调用。
                # 相当于 Java 中 System.out.println("> " + block.getName() + ":");

                print(str(output)[:200])
                # 【打印工具输出】将输出截断到 200 字符后打印。
                # str(output) - 将输出转换为字符串（如果还不是字符串的话）。
                # 相当于 Java 中 System.out.println(output.toString().substring(0, Math.min(200, output.toString().length())));

                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                # 【收集工具结果】构建工具结果对象并添加到结果列表中。
                # type: "tool_result" - 标识这是一个工具结果（Claude API 要求的格式）。
                # tool_use_id: block.id - 关联到对应的工具调用请求。
                # content: str(output) - 工具的实际输出内容。

        messages.append({"role": "user", "content": results})
        # 【发送工具结果】将所有工具结果以 "user" 角色添加到对话历史中。
        # 注意：在 Claude API 中，工具结果是以 "user" 角色发送的（这是 API 的设计约定）。
        # 然后循环会回到 while True 开头，AI 会看到工具结果后决定下一步操作。
        # 这就是 ReAct 模式的核心：思考 -> 行动（调用工具） -> 观察（工具结果） -> 思考 -> ...


# ========== 程序入口 ==========
# 【主程序入口】Python 通过 if __name__ == "__main__": 来判断是否直接运行此脚本。
# __name__ 是 Python 的内置变量：
#   当文件被直接运行时，__name__ 的值为 "__main__"（类似于 Java 的 public static void main）。
#   当文件被其他模块 import 时，__name__ 的值为模块名。
# 这种模式确保了只有直接运行时才执行入口代码，被导入时不会自动执行。
# 相当于 Java 中 main 方法的隔离机制（Java 没有这个问题，因为 Java 必须显式调用 main）。

if __name__ == "__main__":
    # 【程序开始执行】

    history = []
    # 【对话历史】初始化空的对话历史列表。
    # 注意：这里的对话历史只存在于内存中，程序重启后丢失。
    # 但任务数据（通过 TaskManager）是持久化到文件的，所以任务不会丢失。
    # 这正是本系统的核心设计——"对话可能丢失，但任务永远存在"。

    while True:
        # 【交互式循环】持续等待用户输入。

        try:
            query = input("\033[36ms07 >> \033[0m")
            # 【读取用户输入】
            # input(prompt) - 显示提示符并等待用户输入一行文本。
            #   相当于 Java 中的 Scanner.nextLine() 或 System.console().readLine()。
            # "\033[36m" - ANSI 转义码，设置终端颜色为青色（cyan）。
            # "\033[0m" - ANSI 转义码，重置终端颜色为默认。
            #   这些转义码让命令行提示符 "s07 >> " 显示为青色，提升用户体验。

        except (EOFError, KeyboardInterrupt):
            # 【捕获退出信号】
            # EOFError - 用户按 Ctrl+D（Unix）或 Ctrl+Z（Windows）发送文件结束信号。
            # KeyboardInterrupt - 用户按 Ctrl+C 中断程序。
            # (Exception1, Exception2) - Python 支持在一个 except 中捕获多种异常。
            #   相当于 Java 中 catch (EOFError | InterruptedException e)（Java 的多异常捕获，Java 7+）。

            break
            # 【退出循环】break 跳出 while 循环，与 Java 的 break 完全相同。

        if query.strip().lower() in ("q", "exit", ""):
            # 【退出条件检查】
            # .strip() - 去除首尾空白字符，相当于 Java 中 String.trim()。
            # .lower() - 转换为小写，相当于 Java 中 String.toLowerCase()。
            # in ("q", "exit", "") - 检查是否为退出命令。
            #   空字符串 "" 也作为退出条件（直接按回车退出）。

            break
            # 【退出循环】

        history.append({"role": "user", "content": query})
        # 【记录用户消息】将用户输入添加到对话历史中。

        agent_loop(history)
        # 【调用 Agent 循环】将对话历史传入 Agent 主循环。
        # Agent 会调用 Claude API，可能多次调用工具后返回最终响应。
        # history 在 agent_loop 中会被修改（添加 assistant 的响应和工具结果）。

        response_content = history[-1]["content"]
        # 【获取最后一条响应】history[-1] 是 Python 的负索引语法。
        # -1 表示列表的最后一个元素，-2 表示倒数第二个，以此类推。
        # 相当于 Java 中 history.get(history.size() - 1)。
        # ["content"] - 获取响应内容的 content 字段。

        if isinstance(response_content, list):
            # 【类型检查】isinstance() 检查对象是否为指定类型（或其子类）。
            # 相当于 Java 中 responseContent instanceof List。
            # Claude API 的响应 content 可能是字符串（纯文本）或列表（包含多个内容块）。

            for block in response_content:
                # 【遍历内容块】

                if hasattr(block, "text"):
                    # 【属性检查】hasattr() 检查对象是否具有指定属性。
                    # 相当于 Java 中通过反射检查字段是否存在，或检查 instance 的方法。
                    # 这里检查内容块是否有 text 属性（文本块有，工具调用块没有）。

                    print(block.text)
                    # 【打印文本内容】输出 AI 的文本响应。

        print()
        # 【打印空行】在 AI 响应后打印一个空行，提升终端可读性。
