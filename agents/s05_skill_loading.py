#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
# 同前：shebang 行，告诉 Unix/Linux 用 env 在 PATH 中搜索 python3 解释器来执行此脚本

# ─── 模块文档字符串 (docstring) ──────────────────────────────────────────────────
# 三引号字符串 """...""" 是 Python 的多行字符串字面量
# 当它出现在模块/类/函数的第一条语句时，就成为 docstring（文档字符串）
# 对比 Java：
#   Python 的 docstring -> 通过 obj.__doc__ 访问，也可被 pydoc 工具提取生成文档
#   Java 15+ 的文本块 """ ... """ -> 只是多行字符串语法，没有 docstring 语义
#   Java 通常用 Javadoc 注释 /** ... */ 来生成 API 文档
#   Python 的 """ 更简洁，同时充当代码和文档
# 注意：docstring 是代码的一部分（会在运行时存在），而 Javadoc 注释在编译时被丢弃
"""
s05_skill_loading.py - Skills

Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only
    |   - code-review: Review code...      |
    +--------------------------------------+

    When model calls load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

Key insight: "Don't put everything in the system prompt. Load on demand."
"""

# ─── 模块导入 ────────────────────────────────────────────────────────────────────
import os
# 同前：os 模块提供操作系统接口，类似 Java 的 System.getenv() + java.io.File 的集合

import re
# ─── re：正则表达式模块（标准库） ─────────────────────────────────────────────────
# re 是 Python 标准库的正则表达式模块
# 对比 Java：
#   Python: import re; re.match(pattern, string, flags)
#   Java:   import java.util.regex.Pattern/Matcher;
#           Pattern p = Pattern.compile(pattern, flags);
#           Matcher m = p.matcher(string); if (m.matches()) { ... }
# Python 的 re 模块是"一站式"API，match/findall/sub 都在同一个模块中
# Java 的 regex 需要先 compile 得到 Pattern，再 matcher 得到 Matcher，两步操作
# Python 省去了 compile 步骤（内部会缓存编译结果），使用更简洁

import subprocess
# 同前：子进程管理模块，用于执行外部 shell 命令，类似 Java 的 ProcessBuilder

import yaml
# ─── yaml：第三方 YAML 解析库 ────────────────────────────────────────────────────
# yaml 是 PyYAML 库的模块，用于解析 YAML 格式的文本
# 对比 Java：
#   Python: import yaml; data = yaml.safe_load(text)
#   Java:   Maven 依赖 com.fasterxml.jackson.dataformat:jackson-dataformat-yaml
#           ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
#           Map<String, Object> data = mapper.readValue(text, Map.class);
# 重要区别：Python 的 yaml 不是标准库，需要单独安装（pip install pyyaml）
# 而 Java 的正则表达式在 JDK 自带的 java.util.regex 包中，不需要额外依赖
# safe_load() 是安全的解析方法，只解析基本的 YAML 类型，不会执行任意代码
# 类似 Java 中禁用 ObjectMapper 的默认类型自动检测（enableDefaultTyping 的安全配置）

from pathlib import Path
# Path 是 Python 3.4+ 引入的面向对象路径操作类，位于 pathlib 模块中
# 对比 Java：
#   Python: from pathlib import Path; p = Path("/tmp")
#   Java:   java.nio.file.Path p = Path.of("/tmp")
# 两者功能非常相似，Python 的 Path 稍微更 Pythonic（支持 / 运算符拼接路径）

from anthropic import Anthropic
# 同前：Anthropic 官方 SDK 的客户端类，类似 Java 的 OkHttpClient

from dotenv import load_dotenv
# 同前：python-dotenv 库，从 .env 文件加载环境变量，类似 Spring Boot 的 @PropertySource

# ─── 环境变量与配置 ──────────────────────────────────────────────────────────────
load_dotenv(override=True)
# 同前：加载 .env 文件，override=True 表示覆盖已有的系统环境变量

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
# 同前：如果设置了自定义 API 地址，移除可能冲突的认证 token

# ─── 全局常量初始化 ──────────────────────────────────────────────────────────────
WORKDIR = Path.cwd()
# Path.cwd() 返回当前工作目录的 Path 对象，类似 Java 的 Path.of("").toAbsolutePath()
# 对比 s01 中用的 os.getcwd()，Path.cwd() 返回的是 Path 对象（可链式调用），os.getcwd() 返回字符串

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 同前：创建 Anthropic API 客户端

MODEL = os.environ["MODEL_ID"]
# 同前：从环境变量获取模型 ID，缺失时抛出 KeyError（必需配置）

SKILLS_DIR = WORKDIR / "skills"
# ─── Path 的 / 运算符 ───────────────────────────────────────────────────────────
# Python 的 Path 重载了 / 运算符，用于路径拼接
# WORKDIR / "skills" 等价于 WORKDIR.joinpath("skills")
# 对比 Java：
#   Python: WORKDIR / "skills"
#   Java:   WORKDIR.resolve("skills")  或  Path.of(WORKDIR.toString(), "skills")
# Python 的 / 运算符更直观，类似 Unix 路径语法
# 注意：/ 运算符要求左边是 Path 对象，右边是字符串，顺序不能反


# ═══════════════════════════════════════════════════════════════════════════════
# ★ SkillLoader 类 ★
# ─────────────────────────────────────────────────────────────────────────────
# 这是 s05 新增的核心类，实现了"技能加载"机制：
# 1. 扫描 skills/ 目录下的所有 SKILL.md 文件
# 2. 解析每个 SKILL.md 的 YAML frontmatter（元数据）和正文
# 3. 提供两层访问接口：
#    - Layer 1（廉价）：get_descriptions() -> 返回所有技能的简短描述（注入系统提示词）
#    - Layer 2（按需）：get_content(name) -> 返回指定技能的完整正文（注入 tool_result）
#
# 类比 Java 世界的概念：
#   SkillLoader 类似 Spring 中的 ResourceLoader + ApplicationContext 的组合：
#   - 扫描类路径下的资源文件 -> 类似 Spring 扫描 @Component 注解
#   - 解析 YAML 元数据 -> 类似 @ConfigurationProperties
#   - 按名称获取内容 -> 类似 ApplicationContext.getBean(name)
#
# 设计模式：
#   - 前端控制器模式 (Front Controller)：所有技能通过统一的 load_skill 工具入口访问
#   - 延迟加载 (Lazy Loading)：技能正文不预加载到系统提示词，而是在模型需要时才加载
#   - 策略模式的简化版：每个 SKILL.md 就是一个"策略"，描述特定领域的知识
# ═══════════════════════════════════════════════════════════════════════════════

class SkillLoader:
    # ─── Python 类定义 ──────────────────────────────────────────────────────────
    # Python 用 class 关键字定义类，对比 Java：
    #   Python: class SkillLoader:
    #   Java:   public class SkillLoader {
    # 区别：
    # 1. Python 不需要 public 修饰符（所有类默认都是 public 的）
    # 2. Python 类名遵循 PascalCase（大驼峰），与 Java 一致
    # 3. Python 不需要分号，用冒号和缩进表示类体范围
    # 4. Python 支持 class SkillLoader: 语法（没有显式继承 Object，但隐式继承）
    #    Java 需要 public class SkillLoader（或 public class SkillLoader extends Object）
    # 5. Python 的类定义本质上是"执行"一个代码块，这叫做"类体是执行作用域"
    #    这意味着类体中可以放任意的 Python 语句，不仅限于字段和方法声明

    def __init__(self, skills_dir: Path):
        # ─── 构造函数 __init__ ─────────────────────────────────────────────────
        # __init__ 是 Python 的实例构造函数（初始化方法），对比 Java：
        #   Python: def __init__(self, skills_dir: Path):
        #   Java:   public SkillLoader(Path skillsDir) {
        # 区别：
        # 1. __init__ 名称是固定的（双下划线开头和结尾，称为"dunder"方法）
        #    Java 的构造函数与类同名
        # 2. self 参数是显式的，代表当前实例自身
        #    Java 中用隐式的 this 引用当前实例
        #    self.skills_dir = skills_dir 等价于 Java 的 this.skillsDir = skillsDir;
        # 3. skills_dir: Path 是类型注解，仅作提示，运行时不强制检查
        #    Java 的类型声明是强制的，编译器会检查

        # self 是 Python 中实例方法的第一个参数，代表当前对象实例
        # 类似 Java 中的 this，但 Python 要求显式声明（不声明就不存在）
        # self.skills_dir = ... 是实例属性赋值，相当于 Java 中在构造函数里 this.skillsDir = ...
        # Python 没有字段声明语法，属性在首次赋值时自动创建（动态语言的特性）
        # Java 需要在类体中先声明字段：private Path skillsDir;
        self.skills_dir = skills_dir

        # self.skills 是一个 dict（字典），用于存储所有加载的技能
        # 初始化为空字典 {}，类似 Java 的 Map<String, Map<String, String>> skills = new HashMap<>();
        # 键是技能名称（如 "pdf"），值是包含 meta/body/path 的字典
        # Python 的 dict 类似 Java 的 HashMap<String, Object>，但更灵活：
        #   - 可以存储任意类型的值
        #   - 不需要泛型声明
        #   - 支持 {} 字面量创建
        self.skills = {}

        # 调用 _load_all() 方法，在构造时自动扫描并加载所有技能
        # self._load_all() 等价于 Java 中的 this.loadAll()
        # _ 前缀是 Python 的命名约定，表示"内部方法"（类似 Java 的 private，但不是强制的）
        # Python 没有 private 关键字，_ 前缀只是告诉其他开发者"这是内部实现，不要直接调用"
        self._load_all()

    def _load_all(self):
        # ─── 扫描并加载所有技能 ─────────────────────────────────────────────────
        # _load_all 方法在构造时被调用，扫描 skills_dir 下所有的 SKILL.md 文件
        # 类比 Java：类似 Spring 的 ClassPathScanningCandidateComponentProvider 扫描组件
        # 但这里是扫描文件系统目录，不是 classpath

        # self.skills_dir.exists() 检查目录是否存在
        # Path.exists() 返回 bool，类似 Java 的 Files.exists(path)
        # 如果 skills 目录不存在（用户还没创建），直接返回，不报错
        if not self.skills_dir.exists():
            return

        # ─── Path.rglob() 递归文件搜索 ─────────────────────────────────────────
        # self.skills_dir.rglob("SKILL.md") 递归搜索目录下所有名为 SKILL.md 的文件
        # 返回一个生成器（Generator），产生所有匹配的 Path 对象
        # 对比 Java：
        #   Python: skills_dir.rglob("SKILL.md")
        #   Java:   Files.walk(skillsDir)
        #               .filter(p -> p.getFileName().toString().equals("SKILL.md"))
        #               .collect(Collectors.toList())
        # rglob("*.md") 会匹配 skills/pdf/SKILL.md, skills/code-review/SKILL.md 等
        # rglob 是 "recursive glob" 的缩写
        # 如果用 glob("SKILL.md") 则只搜索当前目录（不递归子目录）

        # ─── sorted() 内置排序 ─────────────────────────────────────────────────
        # sorted() 是 Python 内置函数，对可迭代对象排序，返回新列表
        # 对比 Java：
        #   Python: sorted(iterable)
        #   Java:   list.stream().sorted().collect(Collectors.toList())
        #        或 Collections.sort(list)
        # 区别：sorted() 返回新列表，不修改原数据；Java 的 Collections.sort() 原地排序
        # 为什么排序？为了让技能描述的顺序稳定且可预测（字典在 Python 3.7+ 是有序的）
        # 如果不排序，不同操作系统/文件系统返回的文件顺序可能不同
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            # f.read_text() 读取文件全部内容为字符串
            # 类似 Java 的 Files.readString(path, StandardCharsets.UTF_8)
            # 如果文件编码不是 UTF-8，可以传参数：f.read_text(encoding="gbk")
            text = f.read_text()

            # _parse_frontmatter 解析 SKILL.md 的 YAML 前置元数据和正文
            # 返回 (meta, body) 元组，类似 Java 返回一个 record 或 Pair
            meta, body = self._parse_frontmatter(text)

            # ─── 元组解包赋值 ──────────────────────────────────────────────────
            # meta, body = ... 是 Python 的元组解包（tuple unpacking）
            # _parse_frontmatter 返回一个 tuple (meta, body)，这里拆成两个变量
            # 对比 Java：
            #   Python: meta, body = func()        # 自动解包
            #   Java:   var result = func();       // 需要 Map.Entry 或 record
            #           var meta = result.getKey();
            #           var body = result.getValue();
            # Java 没有 Python 那样简洁的多返回值解包语法（Java 可以用 record 来近似）

            # ─── f.parent.name 获取父目录名 ────────────────────────────────────
            # f 是 Path 对象，例如 skills/pdf/SKILL.md
            # f.parent 返回父目录 Path，即 skills/pdf/
            # f.parent.name 返回父目录的名称部分，即 "pdf"
            # 对比 Java：
            #   Python: f.parent.name
            #   Java:   f.getParent().getFileName().toString()
            # Python 的链式调用更简洁，Path 对象的方法直接返回 Path
            # Java 需要 getFileName() 再 toString() 才能得到纯名称字符串
            name = meta.get("name", f.parent.name)
            # meta.get("name", default) 从字典中取值，key 不存在时返回默认值
            # 类似 Java 的 map.getOrDefault("name", default)
            # 如果 YAML frontmatter 中定义了 name 字段就用它，否则用父目录名作为技能名

            # 将技能信息存入 self.skills 字典
            # 每个技能的值是一个嵌套字典，包含三个键：
            #   "meta" -> YAML frontmatter 解析出的元数据字典（name, description, tags 等）
            #   "body" -> SKILL.md 的正文内容（技能的具体指令）
            #   "path" -> 文件的字符串路径，用于调试和错误提示
            # 类似 Java 中：skills.put(name, Map.of("meta", meta, "body", body, "path", pathStr))
            # 但 Python 的字典字面量 {} 比 Java 的 Map.of() 更简洁
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}
            # str(f) 将 Path 对象转为字符串路径，类似 Java 的 path.toString()

    def _parse_frontmatter(self, text: str) -> tuple:
        # ─── 解析 YAML frontmatter ─────────────────────────────────────────────
        # frontmatter 是一种常见的元数据格式，用在 Markdown 文件开头
        # 格式为：--- 开头和结尾包裹 YAML 内容，后面是正文
        #
        # 示例 SKILL.md：
        #   ---
        #   name: pdf
        #   description: Process PDF files
        #   tags: document
        #   ---
        #   # PDF Processing
        #   Step 1: Extract text from PDF...
        #   Step 2: Analyze the content...
        #
        # 返回值 tuple 是 Python 的元组类型注解，表示返回一个 (dict, str) 元组
        # 对比 Java：需要定义一个 record MetaBody(Map<String, Object> meta, String body)
        # 或者用 Map.Entry<Map<String, Object>, String>
        """Parse YAML frontmatter between --- delimiters."""
        # 同前：docstring 保留原文，这是方法的文档说明

        # ─── re.match() 正则匹配 ──────────────────────────────────────────────
        # re.match(pattern, string, flags) 从字符串开头匹配正则表达式
        # 对比 Java：
        #   Python: re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        #   Java:   Pattern p = Pattern.compile("^---\\n(.*?)\\n---\\n(.*)", Pattern.DOTALL);
        #           Matcher m = p.matcher(text);
        #           if (m.matches()) { ... }
        # 区别：
        # 1. Python 的 re.match() 自动从字符串开头匹配（相当于 Java 的 m.matches()）
        #    如果要从任意位置匹配，用 re.search()（相当于 Java 的 m.find()）
        # 2. Python 用 r"..." 原始字符串表示正则表达式
        #    Java 用普通字符串，反斜杠需要双重转义：\\n 代替 \n

        # ─── re.DOTALL 标志 ────────────────────────────────────────────────────
        # re.DOTALL 使正则中的 . 匹配包括换行符在内的所有字符
        # 默认情况下，. 不匹配 \n（换行符）
        # 对比 Java：
        #   Python: re.DOTALL（或 re.S 简写）
        #   Java:   Pattern.DOTALL
        # 两者功能完全相同
        # 这里必须用 DOTALL，因为 (.*?) 需要跨行匹配 YAML frontmatter 的内容
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        # 如果没有匹配到 frontmatter 格式（文件可能不以 --- 开头）
        # match 为 None，类似 Java 中 matcher.matches() 返回 false
        if not match:
            # 返回空字典和原始文本，表示没有元数据
            # Python 可以直接返回 (dict, str) 形式的元组，不需要包装
            return {}, text

        try:
            # ─── yaml.safe_load() 解析 YAML ───────────────────────────────────
            # yaml.safe_load(string) 将 YAML 格式的字符串解析为 Python 对象（dict/list/str 等）
            # 对比 Java：
            #   Python: yaml.safe_load(yaml_string)
            #   Java:   new Yaml().load(yaml_string)  // SnakeYAML 库
            #       或 ObjectMapper(new YAMLFactory()).readValue(yaml_string, Map.class)
            # safe_load 是"安全版本"，只解析基本 YAML 类型，不会反序列化 Python 对象
            # 这避免了 YAML 反序列化漏洞（类似 Java 中的 Jackson enableDefaultTyping 安全问题）

            # ─── match.group(1) 获取捕获组 ────────────────────────────────────
            # match.group(1) 返回正则表达式中第一个 () 捕获组匹配的内容
            # match.group(2) 返回第二个捕获组
            # 对比 Java：
            #   Python: match.group(1)
            #   Java:   matcher.group(1)
            # 功能完全相同：group(0) 是整个匹配，group(1) 是第一个括号组
            # 这里：
            #   group(1) = YAML frontmatter 内容（--- 之间的部分）
            #   group(2) = 正文内容（--- 之后的 Markdown 内容）

            # or {} 是 Python 的短路或运算，处理 yaml.safe_load() 返回 None 的情况
            # 如果 YAML 内容为空（如 "---\n---\n"），safe_load 返回 None
            # None or {} -> 返回 {}（因为 None 是 falsy 值）
            # 对比 Java：需要显式判空：meta = result != null ? result : new HashMap<>();
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            # 捕获 YAML 解析错误，类似 Java 的 catch (YAMLException e)
            # 如果 YAML 格式有误，不崩溃，而是返回空元数据
            meta = {}

        # match.group(2) 是 frontmatter 之后的所有正文内容
        # .strip() 去除首尾空白，避免多余的空行
        # 返回 (meta, body) 元组——Python 允许函数返回多个值（本质是返回一个 tuple）
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        # ─── Layer 1：返回所有技能的简短描述 ───────────────────────────────────
        # 这些描述会注入到系统提示词中，让模型知道有哪些技能可用
        # 每个 skill 只消耗约 100 tokens，远比注入完整正文（可能数千 tokens）便宜
        # 类比 Java：类似 Spring Actuator 的 /info 端点，只返回摘要信息
        """Layer 1: short descriptions for the system prompt."""
        # 同前：docstring 保留原文

        # if not self.skills: 检查字典是否为空
        # 空字典 {} 是 falsy 值，not {} 为 True
        # 类似 Java 的 skills.isEmpty()
        if not self.skills:
            return "(no skills available)"

        # 初始化空列表，用于逐行构建技能描述
        lines = []

        # self.skills.items() 返回 (key, value) 对的迭代器
        # 类似 Java 的 skills.entrySet()（返回 Set<Map.Entry<K,V>>）
        # for name, skill in ... 是元组解包迭代
        # 类似 Java 的 for (var entry : skills.entrySet()) { String name = entry.getKey(); ... }
        for name, skill in self.skills.items():
            # skill["meta"].get("description", "No description") 从嵌套字典中取值
            # 带默认值的 get，key 不存在时返回默认值
            # 类似 Java 的 ((Map<String,String>) skill.get("meta")).getOrDefault("description", "No description")
            # Python 的字典链式访问比 Java 简洁很多（不需要类型转换）
            desc = skill["meta"].get("description", "No description")

            # 同理获取 tags（标签），默认为空字符串
            tags = skill["meta"].get("tags", "")

            # ─── f-string 格式化字符串 ────────────────────────────────────────
            # f"  - {name}: {desc}" 是 Python 的格式化字符串字面量
            # 花括号内的变量会被求值并嵌入字符串
            # 对比 Java：
            #   Python: f"  - {name}: {desc}"
            #   Java:   String.format("  - %s: %s", name, desc)
            #   Java 15+: "  - %s: %s".formatted(name, desc)
            line = f"  - {name}: {desc}"

            # if tags: 如果 tags 非空（非空字符串是 truthy 值）
            # 类似 Java 的 if (tags != null && !tags.isEmpty())
            if tags:
                # += 字符串拼接，将标签附加到行尾
                # 类似 Java 的 line += " [" + tags + "]"
                line += f" [{tags}]"

            # lines.append(line) 在列表末尾添加元素，类似 Java 的 lines.add(line)
            lines.append(line)

        # "\n".join(lines) 用换行符连接列表中的所有字符串
        # 对比 Java：
        #   Python: "\n".join(list_of_strings)
        #   Java:   String.join("\n", list_of_strings)
        # 注意：Python 的 join 是字符串方法，Java 的 String.join 是静态方法
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        # ─── Layer 2：返回指定技能的完整正文 ───────────────────────────────────
        # 当模型调用 load_skill("pdf") 工具时，这个方法返回完整的技能内容
        # 完整内容作为 tool_result 注入模型的上下文
        # 这是"延迟加载"策略的核心：不在系统提示词中塞入所有技能，而是按需加载
        # 类比 Java：
        #   get_descriptions() -> 类似 JMX 的 MBeanInfo（元数据摘要）
        #   get_content(name) -> 类似实际调用 MBean 方法（完整操作）
        """Layer 2: full skill body returned in tool_result."""
        # 同前：docstring 保留原文

        # self.skills.get(name) 从字典中获取指定名称的技能
        # 如果 name 不存在，返回 None（不是抛异常）
        # 类似 Java 的 map.get(name)，而 map.get() 在 Java 中返回 null
        skill = self.skills.get(name)

        # if not skill: 检查是否为 None（技能不存在）
        # 类似 Java 的 if (skill == null)
        if not skill:
            # 返回错误信息，列出所有可用的技能名称
            # ", ".join(self.skills.keys()) 用逗号连接所有技能名
            # self.skills.keys() 返回字典所有键的视图
            # 类似 Java 的 skills.keySet().stream().collect(Collectors.joining(", "))
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"

        # 返回包裹在 <skill> 标签中的完整技能正文
        # XML 标签帮助模型区分普通文本和技能指令
        # 类似 Java 中返回 "<skill name=\"" + name + "\">\n" + body + "\n</skill>"
        # Python 的 f-string 在多行拼接时更简洁
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
        # skill['body'] 用下标访问字典值
        # 注意：skill["body"] 和 skill['body'] 在 Python 中完全等价
        # 只是在字符串内部需要交替使用单引号和双引号以避免转义


# ─── 全局实例化 ──────────────────────────────────────────────────────────────────
SKILL_LOADER = SkillLoader(SKILLS_DIR)
# 创建 SkillLoader 的全局单例实例
# 构造时会自动扫描 skills/ 目录并加载所有 SKILL.md
# 对比 Java：
#   Python: SKILL_LOADER = SkillLoader(SKILLS_DIR)  # 模块级变量就是"单例"
#   Java:   需要显式写单例模式（或用 Spring 的 @Component + @Autowired）
# Python 没有静态类初始化块，模块级代码在首次 import 时执行一次
# 这相当于 Java 的 static { ... } 初始化块 + 单例模式的结合

# ─── Layer 1：系统提示词（包含技能元数据摘要） ────────────────────────────────────
# f"""...""" 是三引号 f-string，支持多行字符串 + 变量插值
# 对比 Java：
#   Python: f"""...{variable}..."""
#   Java 15+: """...""".formatted(variable)  或  """
#             ...""" + variable + """..."""
# 三引号字符串在 Python 中非常常用，特别适合：
# 1. 多行文本（如 SQL、HTML、系统提示词）
# 2. docstring（文档字符串）
# 3. 需要嵌入引号的字符串（不需要转义）
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""
# SKILL_LOADER.get_descriptions() 在系统提示词构建时调用一次
# 它会生成类似这样的文本：
#   Skills available:
#     - pdf: Process PDF files [document]
#     - code-review: Review code quality [coding]
# 这些是 Layer 1 的廉价元数据（约 100 tokens/skill）
# 完整的技能正文不会出现在系统提示词中


# ─── Tool implementations（工具实现） ────────────────────────────────────────────
# 以下工具函数与 s02/s03/s04 中的实现基本相同
# s05 的新增内容是 load_skill 工具（在 TOOL_HANDLERS 中注册）

def safe_path(p: str) -> Path:
    # 同前：路径安全检查，防止路径遍历攻击
    # 将相对路径解析为绝对路径，并确保不逃逸出工作目录
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    # 同前：执行 shell 命令，带危险命令检查和超时控制
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

def run_read(path: str, limit: int = None) -> str:
    # 同前：读取文件内容，支持行数限制
    # limit: int = None 是带默认值的参数，Python 独有的语法
    # 类似 Java 的方法重载：runRead(path) 和 runRead(path, limit)
    # 但 Python 用一个函数 + 默认值就实现了，Java 需要两个方法
    try:
        lines = safe_path(path).read_text().splitlines()
        # .splitlines() 将字符串按行分割为列表，类似 Java 的 String.split("\\n")
        if limit and limit < len(lines):
            # 如果指定了行数限制且文件超出限制，截断并添加提示
            # lines[:limit] 是列表切片，取前 limit 个元素
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    # 同前：写入文件，自动创建父目录
    try:
        fp = safe_path(path)
        # mkdir(parents=True, exist_ok=True) 递归创建目录
        # parents=True 类似 Java 的 Files.createDirectories()
        # exist_ok=True 表示目录已存在时不报错
        # Java 的 Files.createDirectories() 默认就是幂等的（已存在则返回）
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    # 同前：编辑文件，精确替换文本（只替换第一个匹配）
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # str.replace(old, new, count) 第三个参数限制替换次数
        # 类似 Java 中手动实现"只替换第一次出现"的逻辑
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ─── 工具注册表 ──────────────────────────────────────────────────────────────────
# TOOL_HANDLERS 是一个 dict，将工具名称映射到对应的处理函数
# 同前：字典的键是字符串（工具名），值是 lambda 函数

# ★ 新增：load_skill 工具的处理器 ★
# lambda **kw: SKILL_LOADER.get_content(kw["name"])
# 这是一个 lambda 表达式（匿名函数），对比 Java：
#   Python: lambda **kw: SKILL_LOADER.get_content(kw["name"])
#   Java:   (Map<String, Object> kw) -> SKILL_LOADER.getContent((String) kw.get("name"))
# **kw 是 Python 的关键字参数打包语法（kwargs = keyword arguments）
# 它将传入的关键字参数收集为一个字典 kw
# 例如调用 handler(name="pdf") 时，kw = {"name": "pdf"}
# 这与 Java 中接收 Map<String, Object> 参数的效果类似，但 Python 的语法更简洁
# 其他 handler 同前，使用相同模式：
#   lambda **kw: run_bash(kw["command"])
#   lambda **kw: run_read(kw["path"], kw.get("limit"))
# 等，都是将 kw 字典中的参数解包后传给对应的函数
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),  # ★ s05 新增
}

# ─── 工具定义列表 ────────────────────────────────────────────────────────────────
# TOOLS 是 Anthropic API 要求的工具定义格式（JSON Schema）
# 同前：每个工具定义包含 name、description 和 input_schema
# ★ 新增：load_skill 工具定义 ★
# description 告诉模型"加载专业知识"，让模型在遇到不熟悉的领域时主动调用
# input_schema 定义工具接受的参数：name（技能名称）
# 当模型在对话中遇到它不熟悉的任务（如处理 PDF），
# 它会先调用 load_skill("pdf") 获取完整的 PDF 处理指令
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # ★ s05 新增：load_skill 工具定义 ★
    # 这是两层技能注入机制的关键：模型通过调用这个工具来按需加载技能
    # input_schema 定义了 name 参数（字符串类型），是技能的名称
    # required=["name"] 表示 name 是必填参数
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


# ─── Agent Loop（Agent 核心循环） ────────────────────────────────────────────────
def agent_loop(messages: list):
    # 同前：Agent 核心循环，调用模型 -> 执行工具 -> 回传结果 -> 循环直到模型停止
    # s05 与 s01-s04 的 agent_loop 逻辑完全相同，区别在于多了 load_skill 工具
    # 模型在循环中可以选择调用 load_skill 来获取技能知识，然后继续处理任务
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 从 TOOL_HANDLERS 字典中获取工具对应的处理函数
                # .get(block.name) 如果工具名不存在则返回 None
                # 同前：字典查找 + lambda 调用模式
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # handler(**block.input) 用 ** 解包字典为关键字参数
                    # 例如 block.input = {"name": "pdf"} 时
                    # 等价于 handler(name="pdf")
                    # 对比 Java：需要手动从 Map 中取值再传参
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                # 打印工具调用信息到终端
                print(f"> {block.name}:")
                # str(output)[:200] 截断输出到前 200 字符，避免终端刷屏
                print(str(output)[:200])
                # 同前：将工具结果添加到 results 列表
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})


# ─── 程序入口 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 同前：Python 的惯用入口模式
    # __name__ == "__main__" 确保代码只在直接运行时执行
    history = []
    while True:
        try:
            # input() 读取用户输入，\033[36m 是 ANSI 青色，\033[0m 重置样式
            # 提示符显示 "s05 >> " 表示当前是第 5 个示例
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # 同前：捕获 Ctrl+D/Ctrl+C 退出
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
