import gradio as gr
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI
import asyncio
import os
from enum import Enum
import traceback

# 系统配置类
class Config:
    """全局配置管理"""
    VERSION = "1.0.0"
    DEFAULT_API_BASE = "https://open.api.gu28.top/v1"
    GITHUB_REPO = "https://github.com/bone-de/agests"
    LOG_FILE = "chat_assistant.log"
    CONFIG_FILE = "config.yaml"
    WORKFLOW_FILE = "workflows.json"
    CONVERSATION_DIR = "conversations"
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.CONVERSATION_DIR, exist_ok=True)
    
    @classmethod
    def load_config(cls) -> Dict:
        """加载配置文件"""
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
        return {}

    @classmethod
    def save_config(cls, config: Dict) -> bool:
        """保存配置到文件"""
        try:
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
            return True
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
            return False

# 系统状态枚举
class SystemState(Enum):
    UNINITIALIZED = "未初始化"
    INITIALIZED = "已初始化"
    ERROR = "错误状态"

# 数据模型
@dataclass
class Assistant:
    """助手数据模型"""
    id: str
    name: str
    role: str
    prompt: str
    created_at: str = datetime.now().isoformat()
    modified_at: str = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class WorkflowStep:
    """工作流步骤数据模型"""
    assistants: List[str]
    is_parallel: bool
    step_id: str = ""
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

@dataclass
class Workflow:
    """工作流数据模型"""
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    created_at: str = datetime.now().isoformat()
    modified_at: str = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "steps": [asdict(step) for step in self.steps],
            "description": self.description,
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }

@dataclass
class Message:
    """消息数据模型"""
    role: str
    content: str
    timestamp: str = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

# 自定义异常类
class AssistantError(Exception):
    """助手相关错误"""
    pass

class WorkflowError(Exception):
    """工作流相关错误"""
    pass

class APIError(Exception):
    """API调用相关错误"""
    pass

# 默认助手配置
DEFAULT_ASSISTANTS = {
    "code_expert": Assistant(
        id="code_expert",
        name="代码专家",
        role="Python开发者",
        prompt="""你是一个专业的Python开发者，精通软件开发最佳实践。
        职责：
        1. 编写高质量、可维护的代码
        2. 提供详细的代码说明和文档
        3. 优化代码性能和结构
        4. 解决技术难题和debug
        
        回复要求：
        1. 给出完整、可运行的代码
        2. 包含必要的注释和说明
        3. 说明代码的优缺点
        4. 提供可能的优化建议"""
    ),
    "reviewer": Assistant(
        id="reviewer",
        name="代码审查",
        role="审查专家",
        prompt="""你是一个经验丰富的代码审查专家，专注于代码质量和最佳实践。
        职责：
        1. 审查代码质量和风格
        2. 检查潜在的bug和安全问题
        3. 提供改进建议
        4. 确保代码符合最佳实践
        
        审查重点：
        1. 代码可读性和维护性
        2. 性能优化空间
        3. 安全漏洞
        4. 架构设计合理性"""
    ),
    "architect": Assistant(
        id="architect",
        name="系统架构师",
        role="架构专家",
        prompt="""你是一个资深的系统架构师，擅长设计可扩展的系统架构。
        职责：
        1. 系统架构设计
        2. 技术选型建议
        3. 性能和扩展性规划
        4. 解决方案评估
        
        输出要求：
        1. 清晰的架构图和说明
        2. 详细的技术选型理由
        3. 可能的风险和解决方案
        4. 扩展性和维护性考虑"""
    )
}

# 自定义CSS样式
CUSTOM_CSS = """
.container { max-width: 1200px; margin: 0 auto; }
.header { 
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.assistant-box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background: white;
    transition: all 0.3s ease;
}
.assistant-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.workflow-container {
    min-height: 100px;
    border: 2px dashed #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    background: #f8f9fa;
}
.workflow-step {
    border: 1px solid #ddd;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
    background: white;
}
.parallel-step {
    background: #e3f2fd;
    border-color: #90caf9;
}
.sequential-step {
    background: #f1f8e9;
    border-color: #a5d6a7;
}
.status-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px;
    background: rgba(255,255,255,0.9);
    border-top: 1px solid #ddd;
    z-index: 1000;
}
.chat-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.chat-history {
    flex-grow: 1;
    overflow-y: auto;
    padding: 10px;
}
.chat-input {
    padding: 10px;
    border-top: 1px solid #ddd;
}
.github-link {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    align-items: center;
    text-decoration: none;
    color: #333;
    padding: 5px 10px;
    border-radius: 5px;
    transition: all 0.3s ease;
}
.github-link:hover {
    background: #f0f0f0;
}
.error-message {
    color: #d32f2f;
    background: #ffebee;
    padding: 10px;
    border-radius: 4px;
    margin: 5px 0;
}
.success-message {
    color: #388e3c;
    background: #f1f8e9;
    padding: 10px;
    border-radius: 4px;
    margin: 5px 0;
}
.conversation-log {
    font-family: monospace;
    white-space: pre-wrap;
    background: #f5f5f5;
    padding: 10px;
    border-radius: 4px;
    height: 100%;
    overflow-y: auto;
}
"""

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 确保必要的目录存在
Config.ensure_directories()
class AssistantManager:
    """助手管理类"""
    def __init__(self):
        self.assistants: Dict[str, Assistant] = {}
        self.load_default_assistants()
        self.load_saved_assistants()

    def load_default_assistants(self) -> None:
        """加载默认助手配置"""
        try:
            for assistant in DEFAULT_ASSISTANTS.values():
                self.assistants[assistant.id] = assistant
            logger.info("默认助手配置加载完成")
        except Exception as e:
            logger.error(f"加载默认助手配置失败: {e}")
            raise AssistantError("加载默认助手配置失败")

    def load_saved_assistants(self) -> None:
        """从文件加载保存的助手配置"""
        try:
            config = Config.load_config()
            if 'assistants' in config:
                for assistant_data in config['assistants']:
                    assistant = Assistant(**assistant_data)
                    self.assistants[assistant.id] = assistant
                logger.info("已保存的助手配置加载完成")
        except Exception as e:
            logger.error(f"加载保存的助手配置失败: {e}")

    def save_assistants(self) -> bool:
        """保存助手配置到文件"""
        try:
            config = Config.load_config()
            config['assistants'] = [asdict(assistant) for assistant in self.assistants.values()]
            Config.save_config(config)
            logger.info("助手配置保存成功")
            return True
        except Exception as e:
            logger.error(f"保存助手配置失败: {e}")
            return False

    def add_assistant(self, name: str, role: str, prompt: str) -> bool:
        """添加新助手"""
        try:
            assistant_id = name.lower().replace(" ", "_")
            if assistant_id in self.assistants:
                raise AssistantError("助手ID已存在")
            
            assistant = Assistant(
                id=assistant_id,
                name=name,
                role=role,
                prompt=prompt
            )
            self.assistants[assistant_id] = assistant
            self.save_assistants()
            logger.info(f"添加新助手成功: {name}")
            return True
        except Exception as e:
            logger.error(f"添加助手失败: {e}")
            return False

    def update_assistant(self, assistant_id: str, data: Dict) -> bool:
        """更新助手配置"""
        try:
            if assistant_id not in self.assistants:
                raise AssistantError("助手不存在")
            
            assistant = self.assistants[assistant_id]
            for key, value in data.items():
                if hasattr(assistant, key):
                    setattr(assistant, key, value)
            
            assistant.modified_at = datetime.now().isoformat()
            self.save_assistants()
            logger.info(f"更新助手成功: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"更新助手失败: {e}")
            return False

    def get_assistant(self, assistant_id: str) -> Optional[Assistant]:
        """获取指定助手"""
        return self.assistants.get(assistant_id)

    def get_all_assistants(self) -> List[Assistant]:
        """获取所有助手"""
        return list(self.assistants.values())

    def delete_assistant(self, assistant_id: str) -> bool:
        """删除助手"""
        try:
            if assistant_id not in self.assistants:
                return False
            del self.assistants[assistant_id]
            self.save_assistants()
            logger.info(f"删除助手成功: {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"删除助手失败: {e}")
            return False

class WorkflowManager:
    """工作流管理类"""
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.current_workflow: List[WorkflowStep] = []
        self.load_workflows()

    def load_workflows(self) -> None:
        """加载保存的工作流"""
        try:
            if os.path.exists(Config.WORKFLOW_FILE):
                with open(Config.WORKFLOW_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for workflow_data in data.values():
                        workflow = Workflow(
                            name=workflow_data['name'],
                            steps=[WorkflowStep(**step) for step in workflow_data['steps']],
                            description=workflow_data.get('description', ''),
                            created_at=workflow_data.get('created_at', ''),
                            modified_at=workflow_data.get('modified_at', '')
                        )
                        self.workflows[workflow.name] = workflow
                logger.info("工作流配置加载完成")
        except Exception as e:
            logger.error(f"加载工作流配置失败: {e}")

    def save_workflows(self) -> bool:
        """保存工作流配置"""
        try:
            data = {name: workflow.to_dict() for name, workflow in self.workflows.items()}
            with open(Config.WORKFLOW_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("工作流配置保存成功")
            return True
        except Exception as e:
            logger.error(f"保存工作流配置失败: {e}")
            return False

    def add_step(self, assistants: List[str], is_parallel: bool = False) -> bool:
        """添加工作流步骤"""
        try:
            if not assistants:
                raise WorkflowError("未选择助手")
            
            # 确保所有助手ID都是字符串
            assistants = [str(a) for a in assistants if a]
            if not assistants:
                raise WorkflowError("无效的助手选择")
            
            step = WorkflowStep(assistants=assistants, is_parallel=is_parallel)
            self.current_workflow.append(step)
            logger.info(f"添加工作流步骤成功: {'并行' if is_parallel else '顺序'}")
            return True
        except Exception as e:
            logger.error(f"添加工作流步骤失败: {e}")
            return False

    def save_current_workflow(self, name: str, description: str = "") -> bool:
        """保存当前工作流"""
        try:
            if not name or not self.current_workflow:
                raise WorkflowError("工作流名称为空或工作流为空")
            
            workflow = Workflow(
                name=name,
                steps=self.current_workflow.copy(),
                description=description
            )
            self.workflows[name] = workflow
            self.save_workflows()
            logger.info(f"保存工作流成功: {name}")
            return True
        except Exception as e:
            logger.error(f"保存工作流失败: {e}")
            return False

    def load_workflow(self, name: str) -> bool:
        """加载工作流"""
        try:
            if name not in self.workflows:
                raise WorkflowError("工作流不存在")
            
            self.current_workflow = self.workflows[name].steps.copy()
            logger.info(f"加载工作流成功: {name}")
            return True
        except Exception as e:
            logger.error(f"加载工作流失败: {e}")
            return False

    def clear_current_workflow(self) -> None:
        """清空当前工作流"""
        self.current_workflow = []
        logger.info("清空当前工作流")

    def get_workflow_display(self) -> str:
        """获取当前工作流的HTML显示"""
        try:
            if not self.current_workflow:
                return '<div class="workflow-container"><div class="workflow-step">工作流为空</div></div>'
            
            html = '<div class="workflow-container">'
            for i, step in enumerate(self.current_workflow, 1):
                step_class = "parallel-step" if step.is_parallel else "sequential-step"
                step_type = "并行" if step.is_parallel else "顺序"
                html += f'''
                    <div class="workflow-step {step_class}">
                        <div class="step-header">步骤 {i} ({step_type})</div>
                        <div class="step-content">{" + ".join(step.assistants)}</div>
                    </div>
                '''
            html += '</div>'
            return html
        except Exception as e:
            logger.error(f"生成工作流显示失败: {e}")
            return '<div class="workflow-container"><div class="workflow-step error-message">显示错误</div></div>'

class ChatManager:
    """对话管理类"""
    def __init__(self, api_key: str = "", api_base: str = Config.DEFAULT_API_BASE):
        self.api_key = api_key
        self.api_base = api_base
        self.client: Optional[OpenAI] = None
        self.conversation_history: List[Message] = []
        self.system_state = SystemState.UNINITIALIZED

    def initialize(self, api_key: str, api_base: str) -> bool:
        """初始化OpenAI客户端"""
        try:
            self.api_key = api_key
            self.api_base = api_base
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.system_state = SystemState.INITIALIZED
            logger.info("OpenAI客户端初始化成功")
            return True
        except Exception as e:
            self.system_state = SystemState.ERROR
            logger.error(f"OpenAI客户端初始化失败: {e}")
            return False

    async def process_message(self, message: str, workflow: List[WorkflowStep], model: str) -> str:
        """处理消息"""
        if self.system_state != SystemState.INITIALIZED:
            raise APIError("系统未初始化")

        try:
            current_message = message
            results = []

            # 记录用户消息
            self.conversation_history.append(Message(role="user", content=message))

            for step in workflow:
                if step.is_parallel:
                    # 并行处理
                    tasks = [
                        self.get_assistant_response(current_message, assistant_id, model)
                        for assistant_id in step.assistants
                    ]
                    step_results = await asyncio.gather(*tasks)
                    results.extend(step_results)
                    current_message = "\n".join(step_results)
                else:
                    # 顺序处理
                    for assistant_id in step.assistants:
                        response = await self.get_assistant_response(current_message, assistant_id, model)
                        results.append(response)
                        current_message = response

            # 记录助手响应
            final_response = "\n".join(results)
            self.conversation_history.append(Message(role="assistant", content=final_response))
            
            return final_response

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            raise APIError(f"处理消息失败: {str(e)}")

    async def get_assistant_response(self, message: str, assistant_id: str, model: str) -> str:
        """获取助手响应"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": f"You are {assistant_id}"},
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"获取助手响应失败: {e}")
            raise APIError(f"获取助手响应失败: {str(e)}")

    def save_conversation(self, filename: str = None) -> str:
        """保存对话历史"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(Config.CONVERSATION_DIR, f"conversation_{timestamp}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                for msg in self.conversation_history:
                    f.write(f"[{msg.timestamp}] {msg.role}: {msg.content}\n\n")
            
            logger.info(f"对话历史保存成功: {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存对话历史失败: {e}")
            raise Exception("保存对话历史失败")

    def get_formatted_history(self) -> str:
        """获取格式化的对话历史"""
        formatted = []
        for msg in self.conversation_history:
            formatted.append(f"[{msg.timestamp}]\n{msg.role}: {msg.content}\n")
        return "\n".join(formatted)

    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
class ChatInterface:
    """聊天界面类"""
    def __init__(self):
        self.assistant_manager = AssistantManager()
        self.workflow_manager = WorkflowManager()
        self.chat_manager = ChatManager()

    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        with gr.Blocks(css=CUSTOM_CSS) as interface:
            # 页面标题和GitHub链接
            with gr.Row(elem_classes="header"):
                gr.Markdown(f"# 多助手协作对话系统 v{Config.VERSION}")
                gr.HTML(
                    f'<a href="{Config.GITHUB_REPO}" class="github-link" target="_blank">'
                    '<img src="https://github.com/favicon.ico" width="24px">'
                    '<span style="margin-left:8px">查看源码</span></a>'
                )

            # 系统状态栏
            system_status = gr.Textbox(
                value="系统未初始化",
                label="系统状态",
                interactive=False
            )

            # 主要内容标签页
            with gr.Tabs():
                # 系统配置标签页
                with gr.Tab("系统配置"):
                    with gr.Row():
                        api_key = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="请输入您的API Key"
                        )
                        api_base = gr.Textbox(
                            label="API Base URL",
                            value=Config.DEFAULT_API_BASE,
                            placeholder="API基础地址"
                        )
                        model_select = gr.Radio(
                            choices=["gpt-3.5-turbo", "gpt-4"],
                            label="选择模型",
                            value="gpt-3.5-turbo"
                        )
                    init_btn = gr.Button("初始化系统", variant="primary")

                # 助手管理标签页
                with gr.Tab("助手管理"):
                    with gr.Row():
                        # 左侧：助手列表
                        with gr.Column(scale=1):
                            assistant_list = gr.Radio(
                                choices=self.get_assistant_choices(),
                                label="选择助手",
                                interactive=True
                            )
                            refresh_list_btn = gr.Button("刷新列表")

                        # 右侧：助手配置
                        with gr.Column(scale=2):
                            is_new = gr.Checkbox(label="创建新助手")
                            with gr.Group():
                                assistant_name = gr.Textbox(label="助手名称")
                                assistant_role = gr.Textbox(label="角色")
                                assistant_prompt = gr.Textbox(
                                    label="提示词",
                                    lines=5,
                                    placeholder="请输入助手的系统提示词..."
                                )
                            with gr.Row():
                                save_assistant_btn = gr.Button("保存配置", variant="primary")
                                delete_assistant_btn = gr.Button("删除助手", variant="stop")

                # 工作流设计标签页
                with gr.Tab("工作流设计"):
                    with gr.Row():
                        # 左侧：可选助手
                        with gr.Column(scale=1):
                            available_assistants = gr.CheckboxGroup(
                                choices=self.get_assistant_choices(),
                                label="选择助手"
                            )
                            refresh_workflow_assistants_btn = gr.Button("刷新助手列表")

                        # 右侧：操作按钮
                        with gr.Column(scale=1):
                            with gr.Group():
                                add_parallel_btn = gr.Button("添加为并行步骤")
                                add_sequential_btn = gr.Button("添加为顺序步骤")
                                clear_workflow_btn = gr.Button("清空工作流")

                    # 工作流显示和保存
                    workflow_display = gr.HTML(
                        self.workflow_manager.get_workflow_display(),
                        label="当前工作流"
                    )
                    with gr.Row():
                        workflow_name = gr.Textbox(label="工作流名称")
                        workflow_desc = gr.Textbox(
                            label="工作流描述",
                            placeholder="可选：添加工作流说明..."
                        )
                        save_workflow_btn = gr.Button("保存工作流", variant="primary")

                    # 已保存的工作流
                    with gr.Row():
                        saved_workflows = gr.Dropdown(
                            choices=list(self.workflow_manager.workflows.keys()),
                            label="已保存的工作流",
                            interactive=True
                        )
                        load_workflow_btn = gr.Button("加载")
                        delete_workflow_btn = gr.Button("删除", variant="stop")

                # 对话标签页
                with gr.Tab("对话"):
                    with gr.Row():
                        # 左侧：对话区域
                        with gr.Column(scale=2):
                            # 工作流选择
                            with gr.Row():
                                active_workflow = gr.Dropdown(
                                    choices=list(self.workflow_manager.workflows.keys()),
                                    label="选择工作流",
                                    value=None
                                )
                                refresh_workflow_btn = gr.Button("刷新工作流")
                            
                            # 当前工作流显示
                            current_workflow_display = gr.HTML(
                                value="<div class='workflow-container'><div class='workflow-step'>未选择工作流</div></div>",
                                label="当前工作流"
                            )
                            
                            # 对话区域
                            chat_history = gr.Chatbot(height=400)
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    label="输入消息",
                                    placeholder="请输入您的问题...",
                                    lines=3
                                )
                                send_btn = gr.Button("发送", variant="primary")
                            
                            with gr.Row():
                                clear_btn = gr.Button("清空对话")
                                download_btn = gr.Button("下载对话记录")
                        
                        # 右侧：完整对话记录
                        with gr.Column(scale=1):
                            gr.Markdown("### 完整对话记录")
                            conversation_log = gr.TextArea(
                                label="对话日志",
                                interactive=False,
                                lines=25
                            )
                    
                    file_output = gr.File(label="下载文件", visible=False)

            # 事件处理函数
            def init_system(api_key_value: str, api_base_value: str, model_value: str) -> str:
                """初始化系统"""
                try:
                    if not api_key_value:
                        return "请输入API Key"
                    
                    success = self.chat_manager.initialize(api_key_value, api_base_value)
                    if success:
                        return "系统初始化完成"
                    return "初始化失败，请检查配置"
                except Exception as e:
                    logger.error(f"系统初始化失败: {e}")
                    return f"初始化失败: {str(e)}"

            def refresh_assistant_list():
                """刷新助手列表"""
                choices = self.get_assistant_choices()
                return gr.Radio(choices=choices)

            def refresh_workflow_list():
                """刷新工作流列表"""
                return gr.Dropdown(choices=list(self.workflow_manager.workflows.keys()))

            def load_assistant_config(assistant_id: str, is_new: bool) -> Tuple[str, str, str]:
                """加载助手配置"""
                if is_new or not assistant_id:
                    return "", "", ""
                
                assistant = self.assistant_manager.get_assistant(assistant_id)
                if assistant:
                    return assistant.name, assistant.role, assistant.prompt
                return "", "", ""

            def save_assistant_config(
                is_new: bool,
                name: str,
                role: str,
                prompt: str,
                current_id: str = None
            ) -> Tuple[str, List[str]]:
                """保存助手配置"""
                try:
                    if not all([name, role, prompt]):
                        return "请填写完整信息", self.get_assistant_choices()

                    if is_new:
                        success = self.assistant_manager.add_assistant(name, role, prompt)
                        message = "新助手创建成功" if success else "创建失败"
                    else:
                        current_id = current_id or name.lower().replace(" ", "_")
                        success = self.assistant_manager.update_assistant(
                            current_id,
                            {"name": name, "role": role, "prompt": prompt}
                        )
                        message = "助手配置更新成功" if success else "更新失败"

                    return message, self.get_assistant_choices()
                except Exception as e:
                    logger.error(f"保存助手配置失败: {e}")
                    return f"操作失败: {str(e)}", self.get_assistant_choices()

            def delete_assistant(assistant_id: str) -> Tuple[str, List[str]]:
                """删除助手"""
                try:
                    if not assistant_id:
                        return "请先选择助手", self.get_assistant_choices()
                    
                    success = self.assistant_manager.delete_assistant(assistant_id)
                    message = "删除成功" if success else "删除失败"
                    return message, self.get_assistant_choices()
                except Exception as e:
                    logger.error(f"删除助手失败: {e}")
                    return f"删除失败: {str(e)}", self.get_assistant_choices()

            def load_active_workflow(workflow_name: str) -> Tuple[str, str]:
                """加载选中的工作流"""
                if not workflow_name:
                    return (
                        "<div class='workflow-container'><div class='workflow-step'>未选择工作流</div></div>",
                        "请选择工作流"
                    )
                
                try:
                    success = self.workflow_manager.load_workflow(workflow_name)
                    if success:
                        return (
                            self.workflow_manager.get_workflow_display(),
                            f"已加载工作流: {workflow_name}"
                        )
                    return (
                        "<div class='workflow-container'><div class='workflow-step error-message'>加载失败</div></div>",
                        "工作流加载失败"
                    )
                except Exception as e:
                    logger.error(f"加载工作流失败: {e}")
                    return (
                        "<div class='workflow-container'><div class='workflow-step error-message'>加载错误</div></div>",
                        f"错误: {str(e)}"
                    )

            def format_conversation_log(history: List) -> str:
                """格式化对话记录"""
                if not history:
                    return "暂无对话记录"
                
                formatted = []
                for i, (user_msg, assistant_msg) in enumerate(history, 1):
                    formatted.append(f"[对话 {i}]\n用户: {user_msg}\n")
                    if assistant_msg:
                        formatted.append(f"助手: {assistant_msg}\n")
                
                return "\n".join(formatted)

            async def process_message_with_log(
                message: str,
                history: List,
                workflow_name: str,
                model: str
            ) -> Tuple[List, str, str]:
                """处理消息并更新日志"""
                try:
                    if not message:
                        return history, None, conversation_log.value
                    if not self.chat_manager.client:
                        error_msg = "请先初始化系统"
                        return (
                            history + [[message, error_msg]],
                            None,
                            format_conversation_log(history + [[message, error_msg]])
                        )
                    if not workflow_name:
                        error_msg = "请先选择工作流"
                        return (
                            history + [[message, error_msg]],
                            None,
                            format_conversation_log(history + [[message, error_msg]])
                        )
                    
                    # 添加用户消息到历史记录
                    new_history = history + [[message, None]]
                    
                    # 处理消息
                    response = await self.chat_manager.process_message(
                        message,
                        self.workflow_manager.current_workflow,
                        model
                    )
                    
                    # 更新历史记录
                    new_history[-1][1] = response
                    
                    # 更新对话日志
                    log_text = format_conversation_log(new_history)
                    
                    return new_history, None, log_text
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")
                    error_history = history + [[message, f"错误: {str(e)}"]]
                    return (
                        error_history,
                        None,
                        format_conversation_log(error_history)
                    )

            # 绑定事件
            init_btn.click(
                init_system,
                inputs=[api_key, api_base, model_select],
                outputs=[system_status]
            )
            
            refresh_list_btn.click(
                refresh_assistant_list,
                outputs=[assistant_list]
            )

            refresh_workflow_assistants_btn.click(
                refresh_assistant_list,
                outputs=[available_assistants]
            )

            refresh_workflow_btn.click(
                refresh_workflow_list,
                outputs=[active_workflow]
            )
            
            is_new.change(
                load_assistant_config,
                inputs=[assistant_list, is_new],
                outputs=[assistant_name, assistant_role, assistant_prompt]
            )
            
            assistant_list.change(
                load_assistant_config,
                inputs=[assistant_list, is_new],
                outputs=[assistant_name, assistant_role, assistant_prompt]
            )
            
            save_assistant_btn.click(
                save_assistant_config,
                inputs=[is_new, assistant_name, assistant_role, assistant_prompt, assistant_list],
                outputs=[system_status, assistant_list]
            )
            
            delete_assistant_btn.click(
                delete_assistant,
                inputs=[assistant_list],
                outputs=[system_status, assistant_list]
            )
            
            add_parallel_btn.click(
                lambda x: (
                    "步骤添加成功" if self.workflow_manager.add_step(x, True) else "添加失败",
                    self.workflow_manager.get_workflow_display()
                ),
                inputs=[available_assistants],
                outputs=[system_status, workflow_display]
            )

            add_sequential_btn.click(
                lambda x: (
                    "步骤添加成功" if self.workflow_manager.add_step(x, False) else "添加失败",
                    self.workflow_manager.get_workflow_display()
                ),
                inputs=[available_assistants],
                outputs=[system_status, workflow_display]
            )

            clear_workflow_btn.click(
                lambda: (
                    self.workflow_manager.clear_current_workflow(),
                    self.workflow_manager.get_workflow_display(),
                    "工作流已清空"
                ),
                outputs=[workflow_display, system_status]
            )
            
            save_workflow_btn.click(
                lambda name, desc: (
                    "保存成功" if self.workflow_manager.save_current_workflow(name, desc) else "保存失败",
                    gr.Dropdown(choices=list(self.workflow_manager.workflows.keys()))
                ),
                inputs=[workflow_name, workflow_desc],
                outputs=[system_status, saved_workflows]
            )
            
            active_workflow.change(
                load_active_workflow,
                inputs=[active_workflow],
                outputs=[current_workflow_display, system_status]
            )

            send_btn.click(
                process_message_with_log,
                inputs=[msg_input, chat_history, active_workflow, model_select],
                outputs=[chat_history, file_output, conversation_log]
            )
            
            clear_btn.click(
                lambda: (None, "暂无对话记录"),
                outputs=[chat_history, conversation_log]
            )
            
            download_btn.click(
                lambda: self.chat_manager.save_conversation(),
                outputs=[file_output]
            )

        return interface

    def get_assistant_choices(self) -> List[str]:
        """获取助手选项列表"""
        return [
            f"{assistant.name} ({assistant.role})"
            for assistant in self.assistant_manager.get_all_assistants()
        ]

# 主程序入口
def main():
    """主程序入口"""
    try:
        # 确保必要的目录存在
        Config.ensure_directories()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # 创建并启动界面
        app = ChatInterface()
        interface = app.create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True
        )
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
