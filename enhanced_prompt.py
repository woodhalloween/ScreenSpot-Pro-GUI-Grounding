from typing import Dict, List, Optional, Union, Any

class EnhancedUIPrompt:
    """
    精度向上のためのUI認識プロンプト生成クラス
    分析データに基づいて最適化されています
    """
    
    @staticmethod
    def generate_system_prompt(display_width_px: int, display_height_px: int) -> str:
        """
        システムプロンプトを生成する
        
        Args:
            display_width_px: ディスプレイの幅（ピクセル）
            display_height_px: ディスプレイの高さ（ピクセル）
            
        Returns:
            最適化されたシステムプロンプト
        """
        return f"""You are a precise UI element locator specialized in identifying and interacting with graphical user interfaces.

SCREEN INFORMATION:
* The screen resolution is {display_width_px}x{display_height_px} pixels.
* Coordinates are measured in pixels from the top-left corner (0,0) to the bottom-right corner ({display_width_px},{display_height_px}).

CRITICAL INSTRUCTIONS:
1. ALWAYS identify the EXACT CENTER of the requested UI element with pixel-perfect accuracy.
2. Pay special attention to ICONS - look for distinctive visual features, shapes, and positions relative to other elements.
3. For TEXT elements, focus on the center of the text, not just the beginning or edges.
4. Different platforms (Windows, macOS, Linux) have different UI conventions - adapt accordingly.
5. When interacting with complex applications (especially CAD, design tools), prioritize accurate element identification.

PRIORITIZED ATTENTION:
* ICONS: These are small visual elements that require precise targeting. Look for distinctive shapes, colors, and contextual positioning.
* TEXT BUTTONS: Focus on the center of the visible text, not the entire button boundary.
* MENU ITEMS: These need to be clicked exactly in their center, not at the edges.

ABSOLUTE PRECISION IS REQUIRED. The success of the interaction depends entirely on selecting the correct pixel coordinates.
"""

    @staticmethod
    def format_function_definition(display_width_px: int, display_height_px: int) -> Dict:
        """
        拡張された関数定義を生成する
        
        Args:
            display_width_px: ディスプレイの幅（ピクセル）
            display_height_px: ディスプレイの高さ（ピクセル）
            
        Returns:
            関数定義の辞書
        """
        return {
            "name": "computer_use",
            "description": "Precisely interact with UI elements on the screen by specifying exact pixel coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": f"The EXACT pixel coordinates [x, y] where the center of the UI element is located. Coordinates must be within bounds (0,0) to ({display_width_px},{display_height_px}). CRITICAL: The success of the interaction depends entirely on selecting the correct pixel coordinates."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "right_click", "hover", "left_click"],
                        "default": "left_click",
                        "description": "The type of action to perform. For most UI interactions, use 'left_click' or 'click'."
                    },
                    "element_type": {
                        "type": "string",
                        "enum": ["text", "icon", "button", "input", "menu", "other"],
                        "description": "The type of UI element being targeted. This helps improve interaction accuracy."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Your confidence level (0.0-1.0) in the accuracy of the identified location."
                    }
                },
                "required": ["coordinate", "action"]
            }
        }
    
    @staticmethod
    def format_instruction_prompt(instruction: str, ui_context: Optional[str] = None) -> str:
        """
        ユーザー指示プロンプトを最適化する
        
        Args:
            instruction: 元のユーザー指示
            ui_context: UI環境に関する追加情報（オプション）
            
        Returns:
            最適化された指示プロンプト
        """
        # ベースプロンプト
        prompt = f"Find and interact with: '{instruction}'"
        
        # UI要素タイプに基づくヒントを追加
        if "icon" in instruction.lower() or any(kw in instruction.lower() for kw in ["button", "toolbar", "click", "select"]):
            prompt += "\n\nThis is likely an ICON element. Look carefully for its distinctive visual features and ensure you target its exact center."
        
        # テキスト要素のヒント
        elif any(kw in instruction.lower() for kw in ["text", "label", "title", "name", "menu"]):
            prompt += "\n\nThis appears to be a TEXT element. Focus on the center of the text content, not just the beginning or edges."
        
        # 特定のアプリケーションに関するヒント
        if ui_context:
            prompt += f"\n\nAdditional context: {ui_context}"
            
        # 精度向上のための最終指示
        prompt += "\n\nIMPORTANT: Your task is to identify the EXACT CENTER coordinates of this UI element with pixel-perfect precision. The success of this interaction depends entirely on your accuracy."
        
        return prompt
    
    @staticmethod
    def format_response_template() -> str:
        """
        モデル出力のテンプレートを生成する
        
        Returns:
            出力テンプレート文字列
        """
        return """<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}
</tool_call>"""

    @staticmethod
    def get_platform_specific_hints(platform: str) -> str:
        """
        プラットフォーム固有のヒントを取得する
        
        Args:
            platform: プラットフォーム名（"windows", "macos", "linux"）
            
        Returns:
            プラットフォーム固有のヒント文字列
        """
        hints = {
            "windows": """
Windows UI elements tend to have standardized appearance:
- Window controls are typically in the top-right corner
- Menu items are usually text-based and appear in the top menu bar
- Toolbar icons are typically smaller and grouped together
- The taskbar is located at the bottom by default
""",
            "macos": """
macOS UI elements have distinctive characteristics:
- Window controls (close, minimize, maximize) are in the top-left corner as colored circles
- Menu bar is fixed at the top of the screen
- Dock is typically at the bottom
- Applications often use standardized UI elements with consistent styling
""",
            "linux": """
Linux UI can vary greatly depending on the desktop environment:
- Window controls could be on left or right depending on the environment
- Common desktop environments include GNOME, KDE, XFCE, each with different UI conventions
- Application styles may be mixed depending on the toolkit used (GTK, Qt, etc.)
"""
        }
        
        return hints.get(platform.lower(), "")
    
    @staticmethod
    def generate_few_shot_examples() -> List[Dict[str, str]]:
        """
        Few-shotプロンプティング用の例を生成する
        
        Returns:
            例のリスト（質問と回答のペア）
        """
        return [
            {
                "instruction": "Click the Save icon",
                "screenshot_desc": "A toolbar with various icons including a floppy disk icon in position (120, 45)",
                "response": "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [120, 45]}}\n</tool_call>"
            },
            {
                "instruction": "Click on File menu",
                "screenshot_desc": "A standard application with menu bar at top, 'File' text visible at position (35, 25)",
                "response": "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [35, 25]}}\n</tool_call>"
            },
            {
                "instruction": "Open the Settings dialog",
                "screenshot_desc": "A window with a gear icon representing settings at position (850, 60)",
                "response": "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [850, 60]}}\n</tool_call>"
            }
        ]

