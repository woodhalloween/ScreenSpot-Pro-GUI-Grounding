from typing import Dict, List, Optional, Union, Any

class EnhancedComputerUse:
    """
    強化されたコンピュータ操作に関連する関数呼び出しの処理を行うクラス
    ベースのComputerUseを拡張し、UI要素認識の精度を向上させる
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初期化関数
        
        Args:
            cfg: 設定情報を含む辞書
                - display_width_px: ディスプレイの幅（ピクセル）
                - display_height_px: ディスプレイの高さ（ピクセル）
                - ui_elements: オプションでUI要素のリスト（事前知識として使用可能）
        """
        self.cfg = cfg
        self.display_width_px = cfg.get("display_width_px", 1000)
        self.display_height_px = cfg.get("display_height_px", 1000)
        # UI要素の事前知識（オプション）
        self.ui_elements = cfg.get("ui_elements", [])
        # 履歴情報を格納する変数（オプション）
        self.action_history = []
        
        # 関数の定義
        self.function = {
            "name": "computer_use",
            "description": "Perform a computer action by specifying coordinates to interact with UI elements on the screen",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": f"The precise x,y coordinate on the screen where the action should be performed. The coordinate MUST be in pixels, where (0,0) is the top-left corner and ({self.display_width_px},{self.display_height_px}) is the bottom-right corner. Always aim for the CENTER of the UI element you are interacting with."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "right_click", "hover", "left_click"],
                        "default": "click",
                        "description": "The type of action to perform at the specified coordinate. Use 'click' or 'left_click' for standard interaction with buttons, links, and other UI elements."
                    },
                    "element_type": {
                        "type": "string",
                        "enum": ["text", "icon", "button", "input", "menu", "other"],
                        "description": "The type of UI element being interacted with. This helps improve accuracy of interaction."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Optional confidence score (0.0-1.0) for the identified UI element location."
                    }
                },
                "required": ["coordinate"]
            }
        }
        
        # あわせて公式のパラメータスキーマも提供
        self.parameters = {
            "name": "computer_use", 
            "description": f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "scroll",
                            "wait",
                            "terminate",
                        ],
                    },
                    "coordinate": {
                        "type": "array",
                        "description": f"The x,y coordinate on the screen where the action should be performed. The coordinate should be in pixels, where (0,0) is the top-left corner and ({self.display_width_px},{self.display_height_px}) is the bottom-right corner. Always aim for the CENTER of the UI element."
                    }
                },
                "required": ["action"]
            }
        }
    
    def execute(self, args: Dict[str, Any]) -> str:
        """
        関数を実行する
        
        Args:
            args: 関数の引数
                - coordinate: [x, y] 形式の座標
                - action: アクションタイプ（デフォルト: "click"）
                - element_type: UI要素タイプ（オプション）
                - confidence: 信頼度スコア（オプション）
            
        Returns:
            実行結果を示す文字列
        """
        coordinate = args.get("coordinate", [0, 0])
        action_type = args.get("action", "click")
        element_type = args.get("element_type", "other")
        confidence = args.get("confidence", 1.0)
        
        # 標準化: "left_click" を "click" に変換
        if action_type == "left_click":
            action_type = "click"
        
        # 座標の検証
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            return "Error: Coordinate must be a list of two numbers."
        
        x, y = coordinate
        
        # 数値型への変換（文字列などが渡された場合）
        try:
            x = float(x)
            y = float(y)
        except (ValueError, TypeError):
            return "Error: Coordinates must be numeric values."
        
        # 座標範囲の検証
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            # 範囲外の座標は自動的に画面内に調整
            x = max(0, min(x, self.display_width_px))
            y = max(0, min(y, self.display_height_px))
        
        # アクション履歴に追加
        self.action_history.append({
            "action": action_type,
            "coordinate": [x, y],
            "element_type": element_type,
            "confidence": confidence,
            "timestamp": None  # 実際の実装ではタイムスタンプを追加
        })
        
        # 応答メッセージ生成
        action_descriptions = {
            "click": "クリック",
            "double_click": "ダブルクリック",
            "right_click": "右クリック",
            "hover": "ホバー（カーソルを置く）"
        }
        
        action_desc = action_descriptions.get(action_type, action_type)
        element_desc = f"{element_type}要素" if element_type != "other" else "要素"
        
        # ツール呼び出しの応答を返す（この部分はコメントアウト）
        # return f"<tool_call>\n{{\"name\": \"computer_use\", \"arguments\": {{\"action\": \"{action_type}\", \"coordinate\": [{int(x)}, {int(y)}]}}}}\n</tool_call>"
        
        # 通常の応答を返す
        return f"座標 ({int(x)}, {int(y)}) の{element_desc}を{action_desc}しました。"
    
    def format_as_tool_call(self, action_type: str, coordinate: List[float], **kwargs) -> str:
        """
        ツール呼び出し形式でフォーマットする
        
        Args:
            action_type: 実行するアクション
            coordinate: 座標 [x, y]
            **kwargs: その他のオプションパラメータ
            
        Returns:
            ツール呼び出し形式の文字列
        """
        # 整数座標に変換
        x, y = int(coordinate[0]), int(coordinate[1])
        
        # 公式互換形式
        if action_type == "click":
            action_type = "left_click"  # 公式形式との互換性のため
            
        args = {
            "action": action_type,
            "coordinate": [x, y]
        }
        
        # 追加パラメータがあれば追加
        args.update({k: v for k, v in kwargs.items() if v is not None})
        
        import json
        json_args = json.dumps(args)
        
        return f"<tool_call>\n{{\"name\": \"computer_use\", \"arguments\": {json_args}}}\n</tool_call>"
    
    def find_nearest_ui_element(self, target_desc: str, approx_coordinate: List[float] = None) -> Dict[str, Any]:
        """
        指定された説明に最も一致するUI要素を見つける
        
        Args:
            target_desc: 対象UI要素の説明
            approx_coordinate: 概算の座標（オプション）
            
        Returns:
            最も一致するUI要素の情報
        """
        if not self.ui_elements:
            # UI要素の事前知識がない場合
            return None
            
        # 実際の実装では、ここで最適なUI要素を検索するロジックを追加
        # 例: テキストマッチング、距離計算、信頼度スコアリングなど
        
        return None  # 実際の実装ではUI要素を返す


# 元のクラスとの互換性のためのエイリアス
ComputerUse = EnhancedComputerUse 