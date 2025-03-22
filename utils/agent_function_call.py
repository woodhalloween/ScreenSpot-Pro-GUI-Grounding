from typing import Dict, List, Optional, Union, Any

class ComputerUse:
    """
    コンピュータ操作に関連する関数呼び出しの処理を行うクラス
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        初期化関数
        
        Args:
            cfg: 設定情報を含む辞書（display_width_px, display_height_pxなど）
        """
        self.cfg = cfg
        self.display_width_px = cfg.get("display_width_px", 1000)
        self.display_height_px = cfg.get("display_height_px", 1000)
        
        # 関数の定義
        self.function = {
            "name": "computer_use",
            "description": "Perform a computer action by specifying coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": f"The x,y coordinate on the screen where the action should be performed. The coordinate should be in pixels, where (0,0) is the top-left corner and ({self.display_width_px},{self.display_height_px}) is the bottom-right corner."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "right_click", "hover"],
                        "default": "click",
                        "description": "The type of action to perform at the specified coordinate."
                    }
                },
                "required": ["coordinate"]
            }
        }
    
    def execute(self, args: Dict[str, Any]) -> str:
        """
        関数を実行する
        
        Args:
            args: 関数の引数
            
        Returns:
            実行結果を示す文字列
        """
        coordinate = args.get("coordinate", [0, 0])
        action_type = args.get("action", "click")
        
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            return "Error: Coordinate must be a list of two numbers."
        
        x, y = coordinate
        
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            return f"Error: Coordinate ({x}, {y}) is outside the display bounds (0, 0) to ({self.display_width_px}, {self.display_height_px})."
        
        return f"Successfully performed {action_type} at coordinate ({x}, {y})." 