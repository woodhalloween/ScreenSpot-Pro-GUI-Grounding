from typing import Union, Tuple, List
import time

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        """キーイベントを実行する"""
        return f"キーイベント '{text}' を実行しました"
        
    def _click(self, coordinate: Tuple[int, int]):
        """画面上の座標をクリックする"""
        x, y = coordinate
        # 座標の有効性をチェック
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            return f"エラー: 座標 ({x}, {y}) は画面の範囲外です (0, 0) から ({self.display_width_px}, {self.display_height_px})"
            
        return f"座標 ({x}, {y}) をクリックしました"

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        """指定した座標を長押しする"""
        x, y = coordinate
        # 座標の有効性をチェック
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            return f"エラー: 座標 ({x}, {y}) は画面の範囲外です"
            
        return f"座標 ({x}, {y}) を {time} 秒間長押ししました"

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        """ある座標から別の座標にスワイプする"""
        x1, y1 = coordinate
        x2, y2 = coordinate2
        # 座標の有効性をチェック
        if (x1 < 0 or x1 > self.display_width_px or y1 < 0 or y1 > self.display_height_px or
            x2 < 0 or x2 > self.display_width_px or y2 < 0 or y2 > self.display_height_px):
            return f"エラー: 座標のいずれかが画面の範囲外です"
            
        return f"座標 ({x1}, {y1}) から ({x2}, {y2}) にスワイプしました"

    def _type(self, text: str):
        """テキストを入力する"""
        return f"テキスト '{text}' を入力しました"

    def _system_button(self, button: str):
        """システムボタンを押す"""
        return f"システムボタン '{button}' を押しました"

    def _open(self, text: str):
        """アプリを開く"""
        return f"アプリ '{text}' を開きました"

    def _wait(self, time: int):
        """指定した秒数待機する"""
        # 実際には待機処理を実装
        return f"{time} 秒間待機しました"

    def _terminate(self, status: str):
        """タスクを終了する"""
        return f"タスクを '{status}' ステータスで終了しました"
    
@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
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
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        # 現在のマウス座標を保持
        self.current_mouse_x = 0
        self.current_mouse_y = 0
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        """マウスのボタンをクリックする"""
        button_names = {
            "left_click": "左",
            "right_click": "右",
            "middle_click": "中",
            "double_click": "左（ダブル）"
        }
        button_name = button_names.get(button, "不明")
        
        # マウスが画面内にあるかチェック
        if (self.current_mouse_x < 0 or self.current_mouse_x > self.display_width_px or 
            self.current_mouse_y < 0 or self.current_mouse_y > self.display_height_px):
            return f"エラー: マウスの位置 ({self.current_mouse_x}, {self.current_mouse_y}) が画面の範囲外です"
            
        return f"マウスの{button_name}ボタンを座標 ({self.current_mouse_x}, {self.current_mouse_y}) でクリックしました"

    def _key(self, keys: List[str]):
        """キーボードのキーを押す"""
        return f"キー {', '.join(keys)} を押しました"

    def _type(self, text: str):
        """テキストを入力する"""
        return f"テキスト '{text}' を入力しました"

    def _mouse_move(self, coordinate: Tuple[int, int]):
        """マウスカーソルを移動する"""
        x, y = coordinate
        # 座標の有効性をチェック
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            return f"エラー: 座標 ({x}, {y}) は画面の範囲外です (0, 0) から ({self.display_width_px}, {self.display_height_px})"
        
        # マウス位置を更新
        self.current_mouse_x = x
        self.current_mouse_y = y
        return f"マウスカーソルを座標 ({x}, {y}) に移動しました"

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        """マウスをドラッグする"""
        x, y = coordinate
        # 座標の有効性をチェック
        if x < 0 or x > self.display_width_px or y < 0 or y > self.display_height_px:
            return f"エラー: 座標 ({x}, {y}) は画面の範囲外です"
        
        start_x, start_y = self.current_mouse_x, self.current_mouse_y
        # マウス位置を更新
        self.current_mouse_x = x
        self.current_mouse_y = y
        return f"左クリックを押しながら座標 ({start_x}, {start_y}) から ({x}, {y}) にドラッグしました"

    def _scroll(self, pixels: int):
        """マウスホイールをスクロールする"""
        direction = "上" if pixels > 0 else "下"
        return f"マウスホイールを{direction}に {abs(pixels)} ピクセルスクロールしました"

    def _wait(self, time: int):
        """指定した秒数待機する"""
        # 実際には待機処理を実装
        return f"{time} 秒間待機しました"

    def _terminate(self, status: str):
        """タスクを終了する"""
        return f"タスクを '{status}' ステータスで終了しました" 