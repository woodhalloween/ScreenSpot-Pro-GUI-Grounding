import copy
import itertools

import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import datetime


logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)

    args = parser.parse_args()
    return args

def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "cogagent":
        from models.cogagent import CogAgentModel
        model = CogAgentModel()
        model.load_model()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel
        model = SeeClickModel()
        model.load_model()
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel
        model = Qwen1VLModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel
        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen25vl":
        from models.qwen25vl import Qwen25VLModel
        model = Qwen25VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen25vlsft":
        from models.qwen25vlsft import Qwen25VLModel
        model = Qwen25VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel
        model = MiniCPMVModel()
        model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel
        model = InternVLModel()
        model.load_model()
    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel
        model = GPT4XModel()
    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel
        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel
        model = OSAtlas7BModel()
        model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel
        model = UGroundModel()
        model.load_model()
    elif model_type == "fuyu":
        from models.fuyu import FuyuModel
        model = FuyuModel()
        model.load_model()
    elif model_type == "showui":
        from models.showui import ShowUIModel
        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel
        model = AriaUIVLLMModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model
        model = CogAgent24Model()
        model.load_model()

    elif model_type == "seeclick-pro-agent":
        from models.seeclick_pro import SeeClickProAgent
        from models.osatlas7b import OSAtlas7BVLLMModel
        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = SeeClickProAgent(grounder=grounder)
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]  # may be none
    print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report

def parse_task_names(task_arg, screenspot_test):
    """コマンドライン引数からタスク名を解析する"""
    if task_arg == "all":
        return [os.path.splitext(f)[0] for f in os.listdir(screenspot_test) if f.endswith(".json")]
    else:
        return task_arg.split(",")

def parse_instruction_styles(inst_style_arg):
    """コマンドライン引数から指示スタイルを解析する"""
    if inst_style_arg == "all":
        return INSTRUCTION_STYLES
    else:
        return inst_style_arg.split(",")

def parse_languages(language_arg):
    """コマンドライン引数から言語を解析する"""
    if language_arg == "all":
        return LANGUAGES
    else:
        return language_arg.split(",")

def parse_gt_types(gt_type_arg):
    """コマンドライン引数からGTタイプを解析する"""
    if gt_type_arg == "all":
        return GT_TYPES
    else:
        return gt_type_arg.split(",")

def main(args):
    model = build_model(args)
    print("Load model success")
    
    # 結果保存ディレクトリを作成
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    # 途中結果の保存先ファイルパス
    partial_log_path = args.log_path + ".partial"
    
    # 既存の途中結果を読み込む
    if os.path.exists(partial_log_path):
        try:
            with open(partial_log_path, 'r') as f:
                partial_data = json.load(f)
                results = partial_data.get('results', [])
                print(f"Loaded {len(results)} results from partial file")
                
                # 既に処理済みのタスクを識別するIDを収集
                processed_ids = set()
                for r in results:
                    if 'img_path' in r and 'task_filename' in r:
                        task_id = r['task_filename'] + '_' + os.path.basename(r['img_path'])
                        processed_ids.add(task_id)
                print(f"Found {len(processed_ids)} processed task IDs")
                
                # 既に評価メトリクスが計算されていれば表示
                if 'partial_metrics' in partial_data:
                    print("最新の部分評価結果:")
                    print(f"Action Accuracy: {partial_data['partial_metrics']['overall']['action_acc']:.4f}")
                    print(f"Text Accuracy: {partial_data['partial_metrics']['overall']['text_acc']:.4f}")
                    print(f"Icon Accuracy: {partial_data['partial_metrics']['overall']['icon_acc']:.4f}")
                    
        except Exception as e:
            print(f"Error loading partial results: {e}")
            results = []
            processed_ids = set()
    else:
        results = []
        processed_ids = set()
        
    # タスクの読み込みと準備
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")
    
    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        try:
            with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
                task_data = json.load(f)
        except Exception as e:
            print(f"Error loading task data {dataset}: {e}")
            continue

        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        
                        # 既に処理済みかチェック
                        img_filename = task_instance["img_filename"]
                        task_id = task_filename + '_' + os.path.basename(img_filename)
                        if task_id in processed_ids:
                            # print(f"Skipping already processed task: {task_id}")
                            continue
                            
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                continue
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    
    print(f"Total tasks: {len(tasks_to_run) + len(processed_ids)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining tasks: {len(tasks_to_run)}")
    
    # 10件タスク処理ごとのカウンター
    tasks_since_last_eval = 0
    
    for i, sample in enumerate(tqdm(tasks_to_run)):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        print(f"Processing image: {filename}")
        print(f"Full image path: {img_path}")
        print(f"Image exists: {os.path.exists(img_path)}")

        if sample["gt_type"] == "positive":
            response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path, platform=sample["platform"].lower())
        elif sample["gt_type"] == "negative":
            response = model.ground_allow_negative(instruction=sample["prompt_to_evaluate"], image=img_path, platform=sample["platform"].lower())
        # print(response)
        point = response["point"]
        img_size = sample["img_size"]
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
        
        sample_result = {
            "img_path": img_path, 
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"], 
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"], 
            "task_filename": sample["task_filename"], 
            "pred": point_in_pixel, 
            "raw_response": response["raw_response"]
        }
        
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({
                "bbox": sample["bbox"], 
            })
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        sample_result.update({
            "correctness": correctness,
        })
        results.append(sample_result)
        tasks_since_last_eval += 1
        
        # 10タスクごとに途中結果とメトリクスを保存
        if tasks_since_last_eval >= 10 or i == len(tasks_to_run) - 1:
            tasks_since_last_eval = 0
            try:
                # 確実に保存するためにディレクトリを作成
                os.makedirs(os.path.dirname(partial_log_path), exist_ok=True)
                
                # 部分的な評価結果を計算
                partial_metrics = evaluate(results)
                
                # 結果を文字列としてフォーマット
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metrics_summary = f"=== メトリクス更新: {current_time} (処理済み: {len(results)}/{len(tasks_to_run) + len(processed_ids)}) ===\n"
                metrics_summary += f"全体精度: {partial_metrics['metrics']['overall']['action_acc']:.4f}\n"
                metrics_summary += f"テキスト精度: {partial_metrics['metrics']['overall']['text_acc']:.4f}\n"
                metrics_summary += f"アイコン精度: {partial_metrics['metrics']['overall']['icon_acc']:.4f}\n"
                
                # wrong_formatの割合を計算
                wrong_format_rate = partial_metrics['metrics']['overall']['wrong_format_num'] / partial_metrics['metrics']['overall']['num_total'] if partial_metrics['metrics']['overall']['num_total'] > 0 else 0
                metrics_summary += f"Wrong Format率: {wrong_format_rate:.4f} ({partial_metrics['metrics']['overall']['wrong_format_num']}/{partial_metrics['metrics']['overall']['num_total']})\n"
                
                # 最新のスコアを表示
                print("\n" + metrics_summary)
                
                # 部分的な結果とメトリクスを保存
                with open(partial_log_path, 'w') as f:
                    # wrong_format_rateをメトリクスに追加
                    partial_metrics['metrics']['overall']['wrong_format_rate'] = wrong_format_rate
                    
                    json.dump({
                        'results': results, 
                        'timestamp': str(datetime.datetime.now()),
                        'partial_metrics': partial_metrics['metrics'],
                        'metrics_summary': metrics_summary
                    }, f, indent=4)
                print(f"Saved partial results and metrics ({len(results)} items) to {partial_log_path}")
                
                # プラットフォーム別の精度を表示（オプション）
                if 'seeclick_style' in partial_metrics['metrics']:
                    print("\nプラットフォーム別精度:")
                    for key, value in partial_metrics['metrics']['seeclick_style'].items():
                        if 'plat:' in key and 'gt_type:positive' in key:
                            platform = key.split('plat:')[1].split(' ')[0]
                            print(f"{platform}: {value['action_acc']:.4f} ({value['num_correct_action']}/{value['num_total']})")
                
            except Exception as e:
                print(f"Error saving partial results and metrics: {e}")
                import traceback
                traceback.print_exc()

    # 最終評価と保存
    result_report = evaluate(results)
    # Save to file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpot finished.")


if __name__ == "__main__":
    main(parse_args())
